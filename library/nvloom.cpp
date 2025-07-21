/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <iomanip>
#include <string.h>
#include <unistd.h>
#include <set>
#include "error.h"
#include "kernels.cuh"
#include "nvloom.h"
#include <cuda.h>
#define NVML_NO_UNVERSIONED_FUNC_DEFS
#include <nvml.h>

auto getCopiesWithUniqueSources(std::vector<Copy> copies) {
    auto comp = [](Copy a, Copy b) {return a.src.get() < b.src.get();};
    std::set<Copy, decltype(comp) > uniqueSources(comp);
    for (auto &copy : copies) {
        uniqueSources.insert(copy);
    }
    return uniqueSources;
}

auto getCopiesWithUniqueDestinations(std::vector<Copy> copies) {
    auto comp = [](Copy a, Copy b) {return a.dst.get() < b.dst.get();};
    std::set<Copy, decltype(comp) > uniqueDestinations(comp);
    for (auto &copy : copies) {
        uniqueDestinations.insert(copy);
    }
    return uniqueDestinations;
}

std::vector<double> NvLoom::doBenchmark(std::vector<Copy> copies) {
    if (copies.size() == 0) {return {};};

    std::vector<double> results;

    std::vector<Copy> filteredCopies;
    for (auto copy : copies) {
        if (MPIWrapper::getWorldRank() == copy.executingMPIrank) {
            filteredCopies.push_back(copy);
        }
    }

    std::vector<CUstream> filteredStreams(filteredCopies.size());
    std::vector<CUevent> filteredStartEvents(filteredCopies.size());
    std::vector<CUevent> filteredEndEvents(filteredCopies.size());
    std::vector<double> filteredBandwidths(filteredCopies.size());
    std::vector<int> filteredExecutedIterations(filteredCopies.size());

    AllocationPool<HostMemoryAllocation> blockingVarHost(sizeof(int), copies[0].executingMPIrank);
    AllocationPool<MultinodeMemoryAllocationUnicast> blockingVarDevice(sizeof(int), copies[0].executingMPIrank);

    // init
    if (MPIWrapper::getWorldRank() == copies[0].executingMPIrank) {
        *((int *) (blockingVarHost.ptr)) = 0;
        CU_ASSERT(cuMemsetD32((CUdeviceptr) blockingVarDevice.ptr, 0, 1));
    }

    for (int i = 0; i < filteredCopies.size(); i++) {
        CU_ASSERT(cuStreamCreate(&filteredStreams[i], CU_STREAM_NON_BLOCKING));

        CU_ASSERT(cuEventCreate(&filteredStartEvents[i], CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&filteredEndEvents[i], CU_EVENT_DEFAULT));
    }

    // fill buffers
    for (auto &copy: getCopiesWithUniqueSources(copies)) {
        copy.src->memset(copy.src->uniqueId, copy.copyType, MemoryPurpose::MEMORY_SOURCE);
    }

    for (auto &copy: getCopiesWithUniqueDestinations(copies)) {
        copy.dst->memset(copy.dst->uniqueId, copy.copyType, MemoryPurpose::MEMORY_DESTINATION);
    }

    // Make sure all buffers are memseted before proceeding
    CU_ASSERT(cuCtxSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // warmup
    for (int i = 0; i < filteredCopies.size(); i++) {
        Copy &copy = filteredCopies[i];
        doMemcpy(copy.copyType, (CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, filteredStreams[i], WARMUP_ITERATIONS);
    }

    // block streams
    for (int i = 0; i < filteredCopies.size(); i++) {
        CU_ASSERT(spinKernelMultistage((MPIWrapper::getWorldRank() == copies[0].executingMPIrank) ? (volatile int *) blockingVarHost.ptr : nullptr,
                                        (volatile int *) blockingVarDevice.ptr,
                                        filteredStreams[i]));
    }

    // schedule work
    for (int i = 0; i < filteredCopies.size(); i++) {
        Copy &copy = filteredCopies[i];
        CU_ASSERT(cuEventRecord(filteredStartEvents[i], filteredStreams[i]));
        filteredExecutedIterations[i] = doMemcpyInSpinKernel(copy.copyType, (CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, filteredStreams[i], copy.iterations);
        if (filteredExecutedIterations[i] == copy.iterations) {
            CU_ASSERT(cuEventRecord(filteredEndEvents[i], filteredStreams[i]));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // release work
    if (MPIWrapper::getWorldRank() == copies[0].executingMPIrank) {
        *((int *) (blockingVarHost.ptr)) = 1;
    }

    int maxIterations = 0;
    for (auto &copy : filteredCopies) {
        maxIterations = std::max(maxIterations, copy.iterations);
    }

    // keep adding copies to streams, one at a time, in a round robin fashion
    bool pendingWork = true;
    while (pendingWork) {
        pendingWork = false;
        for (int i = 0; i < filteredCopies.size(); i++) {
            Copy &copy = filteredCopies[i];
            if (filteredExecutedIterations[i] < copy.iterations) {
                doMemcpyBeyondSpinKernel(copy.copyType, (CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, filteredStreams[i]);
                filteredExecutedIterations[i]++;
                pendingWork = true;
            }
            if (filteredExecutedIterations[i] == copy.iterations) {
                CU_ASSERT(cuEventRecord(filteredEndEvents[i], filteredStreams[i]));
            }
        }
    }

    // Make sure all the work is finished
    CU_ASSERT(cuCtxSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // calculate bandwidths
    for (int i = 0; i < filteredCopies.size(); i++) {
        float elapsedMs;
        CU_ASSERT(cuEventElapsedTime(&elapsedMs, filteredStartEvents[i], filteredEndEvents[i]));

        filteredBandwidths[i] = filteredCopies[i].src->allocationSize * filteredCopies[i].iterations / (1000 * 1000 * elapsedMs);
    }

    // verify buffers
    for (auto &copy: getCopiesWithUniqueDestinations(copies)) {
        ASSERT(0 == copy.dst->check(copy.src->uniqueId, copy.copyType, WARMUP_ITERATIONS + copy.iterations));
    }

    // exchange bandwidths
    int currentIndex = 0;
    for (auto copy : copies) {
        double exchange;
        if (MPIWrapper::getWorldRank() == copy.executingMPIrank) {
            exchange = filteredBandwidths[currentIndex];
            currentIndex++;
        }

        MPI_Bcast(&exchange, 1, MPI_DOUBLE, copy.executingMPIrank, MPI_COMM_WORLD);
        results.push_back(exchange);
    }

    for (int i = 0; i < filteredCopies.size(); i++) {
        CU_ASSERT(cuStreamDestroy(filteredStreams[i]));
        CU_ASSERT(cuEventDestroy(filteredStartEvents[i]));
        CU_ASSERT(cuEventDestroy(filteredEndEvents[i]));
    }

    return results;
}

void NvLoom::doMemcpy(CopyType copyType, CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream, unsigned long long loopCount) {
    if (copyType == COPY_TYPE_CE) {
        for (int iter = 0; iter < loopCount; iter++) {
            CU_ASSERT(cuMemcpyAsync(dst, src, byteCount, hStream));
        }
    } else if (copyType == COPY_TYPE_SM) {
        copyKernel(dst, src, byteCount, hStream, loopCount);
    } else if (copyType == COPY_TYPE_MULTICAST_WRITE) {
        copyKernelMulticast(dst, src, byteCount, hStream, loopCount);
    } else if (copyType == COPY_TYPE_MULTICAST_LD_REDUCE) {
        copyKernelMulticastLdReduce(dst, src, byteCount, hStream, loopCount);
    } else if (copyType == COPY_TYPE_MULTICAST_RED_ALL || copyType == COPY_TYPE_MULTICAST_RED_SINGLE) {
        copyKernelMulticastRed(dst, src, byteCount, hStream, loopCount);
    } else {
        ASSERT(0);
    }
}

unsigned long long NvLoom::doMemcpyInSpinKernel(CopyType copyType, CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream, unsigned long long loopCount) {
    if (copyType == COPY_TYPE_CE) {
        // We're only launching 128 iterations in the spinLock guarded loop to avoid possibility of a deadlock
        loopCount = std::min(loopCount, (unsigned long long) 128);
        for (int iter = 0; iter < loopCount; iter++) {
            CU_ASSERT(cuMemcpyAsync(dst, src, byteCount, hStream));
        }
    } else if (copyType == COPY_TYPE_SM) {
        copyKernel(dst, src, byteCount, hStream, loopCount);
    } else if (copyType == COPY_TYPE_MULTICAST_WRITE) {
        copyKernelMulticast(dst, src, byteCount, hStream, loopCount);
    } else if (copyType == COPY_TYPE_MULTICAST_LD_REDUCE) {
        copyKernelMulticastLdReduce(dst, src, byteCount, hStream, loopCount);
    } else if (copyType == COPY_TYPE_MULTICAST_RED_ALL || copyType == COPY_TYPE_MULTICAST_RED_SINGLE) {
        copyKernelMulticastRed(dst, src, byteCount, hStream, loopCount);
    } else {
        ASSERT(0);
    }
    return loopCount;
}

void NvLoom::doMemcpyBeyondSpinKernel(CopyType copyType, CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream) {
    if (copyType == COPY_TYPE_CE) {
        CU_ASSERT(cuMemcpyAsync(dst, src, byteCount, hStream));
    }
}

// Utility function to convert CUDA format to NVML format
static std::string convertUuid(CUuuid& cuUuid)
{
    uint8_t *uuidPtr = (uint8_t *) &cuUuid;
    std::stringstream s;
    s << "GPU-";
    for (int i = 0; i < sizeof(CUuuid); i++) {
        s << std::setfill('0') << std::setw(2) << std::hex << (int) uuidPtr[i];
        if (std::set<int>{3,5,7,9}.count(i)) {
            s << "-";
        }
    }
    return s.str();
}

static nvmlDevice_t getNvmlDevice(int device) {
    NVML_ASSERT(nvmlInit());
    CU_ASSERT(cuInit(0));

    CUuuid uuid;
    CU_ASSERT(cuDeviceGetUuid_v2(&uuid, device));

    std::string nvmlUuid = convertUuid(uuid);

    nvmlDevice_t nvmlDev;
    NVML_ASSERT(nvmlDeviceGetHandleByUUID((const char *) nvmlUuid.c_str(), &nvmlDev));
    return nvmlDev;
}

static int checkCliques() {
    nvmlDevice_t nvmlDev = getNvmlDevice(NvLoom::getLocalDevice());

    nvmlGpuFabricInfoV_t fabricInfo { .version = nvmlGpuFabricInfo_v2 };
    fabricInfo.state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;
    NVML_ASSERT(nvmlDeviceGetGpuFabricInfoV(nvmlDev, &fabricInfo));

    // allowing running without MNNVL for development purposes
    if (fabricInfo.state == NVML_GPU_FABRIC_STATE_NOT_SUPPORTED) {
        OUTPUT << "WARNING: MNNVL fabric not available. Only single node operation available." << std::endl;
    }

    auto cliqueIdArray = std::vector<unsigned int>(MPIWrapper::getWorldSize());
    MPI_Allgather(&fabricInfo.cliqueId, 1, MPI_UNSIGNED, &cliqueIdArray[0], 1, MPI_UNSIGNED, MPI_COMM_WORLD);

    auto clusterUuidArray = std::string(MPIWrapper::getWorldSize() * NVML_GPU_FABRIC_UUID_LEN, 0);
    MPI_Allgather(&fabricInfo.clusterUuid, NVML_GPU_FABRIC_UUID_LEN, MPI_UNSIGNED_CHAR, &clusterUuidArray[0], NVML_GPU_FABRIC_UUID_LEN, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);

    for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
        if (0 != memcmp(&fabricInfo.clusterUuid, &clusterUuidArray[i * NVML_GPU_FABRIC_UUID_LEN], NVML_GPU_FABRIC_UUID_LEN)) {
            std::cerr << "Process " << MPIWrapper::getWorldRank() << " clusterUuid=" << ((unsigned long *)&fabricInfo.clusterUuid)[0] << ";" << ((unsigned long *)&fabricInfo.clusterUuid)[1] <<
                " is different than process " << i << " clusterUuid=" << ((unsigned long *)&clusterUuidArray[i * NVML_GPU_FABRIC_UUID_LEN])[0] << ";" <<  ((unsigned long *)&clusterUuidArray[i * NVML_GPU_FABRIC_UUID_LEN])[1] << std::endl;
            ASSERT(0);
        }

        if (cliqueIdArray[i] != fabricInfo.cliqueId) {
            std::cerr << "Process " << MPIWrapper::getWorldRank() << " cliqueId=" << fabricInfo.cliqueId << " is different than process " << i << " cliqueId=" << cliqueIdArray[i] << std::endl;
            ASSERT(0);
        }
    }

    return 0;
}

std::string trimRackGuid(std::string rackGuid) {
    if (rackGuid.find('\0') != std::string::npos) {
        rackGuid.resize(rackGuid.find('\0'));
    }
    return rackGuid;
}

std::string getRackGuid(int device) {
    nvmlDevice_t nvmlDev = getNvmlDevice(device);
    nvmlPlatformInfo_v2_t platformInfo;

    platformInfo.version = nvmlPlatformInfo_v2;
    NVML_ASSERT(nvmlDeviceGetPlatformInfo(nvmlDev, &platformInfo));
    std::string rackGuid((char *) &platformInfo.chassisSerialNumber, sizeof(platformInfo.chassisSerialNumber));
    return rackGuid;
}

static std::string getHostname() {
#define HOSTNAME_LENGTH 128
    char _hostname[HOSTNAME_LENGTH] = {};
    gethostname(_hostname, HOSTNAME_LENGTH - 1);
#undef HOSTNAME_LENGTH
    return std::string(_hostname);
}

void NvLoom::initialize(int _localDevice, std::map<std::string, std::vector<int> > _rackToProcessMap) {
    localDevice = _localDevice;
    rackToProcessMap = _rackToProcessMap;
    CU_ASSERT(cuInit(0));

    CU_ASSERT(cuDeviceGet(&localCuDevice, localDevice));
    CU_ASSERT(cuDevicePrimaryCtxRetain(&localCtx, localCuDevice));
    CU_ASSERT(cuCtxSetCurrent(localCtx));

    CU_ASSERT(cuDeviceGetAttribute(&localCpuNumaNode, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, localCuDevice));
    CU_ASSERT(cuDeviceGetAttribute(&localMultiprocessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, localCuDevice));
    CU_ASSERT(cuDeviceGetAttribute(&localClockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, localCuDevice));

    localHostname = getHostname();

    preloadKernels(localDevice);
    MPIWrapper::getWorldRank();
    checkCliques();
};

void NvLoom::finalize() {
    int initialized;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Finalize();
    }
};

MPIOutput OUTPUT;
