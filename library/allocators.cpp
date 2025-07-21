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

#include <cuda.h>
#include <chrono>
#include <thread>
#include "error.h"
#include "kernels.cuh"
#include "nvloom.h"
#include "enums.h"

unsigned long long MemoryAllocation::check(int value, CopyType copyType, int iterations) {
    if (MPIrank == MPIWrapper::getWorldRank()) {
        return checkBuffer(ptr, value, allocationSize, CU_STREAM_PER_THREAD, copyType, iterations);
    }
    return 0;
}

void MemoryAllocation::memset(int value, CopyType copyType, MemoryPurpose memoryPurpose) {
    if (MPIrank == MPIWrapper::getWorldRank()) {
        memsetBuffer(ptr, value, allocationSize, CU_STREAM_PER_THREAD, copyType, memoryPurpose);
    }
}

DeviceMemoryAllocation::DeviceMemoryAllocation(size_t _allocationSize, int _MPIrank) {
    MPIrank = _MPIrank;
    if (MPIrank == MPIWrapper::getWorldRank()) {
        allocationSize = _allocationSize;
        CU_ASSERT(cuMemAlloc((CUdeviceptr *) &ptr, _allocationSize));
    }
}

DeviceMemoryAllocation::~DeviceMemoryAllocation() {
    if (MPIrank == MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuMemFree((CUdeviceptr) ptr));
    }
}

HostMemoryAllocation::HostMemoryAllocation(size_t _allocationSize, int _MPIrank) {
    MPIrank = _MPIrank;
    if (MPIrank == MPIWrapper::getWorldRank()) {
        allocationSize = _allocationSize;
        CU_ASSERT(cuMemAllocHost(&ptr, _allocationSize));
    }
}

HostMemoryAllocation::~HostMemoryAllocation() {
    if (MPIrank == MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuMemFreeHost(ptr));
    }
}

static size_t roundUp(size_t number, size_t multiple) {
    return ((number + multiple - 1) / multiple) * multiple;
}

MultinodeMemoryAllocationBase::MultinodeMemoryAllocationBase(size_t _allocationSize, int _MPIrank, CUmemLocationType location) {
    handleType = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = handleType;

    prop.location.type = location;
    if (location == CU_MEM_LOCATION_TYPE_DEVICE) {
        prop.location.id = NvLoom::getLocalDevice();
    } else {
        prop.location.id = NvLoom::getLocalCpuNumaNode();
    }

    size_t granularity = 0;
    CU_ASSERT(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    roundedUpAllocationSize = roundUp(_allocationSize, granularity);

    if (_MPIrank == MPIWrapper::getWorldRank()) {
        // Allocate the memory
        CU_ASSERT(cuMemCreate(&handle, roundedUpAllocationSize, &prop, 0 /*flags*/));

        // Export the allocation to the importing process
        CU_ASSERT(cuMemExportToShareableHandle(&fh, handle, handleType, 0 /*flags*/));
    }

    MPI_Bcast(&fh, sizeof(fh), MPI_BYTE, _MPIrank, MPI_COMM_WORLD);

    if (_MPIrank != MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuMemImportFromShareableHandle(&handle, (void *)&fh, handleType));
    }

    // Map the memory
    CU_ASSERT(cuMemAddressReserve((CUdeviceptr *) &ptr, roundedUpAllocationSize, 0, 0 /*baseVA*/, 0 /*flags*/));

    CU_ASSERT(cuMemMap((CUdeviceptr) ptr, roundedUpAllocationSize, 0 /*offset*/, handle, 0 /*flags*/));

    // Map EGM memory on host on the exporting node
    if ((_MPIrank == MPIWrapper::getWorldRank()) && (location == CU_MEM_LOCATION_TYPE_HOST_NUMA)) {
        desc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        desc.location.id = 0;
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_ASSERT(cuMemSetAccess((CUdeviceptr) ptr, roundedUpAllocationSize, &desc, 1 /*count*/));
    }

    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = NvLoom::getLocalDevice();
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_ASSERT(cuMemSetAccess((CUdeviceptr) ptr, roundedUpAllocationSize, &desc, 1 /*count*/));

    MPIrank = _MPIrank;
    allocationSize = _allocationSize;

    // Make sure that everyone is done with mapping the fabric allocation
    MPI_Barrier(MPI_COMM_WORLD);
}

MultinodeMemoryAllocationBase::~MultinodeMemoryAllocationBase() {
    // Make sure that everyone is done using the memory
    MPI_Barrier(MPI_COMM_WORLD);

    CU_ASSERT(cuMemUnmap((CUdeviceptr) ptr, roundedUpAllocationSize));
    CU_ASSERT(cuMemRelease(handle));
    CU_ASSERT(cuMemAddressFree((CUdeviceptr) ptr, roundedUpAllocationSize));
}

bool MultinodeMemoryAllocationUnicast::filter() {
    int pi;
    CU_ASSERT(cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, NvLoom::getLocalCuDevice()));
    return pi != 0;
};

bool MultinodeMemoryAllocationEGM::filter() {
    int pi;
    CU_ASSERT(cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED, NvLoom::getLocalCuDevice()));
    return pi != 0;
};

CUresult MultinodeMemoryAllocationMulticast::tryAllocation(size_t _allocationSize, int _MPIrank) {
    handleType = CU_MEM_HANDLE_TYPE_FABRIC;
    multicastProp.numDevices = MPIWrapper::getWorldSize();
    multicastProp.handleTypes = handleType;
    size_t gran;
    CU_ASSERT(cuMulticastGetGranularity(&gran, &multicastProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    roundedUpAllocationSize = roundUp(_allocationSize, gran);
    multicastProp.size = roundedUpAllocationSize;

    if (_MPIrank == MPIWrapper::getWorldRank()) {
        // Allocate the memory
        CU_ASSERT(cuMulticastCreate(&multicastHandle, &multicastProp));

        // Export the allocation to the importing process
        CU_ASSERT(cuMemExportToShareableHandle(&fh, multicastHandle, handleType, 0 /*flags*/));
    }

    MPI_Bcast(&fh, sizeof(fh), MPI_BYTE, _MPIrank, MPI_COMM_WORLD);

    if (_MPIrank != MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuMemImportFromShareableHandle(&multicastHandle, (void *)&fh, handleType));
    }

    CU_ASSERT(cuMulticastAddDevice(multicastHandle, NvLoom::getLocalCuDevice()));

    // Ensure all devices in this process are added BEFORE binding mem on any device
    MPI_Barrier(MPI_COMM_WORLD);

    // Allocate the memory (same as unicast) and bind to MC handle
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = NvLoom::getLocalDevice();
    prop.requestedHandleTypes = handleType;
    CU_ASSERT(cuMemCreate(&handle, roundedUpAllocationSize, &prop, 0 /*flags*/));

    auto error = cuMulticastBindMem(multicastHandle, 0, handle, 0, roundedUpAllocationSize, 0);
    uint8_t earlyReturn = 0;
    if (error == CUDA_ERROR_SYSTEM_NOT_READY) {
        earlyReturn = 1;
    }

    uint8_t reducedEarlyReturn = 0;
    MPI_Allreduce(&earlyReturn, &reducedEarlyReturn, 1, MPI_BYTE, MPI_MAX, MPI_COMM_WORLD);

    if (reducedEarlyReturn > 0) {
        if (error != CUDA_ERROR_SYSTEM_NOT_READY) {
            CU_ASSERT(cuMulticastUnbind(multicastHandle, NvLoom::getLocalCuDevice(), 0, roundedUpAllocationSize));
        }

        CU_ASSERT(cuMemRelease(handle));
        CU_ASSERT(cuMemRelease(multicastHandle));

        return CUDA_ERROR_SYSTEM_NOT_READY;
    }

    CU_ASSERT(error);

    // Map the memory - unicast
    CU_ASSERT(cuMemAddressReserve((CUdeviceptr *) &unicastMappingPtr, roundedUpAllocationSize, 0, 0 /*baseVA*/, 0 /*flags*/));
    CU_ASSERT(cuMemMap((CUdeviceptr) unicastMappingPtr, roundedUpAllocationSize, 0 /*offset*/, handle, 0 /*flags*/));
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = NvLoom::getLocalDevice();
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_ASSERT(cuMemSetAccess((CUdeviceptr) unicastMappingPtr, roundedUpAllocationSize, &desc, 1 /*count*/));

    // Map the memory
    CU_ASSERT(cuMemAddressReserve((CUdeviceptr *) &ptr, roundedUpAllocationSize, 0, 0 /*baseVA*/, 0 /*flags*/));
    CU_ASSERT(cuMemMap((CUdeviceptr) ptr, roundedUpAllocationSize, 0 /*offset*/, multicastHandle, 0 /*flags*/));
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = NvLoom::getLocalDevice();
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_ASSERT(cuMemSetAccess((CUdeviceptr) ptr, roundedUpAllocationSize, &desc, 1 /*count*/));

    MPIrank = _MPIrank;
    allocationSize = _allocationSize;

    // Make sure that everyone is done with mapping the fabric allocation
    MPI_Barrier(MPI_COMM_WORLD);
    return CUDA_SUCCESS;
}


MultinodeMemoryAllocationMulticast::MultinodeMemoryAllocationMulticast(size_t _allocationSize, int _MPIrank) {
    auto error = tryAllocation(_allocationSize, _MPIrank);

    if (error == CUDA_ERROR_SYSTEM_NOT_READY) {
        int totalTimeWaited = 0;

        while ((error == CUDA_ERROR_SYSTEM_NOT_READY) && (totalTimeWaited < 60)) {
            int retryTimeout = 15;
            totalTimeWaited += retryTimeout;

            if (MPIWrapper::getWorldRank() == 0) {
                std::cerr << "sleeping for " << retryTimeout << " seconds before retrying allocating multicast memory, total wait: " << totalTimeWaited << std::endl;
            }

            std::this_thread::sleep_for(std::chrono::seconds(retryTimeout));
            error = tryAllocation(_allocationSize, _MPIrank);
        }
    }
}

MultinodeMemoryAllocationMulticast::~MultinodeMemoryAllocationMulticast() {
    // Make sure that everyone is done using the memory
    MPI_Barrier(MPI_COMM_WORLD);

    CU_ASSERT(cuMulticastUnbind(multicastHandle, NvLoom::getLocalCuDevice(), 0, roundedUpAllocationSize));
    CU_ASSERT(cuMemRelease(handle));

    CU_ASSERT(cuMemUnmap((CUdeviceptr) ptr, roundedUpAllocationSize));
    CU_ASSERT(cuMemRelease(multicastHandle));
    CU_ASSERT(cuMemAddressFree((CUdeviceptr) ptr, roundedUpAllocationSize));
}

void MultinodeMemoryAllocationMulticastRef::memset(int value, CopyType copyType, MemoryPurpose memoryPurpose) {
    memsetBuffer(unicastMappingPtr, value, allocationSize, CU_STREAM_PER_THREAD, copyType, memoryPurpose);
}

unsigned long long MultinodeMemoryAllocationMulticastRef::check(int value, CopyType copyType, int iterations) {
    return checkBuffer(unicastMappingPtr, value, allocationSize, CU_STREAM_PER_THREAD, copyType, iterations);
}

bool MultinodeMemoryAllocationMulticastRef::filter() {
    int pi;
    CU_ASSERT(cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, NvLoom::getLocalCuDevice()));
    return pi != 0;
};

std::vector<CUmemoryPool> MultinodeMemoryPoolAllocationBase::initPoolsLazy() {
    std::vector<CUmemoryPool> pools;
    pools.resize(MPIWrapper::getWorldSize());
    fh_vector.resize(MPIWrapper::getWorldSize());
    handleType = CU_MEM_HANDLE_TYPE_FABRIC;
    poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    poolProps.handleTypes = handleType;
    poolProps.location.type = mem_location;
    cuuint64_t thresholdSize = 1073741824; // 1 GB in bytes;

    if (mem_location == CU_MEM_LOCATION_TYPE_DEVICE) {
        poolProps.location.id = NvLoom::getLocalDevice();
    } else {
        poolProps.location.id = NvLoom::getLocalCpuNumaNode();
    }

    CU_ASSERT(cuMemPoolCreate(&pools[MPIWrapper::getWorldRank()], &poolProps));

    // Set pool release threshold to reserve 1GB before it releases memory back to the OS - Pending investigation @ https://gitlab-master.nvidia.com/dcse-appsys/nvloom/-/issues/24
    CU_ASSERT(cuMemPoolSetAttribute(pools[MPIWrapper::getWorldRank()], CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &thresholdSize));
    CU_ASSERT(cuMemPoolExportToShareableHandle(&fh_vector[MPIWrapper::getWorldRank()], pools[MPIWrapper::getWorldRank()], handleType, 0 /*flags*/));
    MPI_Barrier(MPI_COMM_WORLD);

    // Import handles of other pools to fill vector
    for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
        MPI_Bcast(&fh_vector[i], sizeof(CUmemFabricHandle), MPI_BYTE, i, MPI_COMM_WORLD);
        if (i != MPIWrapper::getWorldRank()) {
            CU_ASSERT(cuMemPoolImportFromShareableHandle(&pools[i], (void *)&fh_vector[i], handleType, 0));
        }
    }

    // Set access for the pools
    for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
        if (mem_location == CU_MEM_LOCATION_TYPE_HOST_NUMA) {
            desc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
            desc.location.id = NvLoom::getLocalCpuNumaNode();
            desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CU_ASSERT(cuMemPoolSetAccess(pools[i], &desc, 1));
        }
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.id = NvLoom::getLocalCuDevice();
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_ASSERT(cuMemPoolSetAccess(pools[i], &desc, 1));
    }
    return pools;
}

MultinodeMemoryPoolAllocationBase::MultinodeMemoryPoolAllocationBase(size_t _allocationSize, int _MPIrank, CUmemLocationType location) {
    allocationSize = _allocationSize;
    MPIrank = _MPIrank;
    mem_location = location;

    if (!devicePoolsInitialized && mem_location == CU_MEM_LOCATION_TYPE_DEVICE) {
        devicePoolsInitialized = true;
        device_pools.resize((MPIWrapper::getWorldSize()));
        device_pools = initPoolsLazy();
    }

    if (!egmPoolsInitialized && mem_location == CU_MEM_LOCATION_TYPE_HOST_NUMA) {
        egmPoolsInitialized = true;
        egm_pools.resize((MPIWrapper::getWorldSize()));
        egm_pools = initPoolsLazy();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (_MPIrank == MPIWrapper::getWorldRank() && mem_location == CU_MEM_LOCATION_TYPE_HOST_NUMA) {
        CU_ASSERT(cuMemAllocFromPoolAsync((CUdeviceptr *)&ptr, allocationSize, egm_pools[_MPIrank], CU_STREAM_PER_THREAD));
        CU_ASSERT(cuStreamSynchronize(CU_STREAM_PER_THREAD));
        CU_ASSERT(cuMemPoolExportPointer(&data, (CUdeviceptr) ptr));
    }
    if (_MPIrank == MPIWrapper::getWorldRank() && mem_location == CU_MEM_LOCATION_TYPE_DEVICE) {
        CU_ASSERT(cuMemAllocFromPoolAsync((CUdeviceptr *)&ptr, allocationSize, device_pools[_MPIrank], CU_STREAM_PER_THREAD));
        CU_ASSERT(cuStreamSynchronize(CU_STREAM_PER_THREAD));
        CU_ASSERT(cuMemPoolExportPointer(&data, (CUdeviceptr) ptr));
    }

    MPI_Bcast(&data, sizeof(data), MPI_BYTE, _MPIrank, MPI_COMM_WORLD);

    if (_MPIrank != MPIWrapper::getWorldRank() && mem_location == CU_MEM_LOCATION_TYPE_HOST_NUMA) {
        CU_ASSERT(cuMemPoolImportPointer((CUdeviceptr*) &ptr, egm_pools[_MPIrank], &data));
    }
    if (_MPIrank != MPIWrapper::getWorldRank() && mem_location == CU_MEM_LOCATION_TYPE_DEVICE){
        CU_ASSERT(cuMemPoolImportPointer((CUdeviceptr*) &ptr, device_pools[_MPIrank], &data));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

MultinodeMemoryPoolAllocationBase::~MultinodeMemoryPoolAllocationBase() {
    CU_ASSERT(cuStreamSynchronize(CU_STREAM_PER_THREAD));

    if (MPIrank != MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuMemFreeAsync((CUdeviceptr) ptr, CU_STREAM_PER_THREAD));
        CU_ASSERT(cuStreamSynchronize(CU_STREAM_PER_THREAD));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (MPIrank == MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuMemFreeAsync((CUdeviceptr) ptr, CU_STREAM_PER_THREAD));
        CU_ASSERT(cuStreamSynchronize(CU_STREAM_PER_THREAD));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

bool MultinodeMemoryPoolAllocationUnicast::filter() {
    int pi;
    CU_ASSERT(cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, NvLoom::getLocalCuDevice()));
    return pi != 0;
};

bool MultinodeMemoryPoolAllocationEGM::filter() {
    int pi;
    CU_ASSERT(cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED, NvLoom::getLocalCuDevice()));
    return pi != 0;
};

template<class T>
AllocationPool<T>::AllocationPool(size_t _allocationSize, int _MPIrank) {
    allocationSize = _allocationSize;
    MPIrank = _MPIrank;

    auto key = std::make_pair(allocationSize, MPIrank);
    auto it = pool.find(key);
    if (it == pool.end()) {
        current = new T(_allocationSize, _MPIrank);
    } else {
        current = it->second;
        pool.erase(it);
    }

    ptr = current->ptr;
}

template<class T>
AllocationPool<T>::~AllocationPool() {
    auto key = std::make_pair(allocationSize, MPIrank);
    pool.insert({key, current});
}

template<class T>
void AllocationPool<T>::clear() {
    for (const auto& elem : pool) {
        delete elem.second;
    }
    pool.clear();
}

template<class T>
bool AllocationPool<T>::filter() {
    return T::filter();
}

template<class T>
std::string AllocationPool<T>::getName() {
    return T::getName();
}

MulticastPool::MulticastPool(size_t _allocationSize, int _MPIrank) {
    allocationSize = _allocationSize;
    MPIrank = _MPIrank;

    auto it = pool.find(allocationSize);

    if (it == pool.end()) {
        // Create one large allocation and suballocate it into smaller chunks
        auto largeAllocation = std::make_shared<MultinodeMemoryAllocationMulticast>(insertCount * allocationSize, 0);
        for (int i = 0; i < insertCount; i++) {
            pool.insert({allocationSize, {largeAllocation, i}});
        }

        it = pool.find(allocationSize);
    }

    ASSERT(it != pool.end());

    parent = it->second.first;
    offset = it->second.second;
    pool.erase(it);

    ptr = (char *) parent->ptr + offset * allocationSize;
    void *unicastMappingPtr = (char *) parent->unicastMappingPtr + (size_t) offset * allocationSize;

    current = std::make_unique<MultinodeMemoryAllocationMulticastRef>(ptr, unicastMappingPtr, allocationSize, MPIrank);
}

MulticastPool::~MulticastPool() {
    pool.insert({allocationSize, {parent, offset}});
}

void MulticastPool::clear() {
    setCapacityHint(defaultInsertCount);
    pool.clear();
}

bool MulticastPool::filter() {
    return MultinodeMemoryAllocationMulticast::filter();
}

std::string MulticastPool::getName() {
    return MultinodeMemoryAllocationMulticast::getName();
}

void clearAllocationPools() {
    AllocationPool<MultinodeMemoryAllocationUnicast>::clear();
    AllocationPool<MultinodeMemoryAllocationEGM>::clear();
    AllocationPool<HostMemoryAllocation>::clear();
    MulticastPool::clear();
}
