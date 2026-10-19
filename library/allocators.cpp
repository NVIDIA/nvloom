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
#include "error.h"
#include "kernels.cuh"
#include "nvloom.h"
#include "enums.h"

static size_t roundUp(size_t number, size_t multiple) {
    return ((number + multiple - 1) / multiple) * multiple;
}

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

LocalizedMultinodeMemoryAllocationUnicast::LocalizedMultinodeMemoryAllocationUnicast(size_t _allocationSize, int _MPIrank) {
#if CUDA_VERSION >= 13040
    MPIrank = _MPIrank;
    handleType = CU_MEM_HANDLE_TYPE_FABRIC;
    CU_ASSERT(cuDeviceGetAttribute(&numLocalityDomains, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, NvLoom::getLocalCuDevice()));
    ASSERT(numLocalityDomains >= 1);

    handles.resize(numLocalityDomains);
    fhs.resize(numLocalityDomains);
    memoryChunks.resize(numLocalityDomains);

    auto makeProp = [&](int domain) {
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
        prop.location.localized.deviceId = (unsigned char)NvLoom::getLocalDevice();
        prop.location.localized.localityDomainId = (unsigned char)domain;
        return prop;
    };

    size_t granularity = 0;
    CUmemAllocationProp prop = makeProp(0);
    CU_ASSERT(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    roundedChunkSize = roundUp((_allocationSize + numLocalityDomains - 1) / numLocalityDomains, granularity);
    allocationSize = roundedChunkSize * numLocalityDomains;

    if (_MPIrank == MPIWrapper::getWorldRank()) {
        for (int i = 0; i < numLocalityDomains; i++) {
            CUmemAllocationProp p = makeProp(i);
            CU_ASSERT(cuMemCreate(&handles[i], roundedChunkSize, &p, 0));
            CU_ASSERT(cuMemExportToShareableHandle(&fhs[i], handles[i], handleType, 0));
        }
    }

    MPI_Bcast(fhs.data(), fhs.size() * sizeof(CUmemFabricHandle), MPI_BYTE, _MPIrank, MPI_COMM_WORLD);

    if (_MPIrank != MPIWrapper::getWorldRank()) {
        for (int i = 0; i < numLocalityDomains; i++) {
            CU_ASSERT(cuMemImportFromShareableHandle(&handles[i], (void *)&fhs[i], handleType));
        }
    }

    CU_ASSERT(cuMemAddressReserve((CUdeviceptr *)&ptr, allocationSize, 0, 0, 0));

    for (int i = 0; i < numLocalityDomains; i++) {
        CUdeviceptr chunk = (CUdeviceptr)ptr + (size_t)i * roundedChunkSize;
        CU_ASSERT(cuMemMap(chunk, roundedChunkSize, 0, handles[i], 0));

        CUmemAccessDesc desc = {};
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.id = NvLoom::getLocalDevice();
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_ASSERT(cuMemSetAccess(chunk, roundedChunkSize, &desc, 1));

        memoryChunks[i] = { (void *)chunk, roundedChunkSize, i};
    }

    MPI_Barrier(MPI_COMM_WORLD);
#else
    ASSERT(0);
#endif
}

LocalizedMultinodeMemoryAllocationUnicast::~LocalizedMultinodeMemoryAllocationUnicast() {
#if CUDA_VERSION >= 13040
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < numLocalityDomains; i++) {
        CUdeviceptr chunk = (CUdeviceptr)ptr + (size_t)i * roundedChunkSize;
        CU_ASSERT(cuMemUnmap(chunk, roundedChunkSize));
        CU_ASSERT(cuMemRelease(handles[i]));
    }
    CU_ASSERT(cuMemAddressFree((CUdeviceptr)ptr, allocationSize));

    MPI_Barrier(MPI_COMM_WORLD);
#else
    ASSERT(0);
#endif
}

// A copy through an allocation is only legal when every process belongs to the same fabric
// clique of the type backing that allocation. Clique information is unavailable below CUDA 13.4,
// where nothing is waived on these grounds.
#if CUDA_VERSION >= 13040
static bool unicastPointerCliqueShared() { return !NvLoom::isCliqueSplit(CU_CLIQUE_TYPE_UNICAST_POINTER); }
static bool multicastPointerCliqueShared() { return !NvLoom::isCliqueSplit(CU_CLIQUE_TYPE_MULTICAST_POINTER); }
static bool unicastEndpointCliqueShared() { return !NvLoom::isCliqueSplit(CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT); }
static bool multicastEndpointCliqueShared() { return !NvLoom::isCliqueSplit(CU_CLIQUE_TYPE_MULTICAST_LOGICAL_ENDPOINT); }
#else
static bool unicastPointerCliqueShared() { return true; }
static bool multicastPointerCliqueShared() { return true; }
static bool unicastEndpointCliqueShared() { return true; }
static bool multicastEndpointCliqueShared() { return true; }
#endif

bool LocalizedMultinodeMemoryAllocationUnicast::filter() {
#if CUDA_VERSION >= 13040
    int driverVersion = 0;
    CU_ASSERT(cuDriverGetVersion(&driverVersion));
    if (driverVersion < 13040) return false;

    int fabricSupported = 0;
    int numLocalityDomains = 0;

    CU_ASSERT(cuDeviceGetAttribute(&fabricSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, NvLoom::getLocalCuDevice()));
    CU_ASSERT(cuDeviceGetAttribute(&numLocalityDomains, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, NvLoom::getLocalCuDevice()));

    return (fabricSupported != 0) && (numLocalityDomains > 1) && unicastPointerCliqueShared();
#else
    return false;
#endif
}

#if CUDA_VERSION >= 13030
CUlogicalEndpointId MemoryAllocation::getEndpointId() {
    // This should never be called
    ASSERT(0);
    return {};
}

size_t MemoryAllocation::getEndpointOffset() {
    // This should never be called
    ASSERT(0);
    return 0;
}
#endif

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

    MPI_Barrier(MPI_COMM_WORLD);
}

bool MultinodeMemoryAllocationUnicast::filter() {
    int pi;
    CU_ASSERT(cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, NvLoom::getLocalCuDevice()));
    return pi != 0 && unicastPointerCliqueShared();
};

bool MultinodeMemoryAllocationEGM::filter() {
    int pi;
    CU_ASSERT(cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED, NvLoom::getLocalCuDevice()));
    return pi != 0 && unicastPointerCliqueShared();
};

size_t MemoryAllocationCFT::getRoundedAllocationSize(size_t _allocationSize) {
    memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    memProp.location.id = NvLoom::getLocalDevice();

    size_t granularity = 0;
    CU_ASSERT(cuMemGetAllocationGranularity(&granularity, &memProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    return roundUp(_allocationSize, granularity);
}

void *MemoryAllocationCFT::allocateMemory(size_t _allocationSize) {
    void *ptr;
    roundedUpAllocationSize = getRoundedAllocationSize(_allocationSize);
    // Allocate the memory
    CU_ASSERT(cuMemCreate(&handle, roundedUpAllocationSize, &memProp, 0 /*flags*/));

    // Map the memory
    CU_ASSERT(cuMemAddressReserve((CUdeviceptr *) &ptr, roundedUpAllocationSize, 0, 0 /*baseVA*/, 0 /*flags*/));

    CU_ASSERT(cuMemMap((CUdeviceptr) ptr, roundedUpAllocationSize, 0 /*offset*/, handle, 0 /*flags*/));

    CUmemAccessDesc desc = {};
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = NvLoom::getLocalDevice();
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_ASSERT(cuMemSetAccess((CUdeviceptr) ptr, roundedUpAllocationSize, &desc, 1 /*count*/));

    return ptr;
}

void MemoryAllocationCFT::freeMemory(void *ptr) {
    CU_ASSERT(cuMemUnmap((CUdeviceptr) ptr, roundedUpAllocationSize));
    CU_ASSERT(cuMemRelease(handle));
    CU_ASSERT(cuMemAddressFree((CUdeviceptr) ptr, roundedUpAllocationSize));
}

RangeAllocator::RangeAllocator(size_t _regionSize, size_t _alignment) {
    regionSize = _regionSize;
    alignment = _alignment;
    freeRanges[0] = regionSize;
}

size_t RangeAllocator::allocate(size_t size) {
    size = roundUp(size, alignment);

    // first fit
    for (auto it = freeRanges.begin(); it != freeRanges.end(); it++) {
        if (it->second < size) {
            continue;
        }

        size_t offset = it->first;
        size_t remaining = it->second - size;
        freeRanges.erase(it);
        if (remaining != 0) {
            freeRanges[offset + size] = remaining;
        }
        return offset;
    }

    // no free range is large enough
    ASSERT(0);
    return 0;
}

void RangeAllocator::free(size_t offset, size_t size) {
    size = roundUp(size, alignment);

    // merge with the free range starting right after the released one
    auto next = freeRanges.upper_bound(offset);
    if (next != freeRanges.end()) {
        ASSERT(offset + size <= next->first);
        if (offset + size == next->first) {
            size += next->second;
            next = freeRanges.erase(next);
        }
    }

    // merge with the free range ending right before the released one
    if (next != freeRanges.begin()) {
        auto previous = next;
        previous--;
        ASSERT(previous->first + previous->second <= offset);
        if (previous->first + previous->second == offset) {
            previous->second += size;
            return;
        }
    }

    freeRanges[offset] = size;
}

bool RangeAllocator::isFullyFree() {
    return freeRanges.size() == 1 && freeRanges.begin()->second == regionSize;
}

UnicastLogicalEndpoint::UnicastLogicalEndpoint(int _MPIrank) {
#if CUDA_VERSION >= 13030
    MPIrank = _MPIrank;

    CUlogicalEndpointFabricHandle fh = {};

    CU_ASSERT(cuLogicalEndpointIdReserve(&endpointId, 1 /*count*/));
    CUlogicalEndpointProp endpointProp = {};
    endpointProp.unicast.device = NvLoom::getLocalDevice();
    endpointProp.flags = 0;
    endpointProp.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
    endpointProp.type = CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
    cuuint64_t bindAlignment = 0;
    cuuint64_t maxSize = 0;

    CU_ASSERT(cuLogicalEndpointGetLimits(&bindAlignment, &maxSize, &endpointProp));
    endpointProp.size = maxSize;

    rangeAllocator.emplace(maxSize, bindAlignment);

    if (MPIrank == MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuLogicalEndpointCreate(endpointId, &endpointProp));

        int queryStatus = 0;
        while (queryStatus == 0) {
            CU_ASSERT(cuLogicalEndpointQuery(endpointId, 1, &queryStatus));
        }

        CU_ASSERT(cuLogicalEndpointExport(&fh, endpointId, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC));
    }

    MPI_Bcast(&fh, sizeof(fh), MPI_BYTE, MPIrank, MPI_COMM_WORLD);

    if (MPIrank != MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuLogicalEndpointImport(endpointId, (void *)&fh, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC));

        int queryStatus = 0;
        while (queryStatus == 0) {
            CU_ASSERT(cuLogicalEndpointQuery(endpointId, 1, &queryStatus));
        }
    }
#else
    ASSERT(0);
#endif
}

UnicastLogicalEndpoint::~UnicastLogicalEndpoint() {
#if CUDA_VERSION >= 13030
    // every allocation sharing this endpoint must have released its range by now
    ASSERT(rangeAllocator->isFullyFree());

    CU_ASSERT(cuLogicalEndpointDestroy(endpointId));
    CU_ASSERT(cuLogicalEndpointIdRelease(endpointId, 1 /*count*/));
#else
    ASSERT(0);
#endif
}

#if CUDA_VERSION >= 13030
size_t UnicastLogicalEndpoint::allocateRange(size_t size) {
    return rangeAllocator->allocate(size);
}

void UnicastLogicalEndpoint::freeRange(size_t offset, size_t size) {
    rangeAllocator->free(offset, size);
}
#endif

UnicastLogicalEndpointPool::UnicastLogicalEndpointPool(int MPIrank) {
    auto it = pool.find(MPIrank);
    if (it != pool.end()) {
        // the entry is stale if the endpoint got destroyed since it was inserted
        endpoint = it->second.lock();
    }

    if (endpoint == nullptr) {
        endpoint = std::make_shared<UnicastLogicalEndpoint>(MPIrank);
        pool[MPIrank] = endpoint;
    }

    if (retainEndpoints) {
        retainedEndpoints[MPIrank] = endpoint;
    }
}

void UnicastLogicalEndpointPool::setRetainEndpoints(bool _retainEndpoints) {
    retainEndpoints = _retainEndpoints;
}

void UnicastLogicalEndpointPool::clear() {
    // endpoints which are still used by an allocation outlive this call
    retainedEndpoints.clear();
}

MultinodeMemoryAllocationUnicastCFT::MultinodeMemoryAllocationUnicastCFT(size_t _allocationSize, int _MPIrank) : endpointPool(_MPIrank) {
#if CUDA_VERSION >= 13030
    endpointId = endpointPool.getEndpointId();

    if (_MPIrank == MPIWrapper::getWorldRank()) {
        ptr = allocateMemory(_allocationSize);
        endpointOffset = endpointPool.allocateRange(roundedUpAllocationSize);
        CU_ASSERT(cuLogicalEndpointBindMem(endpointId, NvLoom::getLocalDevice(), endpointOffset, handle, 0, roundedUpAllocationSize, 0 /*flags*/));
    } else {
        // Only local process can access the memory through pointers
        ptr = nullptr;
    }

    MPIrank = _MPIrank;
    allocationSize = _allocationSize;

    MPI_Bcast(&endpointOffset, sizeof(endpointOffset), MPI_BYTE, _MPIrank, MPI_COMM_WORLD);
#else
    ASSERT(0);
#endif
}

MultinodeMemoryAllocationUnicastCFT::~MultinodeMemoryAllocationUnicastCFT() {
#if CUDA_VERSION >= 13030
    // Make sure that everyone is done using the memory
    MPI_Barrier(MPI_COMM_WORLD);

    if (MPIrank == MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuLogicalEndpointUnbind(endpointId, NvLoom::getLocalDevice(), endpointOffset, roundedUpAllocationSize));
        endpointPool.freeRange(endpointOffset, roundedUpAllocationSize);
        freeMemory(ptr);
    }

#else
    ASSERT(0);
#endif
}

bool MultinodeMemoryAllocationUnicastCFT::filter() {
#if CUDA_VERSION >= 13030
    // Drivers older than 13.3 reject the attribute query below with CUDA_ERROR_INVALID_VALUE
    int driverVersion = 0;
    CU_ASSERT(cuDriverGetVersion(&driverVersion));
    if (driverVersion < 13030) {
        return false;
    }

    int unicastSupported;
    CU_ASSERT(cuDeviceGetAttribute(&unicastSupported, CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED, NvLoom::getLocalCuDevice()));
    return unicastSupported != 0 && unicastEndpointCliqueShared();
#else
    return false;
#endif
};

MultinodeMemoryAllocationMulticastCFT::MultinodeMemoryAllocationMulticastCFT(size_t _allocationSize, int _MPIrank) {
#if CUDA_VERSION >= 13030
    CU_ASSERT(cuLogicalEndpointIdReserve(&endpointId, 1 /*count*/));
    CUlogicalEndpointProp endpointProp = {};
    endpointProp.multicast.numDevices = MPIWrapper::getWorldSize();
    endpointProp.flags = 0;
    endpointProp.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
    endpointProp.type = CU_LOGICAL_ENDPOINT_TYPE_MULTICAST;
    // cuLogicalEndpointBindMem binds the granularity-rounded size, which must fit within the endpoint
    endpointProp.size = getRoundedAllocationSize(_allocationSize);

    CUlogicalEndpointFabricHandle fh = {};

    if (_MPIrank == MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuLogicalEndpointCreate(endpointId, &endpointProp));
        CU_ASSERT(cuLogicalEndpointExport(&fh, endpointId, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC));
    }

    MPI_Bcast(&fh, sizeof(fh), MPI_BYTE, _MPIrank, MPI_COMM_WORLD);

    if (_MPIrank != MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuLogicalEndpointImport(endpointId, (void *)&fh, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC));
    }

    CU_ASSERT(cuLogicalEndpointAddDevice(endpointId, NvLoom::getLocalCuDevice()));

    int queryStatus = 0;
    while (queryStatus == 0) {
        CU_ASSERT(cuLogicalEndpointQuery(endpointId, 1, &queryStatus));
    }

    unicastMappingPtr = allocateMemory(_allocationSize);

    CU_ASSERT(cuLogicalEndpointBindMem(endpointId, NvLoom::getLocalDevice(), 0, handle, 0, roundedUpAllocationSize, 0 /*flags*/));

    MPIrank = _MPIrank;
    allocationSize = _allocationSize;

    // Make sure that everyone is done with mapping the fabric allocation
    MPI_Barrier(MPI_COMM_WORLD);
#else
    ASSERT(0);
#endif
}

MultinodeMemoryAllocationMulticastCFT::~MultinodeMemoryAllocationMulticastCFT() {
#if CUDA_VERSION >= 13030
    // Make sure that everyone is done using the memory
    MPI_Barrier(MPI_COMM_WORLD);

    CU_ASSERT(cuLogicalEndpointUnbind(endpointId, NvLoom::getLocalDevice(), 0, roundedUpAllocationSize));

    freeMemory(unicastMappingPtr);

    CU_ASSERT(cuLogicalEndpointDestroy(endpointId));
    CU_ASSERT(cuLogicalEndpointIdRelease(endpointId, 1 /*count*/));

    MPI_Barrier(MPI_COMM_WORLD);
#else
    ASSERT(0);
#endif
}

bool MultinodeMemoryAllocationMulticastCFT::filter() {
#if CUDA_VERSION >= 13030
    // Drivers older than 13.3 reject the attribute query below with CUDA_ERROR_INVALID_VALUE
    int driverVersion = 0;
    CU_ASSERT(cuDriverGetVersion(&driverVersion));
    if (driverVersion < 13030) {
        return false;
    }

    if (getMajorDriverVersion() < 610) {
        return false;
    }

    int multicastSupported;
    CU_ASSERT(cuDeviceGetAttribute(&multicastSupported, CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_MULTICAST_SUPPORTED, NvLoom::getLocalCuDevice()));
    return multicastSupported != 0 && multicastEndpointCliqueShared();
#else
    return false;
#endif
};

void MultinodeMemoryAllocationMulticastCFT::memset(int value, CopyType copyType, MemoryPurpose memoryPurpose) {
    memsetBuffer(unicastMappingPtr, value, allocationSize, CU_STREAM_PER_THREAD, copyType, memoryPurpose);
}

unsigned long long MultinodeMemoryAllocationMulticastCFT::check(int value, CopyType copyType, int iterations) {
    return checkBuffer(unicastMappingPtr, value, allocationSize, CU_STREAM_PER_THREAD, copyType, iterations);
}

std::unique_ptr<MultinodeMemoryAllocationMulticastCFTRef> MultinodeMemoryAllocationMulticastCFT::createRef(int chunkIndex, size_t chunkSize, int _MPIrank) {
    return std::make_unique<MultinodeMemoryAllocationMulticastCFTRef>(this,
        (size_t) chunkIndex * chunkSize,
        (char *) unicastMappingPtr + (size_t) chunkIndex * chunkSize,
        chunkSize, _MPIrank);
}

bool MultinodeMemoryAllocationMulticastCFTRef::filter() {
    return MultinodeMemoryAllocationMulticastCFT::filter();
}


void MultinodeMemoryAllocationMulticast::createMulticastGroup() {
    handleType = CU_MEM_HANDLE_TYPE_FABRIC;
    multicastProp.numDevices = MPIWrapper::getWorldSize();
    multicastProp.handleTypes = handleType;
    size_t gran;
    CU_ASSERT(cuMulticastGetGranularity(&gran, &multicastProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    roundedUpAllocationSize = roundUp(allocationSize, gran);
    multicastProp.size = roundedUpAllocationSize;

    if (MPIrank == MPIWrapper::getWorldRank()) {
        // Allocate the memory
        CU_ASSERT(cuMulticastCreate(&multicastHandle, &multicastProp));

        // Export the allocation to the importing process
        CU_ASSERT(cuMemExportToShareableHandle(&fh, multicastHandle, handleType, 0 /*flags*/));
    }

    MPI_Bcast(&fh, sizeof(fh), MPI_BYTE, MPIrank, MPI_COMM_WORLD);

    if (MPIrank != MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuMemImportFromShareableHandle(&multicastHandle, (void *)&fh, handleType));
    }

    CU_ASSERT(cuMulticastAddDevice(multicastHandle, NvLoom::getLocalCuDevice()));

    // Ensure all devices in this process are added BEFORE binding mem on any device
    MPI_Barrier(MPI_COMM_WORLD);
}

void MultinodeMemoryAllocationMulticast::allocateMemory() {
    // Allocate the memory (same as unicast) and bind to MC handle
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = NvLoom::getLocalDevice();
    prop.requestedHandleTypes = handleType;
    CU_ASSERT(cuMemCreate(&handle, roundedUpAllocationSize, &prop, 0 /*flags*/));

    CU_ASSERT(cuMulticastBindMem(multicastHandle, 0, handle, 0, roundedUpAllocationSize, 0));

    // Map the memory - unicast
    CU_ASSERT(cuMemAddressReserve((CUdeviceptr *) &unicastMappingPtr, roundedUpAllocationSize, 0, 0 /*baseVA*/, 0 /*flags*/));
    CU_ASSERT(cuMemMap((CUdeviceptr) unicastMappingPtr, roundedUpAllocationSize, 0 /*offset*/, handle, 0 /*flags*/));
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = NvLoom::getLocalDevice();
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_ASSERT(cuMemSetAccess((CUdeviceptr) unicastMappingPtr, roundedUpAllocationSize, &desc, 1 /*count*/));
}

void MultinodeMemoryAllocationMulticast::allocateMemoryCudaPool() {
    Pools::initPools(CU_MEM_LOCATION_TYPE_DEVICE);
    CU_ASSERT(cuMemAllocFromPoolAsync((CUdeviceptr *)&unicastMappingPtr, roundedUpAllocationSize, Pools::pools[CU_MEM_LOCATION_TYPE_DEVICE][MPIWrapper::getWorldRank()], CU_STREAM_PER_THREAD));
    CU_ASSERT(cuMulticastBindAddr(multicastHandle, 0, (CUdeviceptr) unicastMappingPtr, roundedUpAllocationSize, 0));
}

void MultinodeMemoryAllocationMulticast::mapMemory() {
    // Map the memory
    CU_ASSERT(cuMemAddressReserve((CUdeviceptr *) &ptr, roundedUpAllocationSize, 0, 0 /*baseVA*/, 0 /*flags*/));
    CU_ASSERT(cuMemMap((CUdeviceptr) ptr, roundedUpAllocationSize, 0 /*offset*/, multicastHandle, 0 /*flags*/));
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = NvLoom::getLocalDevice();
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_ASSERT(cuMemSetAccess((CUdeviceptr) ptr, roundedUpAllocationSize, &desc, 1 /*count*/));
}

MultinodeMemoryAllocationMulticast::MultinodeMemoryAllocationMulticast(size_t _allocationSize, int _MPIrank, bool _useCudaPool) {
    useCudaPool = _useCudaPool;
    MPIrank = _MPIrank;
    allocationSize = _allocationSize;

    createMulticastGroup();

    if (useCudaPool) {
        allocateMemoryCudaPool();
    } else {
        allocateMemory();
    }

    mapMemory();

    // Make sure that everyone is done with mapping the fabric allocation
    MPI_Barrier(MPI_COMM_WORLD);
}

void MultinodeMemoryAllocationMulticast::releaseMemory() {
    CU_ASSERT(cuMemUnmap((CUdeviceptr) unicastMappingPtr, roundedUpAllocationSize));
    CU_ASSERT(cuMemAddressFree((CUdeviceptr) unicastMappingPtr, roundedUpAllocationSize));
    CU_ASSERT(cuMemRelease(handle));
}

void MultinodeMemoryAllocationMulticast::releaseMemoryCudaPool() {
    CU_ASSERT(cuMemFreeAsync((CUdeviceptr) unicastMappingPtr, CU_STREAM_PER_THREAD));
    CU_ASSERT(cuStreamSynchronize(CU_STREAM_PER_THREAD));
}

MultinodeMemoryAllocationMulticast::~MultinodeMemoryAllocationMulticast() {
    // Make sure that everyone is done using the memory
    MPI_Barrier(MPI_COMM_WORLD);

    CU_ASSERT(cuMulticastUnbind(multicastHandle, NvLoom::getLocalCuDevice(), 0, roundedUpAllocationSize));

    if (useCudaPool) {
        releaseMemoryCudaPool();
    } else {
        releaseMemory();
    }

    CU_ASSERT(cuMemUnmap((CUdeviceptr) ptr, roundedUpAllocationSize));
    CU_ASSERT(cuMemRelease(multicastHandle));
    CU_ASSERT(cuMemAddressFree((CUdeviceptr) ptr, roundedUpAllocationSize));

    MPI_Barrier(MPI_COMM_WORLD);
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
    return pi != 0 && multicastPointerCliqueShared();
};

std::unique_ptr<MultinodeMemoryAllocationMulticastRef> MultinodeMemoryAllocationMulticast::createRef(int chunkIndex, size_t chunkSize, int _MPIrank) {
    return std::make_unique<MultinodeMemoryAllocationMulticastRef>(
        (char *) ptr + (size_t) chunkIndex * chunkSize,
        (char *) unicastMappingPtr + (size_t) chunkIndex * chunkSize,
        chunkSize, _MPIrank);
}

void Pools::initPools(CUmemLocationType location) {
    if (!pools[location].empty()) {
        return;
    }
    CUmemPoolProps poolProps = { };
    CUmemAccessDesc desc = {};
    std::vector<CUmemFabricHandle> fh_vector;
    pools[location].resize(MPIWrapper::getWorldSize());
    fh_vector.resize(MPIWrapper::getWorldSize());
    CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_FABRIC;
    poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    poolProps.handleTypes = handleType;
    poolProps.location.type = location;

    if (location == CU_MEM_LOCATION_TYPE_DEVICE) {
        poolProps.location.id = NvLoom::getLocalDevice();
    } else {
        poolProps.location.id = NvLoom::getLocalCpuNumaNode();
    }

    CU_ASSERT(cuMemPoolCreate(&pools[location][MPIWrapper::getWorldRank()], &poolProps));

    CU_ASSERT(cuMemPoolExportToShareableHandle(&fh_vector[MPIWrapper::getWorldRank()], pools[location][MPIWrapper::getWorldRank()], handleType, 0 /*flags*/));
    MPI_Barrier(MPI_COMM_WORLD);

    // Import handles of other pools to fill vector
    for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
        MPI_Bcast(&fh_vector[i], sizeof(CUmemFabricHandle), MPI_BYTE, i, MPI_COMM_WORLD);
        if (i != MPIWrapper::getWorldRank()) {
            CU_ASSERT(cuMemPoolImportFromShareableHandle(&pools[location][i], (void *)&fh_vector[i], handleType, 0));
        }
    }

    // Set access for the pools
    for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.id = NvLoom::getLocalCuDevice();
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_ASSERT(cuMemPoolSetAccess(pools[location][i], &desc, 1));
    }
}

MultinodeMemoryPoolAllocationBase::MultinodeMemoryPoolAllocationBase(size_t _allocationSize, int _MPIrank, CUmemLocationType location) {
    allocationSize = _allocationSize;
    MPIrank = _MPIrank;
    CUmemPoolPtrExportData data;

    Pools::initPools(location);

    MPI_Barrier(MPI_COMM_WORLD);

    if (_MPIrank == MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuMemAllocFromPoolAsync((CUdeviceptr *)&ptr, allocationSize, Pools::pools[location][_MPIrank], CU_STREAM_PER_THREAD));
        CU_ASSERT(cuStreamSynchronize(CU_STREAM_PER_THREAD));
        CU_ASSERT(cuMemPoolExportPointer(&data, (CUdeviceptr) ptr));
    }

    MPI_Bcast(&data, sizeof(data), MPI_BYTE, _MPIrank, MPI_COMM_WORLD);

    if (_MPIrank != MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuMemPoolImportPointer((CUdeviceptr*) &ptr, Pools::pools[location][_MPIrank], &data));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

MultinodeMemoryPoolAllocationBase::~MultinodeMemoryPoolAllocationBase() {
    MPI_Barrier(MPI_COMM_WORLD);

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
    return pi != 0 && unicastPointerCliqueShared();
};

bool MultinodeMemoryPoolAllocationEGM::filter() {
    int pi;
    CU_ASSERT(cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED, NvLoom::getLocalCuDevice()));
    return pi != 0 && unicastPointerCliqueShared();
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

template<class T, class TRef>
MulticastPool<T, TRef>::MulticastPool(size_t _allocationSize, int _MPIrank) {
    allocationSize = _allocationSize;
    MPIrank = _MPIrank;

    auto it = pool.find(allocationSize);

    if (it == pool.end()) {
        // Create one large allocation and suballocate it into smaller chunks
        auto largeAllocation = std::make_shared<T>(insertCount * allocationSize, 0);
        for (int i = 0; i < insertCount; i++) {
            pool.insert({allocationSize, {largeAllocation, i}});
        }

        it = pool.find(allocationSize);
    }

    ASSERT(it != pool.end());

    parent = it->second.first;
    offset = it->second.second;
    pool.erase(it);

    // The parent knows how to address its suballocations
    // (pointer arithmetic for regular multicast, endpointId + offset for CFT)
    current = parent->createRef(offset, allocationSize, MPIrank);
    ptr = current->ptr;
}

template<class T, class TRef>
MulticastPool<T, TRef>::~MulticastPool() {
    pool.insert({allocationSize, {parent, offset}});
}

template<class T, class TRef>
void MulticastPool<T, TRef>::clear() {
    setCapacityHint(defaultInsertCount);
    pool.clear();
}

template<class T, class TRef>
bool MulticastPool<T, TRef>::filter() {
    return T::filter();
}

template<class T, class TRef>
std::string MulticastPool<T, TRef>::getName() {
    return T::getName();
}

void clearAllocationPools() {
    AllocationPool<MultinodeMemoryAllocationUnicast>::clear();
    AllocationPool<MultinodeMemoryAllocationEGM>::clear();
    AllocationPool<HostMemoryAllocation>::clear();
    AllocationPool<LocalizedMultinodeMemoryAllocationUnicast>::clear();
    AllocationPool<MultinodeMemoryAllocationUnicastCFT>::clear();
    MulticastPool<MultinodeMemoryAllocationMulticast, MultinodeMemoryAllocationMulticastRef>::clear();
    MulticastPool<MultinodeMemoryAllocationMulticastCFT, MultinodeMemoryAllocationMulticastCFTRef>::clear();
    // has to come last: the allocations cleared above hold references to the endpoints
    UnicastLogicalEndpointPool::clear();
}
