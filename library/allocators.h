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

#ifndef ALLOCATORS_H_
#define ALLOCATORS_H_

#include <map>
#include <memory>
#include <optional>
#include "kernels.cuh"
#include "enums.h"
#include "nvloom.h"

// Once created, the pools live until the end of the program
// CUDA driver maintains the allocations in pools, so there's no need to destroy them
// Allocations should get reused between testcases
class Pools {
public:
    static inline std::map<CUmemLocationType, std::vector<CUmemoryPool> > pools = {
        {CU_MEM_LOCATION_TYPE_DEVICE, {}},
        {CU_MEM_LOCATION_TYPE_HOST_NUMA, {}}
    };
    static void initPools(CUmemLocationType location);
};

struct MemoryChunk {
    void *ptr;
    size_t size;
    int localityDomainId;
};

class MemoryAllocation {
public:
    void *ptr = nullptr;
    size_t allocationSize = 0;
    int MPIrank;
    static inline int uniqueIdCounter = 0;
    int uniqueId;

    MemoryAllocation(int _MPIrank = -1) {
        uniqueId = uniqueIdCounter;
        uniqueIdCounter++;
        MPIrank = _MPIrank;
    };

    ~MemoryAllocation() {};

    // default filter is NOOP. It can be overloaded by classes inheriting it
    static bool filter() {
        return true;
    };

    static std::string getName() {
        return "N/A";
    }

    virtual std::vector<MemoryChunk> getChunks() {
        MemoryChunk defaultChunk = {ptr, allocationSize, -1};
        return { defaultChunk };
    }

    virtual int getNumLocalityDomains() const {
        return 1;
    }

    // copyType and memoryPurpose are optional arguments, and are only used in multicast reduction benchmarks
    virtual void memset(int value, CopyType copyType = COPY_TYPE_CE, MemoryPurpose memoryPurpose = MemoryPurpose::MEMORY_SOURCE);
    // copyType and iterations are optional arguments, and are only used in multicast reduction benchmarks
    virtual unsigned long long check(int value, CopyType copyType = COPY_TYPE_CE, int iterations = 0);

#if CUDA_VERSION >= 13030
    virtual CUlogicalEndpointId getEndpointId();
    virtual size_t getEndpointOffset();
#endif
};

class DeviceMemoryAllocation : public MemoryAllocation {
public:
    DeviceMemoryAllocation(size_t _allocationSize, int _MPIrank);
    ~DeviceMemoryAllocation();

    // not used in benchmarks
    static std::string getName() {
        return "N/A";
    }
};

class HostMemoryAllocation : public MemoryAllocation {
public:
    HostMemoryAllocation(size_t _allocationSize, int _MPIrank);
    ~HostMemoryAllocation();

    static std::string getName() {
        return "host";
    }
};

class MultinodeMemoryAllocationBase : public MemoryAllocation {
private:
    CUmemGenericAllocationHandle handle = {};
    CUmemFabricHandle fh = {};
    CUmemAllocationHandleType handleType = {};
    CUmemAllocationProp prop = {};
    CUmemAccessDesc desc = {};
    size_t roundedUpAllocationSize;

public:
    MultinodeMemoryAllocationBase(size_t _allocationSize, int _MPIrank, CUmemLocationType location);
    virtual ~MultinodeMemoryAllocationBase();
};

class MultinodeMemoryAllocationUnicast : public MultinodeMemoryAllocationBase {
public:
    MultinodeMemoryAllocationUnicast(size_t _allocationSize, int _MPIrank) : MultinodeMemoryAllocationBase(_allocationSize, _MPIrank, CU_MEM_LOCATION_TYPE_DEVICE) {};
    static bool filter();

    static std::string getName() {
        return "device";
    }
};

class LocalizedMultinodeMemoryAllocationUnicast : public MemoryAllocation {
private:
    int numLocalityDomains = 0;
    CUmemAllocationHandleType handleType = {};
    std::vector<CUmemGenericAllocationHandle> handles;
    std::vector<CUmemFabricHandle> fhs;
    std::vector<MemoryChunk> memoryChunks;
    size_t roundedChunkSize;
public:
    LocalizedMultinodeMemoryAllocationUnicast(size_t _allocationSize, int _MPIrank);
    ~LocalizedMultinodeMemoryAllocationUnicast();

    std::vector<MemoryChunk> getChunks() override {
        return memoryChunks;
    }

    int getNumLocalityDomains() const override {
        return numLocalityDomains;
    }

    static bool filter();
    static std::string getName() {
        return "device";
    }
};

class MemoryChunkAllocation : public MemoryAllocation {
    std::shared_ptr<MemoryAllocation> parent;
public:
    int localityDomainId;
    MemoryChunkAllocation(const MemoryChunk& chunk, int _MPIrank, std::shared_ptr<MemoryAllocation> _parent) : MemoryAllocation(_MPIrank), parent(_parent), localityDomainId(chunk.localityDomainId) {
        ptr = chunk.ptr;
        allocationSize = chunk.size;
    }
};

class MultinodeMemoryAllocationEGM : public MultinodeMemoryAllocationBase {
public:
    MultinodeMemoryAllocationEGM(size_t _allocationSize, int _MPIrank) : MultinodeMemoryAllocationBase(_allocationSize, _MPIrank, CU_MEM_LOCATION_TYPE_HOST_NUMA) {};
    static bool filter();

    static std::string getName() {
        return "egm";
    }
};

// Hands out non-overlapping, aligned ranges within a fixed-size region.
// Only the free ranges are tracked (offset -> size); they are kept disjoint and never
// adjacent by coalescing on release, so the whole region is a single entry when idle.
class RangeAllocator {
private:
    std::map<size_t, size_t> freeRanges;
    size_t regionSize;
    size_t alignment;

public:
    RangeAllocator(size_t _regionSize, size_t _alignment);

    // returns the offset of a range of at least size bytes
    size_t allocate(size_t size);
    void free(size_t offset, size_t size);

    bool isFullyFree();
};

// Owns a unicast logical endpoint: it is created by the constructor and destroyed by
// the destructor. Several allocations share a single endpoint, so the endpoint address
// space has to be suballocated between them.
class UnicastLogicalEndpoint {
    private:
#if CUDA_VERSION >= 13030
        CUlogicalEndpointId endpointId = {};
        int MPIrank;
        std::optional<RangeAllocator> rangeAllocator;
#endif

    public:
        UnicastLogicalEndpoint(int _MPIrank);

        ~UnicastLogicalEndpoint();

        UnicastLogicalEndpoint(const UnicastLogicalEndpoint &) = delete;
        UnicastLogicalEndpoint &operator=(const UnicastLogicalEndpoint &) = delete;

#if CUDA_VERSION >= 13030
        CUlogicalEndpointId getEndpointId() const {
            return endpointId;
        }

        size_t allocateRange(size_t size);
        void freeRange(size_t offset, size_t size);
#endif
};

// Hands out unicast logical endpoints, keyed by the rank owning the memory bound to them.
// Constructing a pool object takes a reference to the endpoint of the given rank, creating
// it if no one uses it yet; the endpoint is destroyed once the last reference is gone.
class UnicastLogicalEndpointPool {
    private:
        static inline std::map<int, std::weak_ptr<UnicastLogicalEndpoint> > pool;
        // references held by the pool itself, see setRetainEndpoints()
        static inline std::map<int, std::shared_ptr<UnicastLogicalEndpoint> > retainedEndpoints;
        static inline bool retainEndpoints = false;

        std::shared_ptr<UnicastLogicalEndpoint> endpoint;

    public:
        UnicastLogicalEndpointPool(int MPIrank);

        // Creating an endpoint takes a collective handshake, so under the reuse strategy the
        // pool keeps the endpoints alive even while no allocation refers to them. Endpoints
        // retained this way live until clear() is called.
        static void setRetainEndpoints(bool _retainEndpoints);
        static void clear();

#if CUDA_VERSION >= 13030
        CUlogicalEndpointId getEndpointId() const {
            return endpoint->getEndpointId();
        }

        size_t allocateRange(size_t size) {
            return endpoint->allocateRange(size);
        }

        void freeRange(size_t offset, size_t size) {
            endpoint->freeRange(offset, size);
        }
#endif
};

class MemoryAllocationCFT : public MemoryAllocation {
protected:
    CUmemAllocationProp memProp = {};
    CUmemGenericAllocationHandle handle = {};
    size_t roundedUpAllocationSize;

#if CUDA_VERSION >= 13030
    CUlogicalEndpointId endpointId = {};
    size_t endpointOffset = 0;
#endif

public:
    MemoryAllocationCFT() {};
    ~MemoryAllocationCFT() {};

#if CUDA_VERSION >= 13030
    CUlogicalEndpointId getEndpointId() override {
        return endpointId;
    }

    size_t getEndpointOffset() override {
        return endpointOffset;
    }
#endif

    size_t getRoundedAllocationSize(size_t _allocationSize);
    void *allocateMemory(size_t _allocationSize);
    void freeMemory(void *ptr);
};

class MultinodeMemoryAllocationUnicastCFT : public MemoryAllocationCFT {
private:
    UnicastLogicalEndpointPool endpointPool;
public:
    MultinodeMemoryAllocationUnicastCFT(size_t _allocationSize, int _MPIrank);
    ~MultinodeMemoryAllocationUnicastCFT();

    static bool filter();

    static std::string getName() {
        return "device";
    }
};

class MultinodeMemoryAllocationMulticastCFTRef;

class MultinodeMemoryAllocationMulticastCFT : public MemoryAllocationCFT {
    void *unicastMappingPtr = nullptr;

public:
    MultinodeMemoryAllocationMulticastCFT(size_t _allocationSize, int _MPIrank);
    ~MultinodeMemoryAllocationMulticastCFT();

    static bool filter();

    static std::string getName() {
        return "multicast";
    }

    virtual void memset(int value, CopyType copyType = COPY_TYPE_CE, MemoryPurpose memoryPurpose = MemoryPurpose::MEMORY_SOURCE);
    virtual unsigned long long check(int value, CopyType copyType = COPY_TYPE_CE, int iterations = 0);

    std::unique_ptr<MultinodeMemoryAllocationMulticastCFTRef> createRef(int chunkIndex, size_t chunkSize, int _MPIrank);

    // Only used for multicast pool
    static void setCapacityHint(int _) {};
};

class MultinodeMemoryAllocationMulticastRef : public MemoryAllocation {
public:
    void *unicastMappingPtr = nullptr;

    MultinodeMemoryAllocationMulticastRef() = default;
    MultinodeMemoryAllocationMulticastRef(void *_ptr, void *_unicastMappingPtr, size_t _allocationSize, int _MPIrank){
        ptr = _ptr;
        allocationSize = _allocationSize;
        MPIrank = _MPIrank;
        unicastMappingPtr = _unicastMappingPtr;
    };
    ~MultinodeMemoryAllocationMulticastRef(){};

    static bool filter();
    static std::string getName() {
        return "multicast";
    }

    virtual void memset(int value, CopyType copyType = COPY_TYPE_CE, MemoryPurpose memoryPurpose = MemoryPurpose::MEMORY_SOURCE);
    virtual unsigned long long check(int value, CopyType copyType = COPY_TYPE_CE, int iterations = 0);

    // Only used for multicast pool
    static void setCapacityHint(int _) {};
};

// CFT allocations are addressed by (endpointId, offset) rather than by pointer,
// so a suballocation keeps its parent's endpoint plus a byte offset into it
class MultinodeMemoryAllocationMulticastCFTRef : public MultinodeMemoryAllocationMulticastRef {
    MemoryAllocationCFT *parent = nullptr;
    size_t endpointOffset = 0;

public:
    MultinodeMemoryAllocationMulticastCFTRef(MemoryAllocationCFT *_parent, size_t _endpointOffset, void *_unicastMappingPtr, size_t _allocationSize, int _MPIrank)
        : MultinodeMemoryAllocationMulticastRef(nullptr, _unicastMappingPtr, _allocationSize, _MPIrank) {
        parent = _parent;
        endpointOffset = _endpointOffset;
    }
    ~MultinodeMemoryAllocationMulticastCFTRef() {};

    static bool filter();

#if CUDA_VERSION >= 13030
    CUlogicalEndpointId getEndpointId() override {
        return parent->getEndpointId();
    }

    size_t getEndpointOffset() override {
        return endpointOffset;
    }
#endif
};

class MultinodeMemoryAllocationMulticast : public MultinodeMemoryAllocationMulticastRef {
private:
    CUmemGenericAllocationHandle handle = {};
    CUmemGenericAllocationHandle multicastHandle = {};
    CUmemFabricHandle fh = {};
    CUmemAllocationHandleType handleType = {};
    CUmulticastObjectProp multicastProp = {};
    CUmemAccessDesc desc = {};
    size_t roundedUpAllocationSize;
    bool useCudaPool = false;

    void createMulticastGroup();
    void allocateMemory();
    void allocateMemoryCudaPool();
    void releaseMemory();
    void releaseMemoryCudaPool();
    void mapMemory();
public:
    MultinodeMemoryAllocationMulticast(size_t _allocationSize, int _MPIrank, bool useCudaPool = false);
    ~MultinodeMemoryAllocationMulticast();

    std::unique_ptr<MultinodeMemoryAllocationMulticastRef> createRef(int chunkIndex, size_t chunkSize, int _MPIrank);
};

class MultinodeMemoryAllocationMulticastCudaPool : public MultinodeMemoryAllocationMulticast {
public:
    MultinodeMemoryAllocationMulticastCudaPool(size_t _allocationSize, int _MPIrank) : MultinodeMemoryAllocationMulticast(_allocationSize, _MPIrank, true) {};
    ~MultinodeMemoryAllocationMulticastCudaPool() {};
};

class MultinodeMemoryPoolAllocationBase : public MemoryAllocation {
private:
public:
    MultinodeMemoryPoolAllocationBase(size_t _allocationSize, int _MPIrank, CUmemLocationType location);
    virtual ~MultinodeMemoryPoolAllocationBase();
};

class MultinodeMemoryPoolAllocationUnicast : public MultinodeMemoryPoolAllocationBase {
public:
    MultinodeMemoryPoolAllocationUnicast(size_t _allocationSize, int _MPIrank) : MultinodeMemoryPoolAllocationBase(_allocationSize, _MPIrank, CU_MEM_LOCATION_TYPE_DEVICE) {};
    static bool filter();

    static std::string getName() {
        return "device";
    }
};

class MultinodeMemoryPoolAllocationEGM : public MultinodeMemoryPoolAllocationBase {
public:
    MultinodeMemoryPoolAllocationEGM(size_t _allocationSize, int _MPIrank) : MultinodeMemoryPoolAllocationBase(_allocationSize, _MPIrank, CU_MEM_LOCATION_TYPE_HOST_NUMA) {};
    static bool filter();

    static std::string getName() {
        return "egm";
    }
};

template<class T>
class AllocationPool : public MemoryAllocation {
private:
    static inline std::multimap< std::pair<size_t, int>, T* > pool;
    T *current;
public:
    AllocationPool(size_t _allocationSize, int _MPIrank);
    ~AllocationPool();

    static void clear();
    static bool filter();

    static std::string getName();

    void memset(int value, CopyType copyType = COPY_TYPE_CE, MemoryPurpose memoryPurpose = MemoryPurpose::MEMORY_SOURCE) {current->memset(value, copyType, memoryPurpose);}
    unsigned long long check(int value, CopyType copyType = COPY_TYPE_CE, int iterations = 0) {return current->check(value, copyType, iterations);}

    std::vector<MemoryChunk> getChunks() override {
        return current->getChunks();
    }

    int getNumLocalityDomains() const override {
        return current->getNumLocalityDomains();
    }
#if CUDA_VERSION >= 13030
    CUlogicalEndpointId getEndpointId() override { return current->getEndpointId(); }
    size_t getEndpointOffset() override { return current->getEndpointOffset(); }
#endif
};

template<class T, class TRef>
class MulticastPool : public MemoryAllocation {
private:
    static inline std::multimap<size_t,
        std::pair<std::shared_ptr<T>, int> > pool;
    static const int defaultInsertCount = 1;
    static inline int insertCount = defaultInsertCount;
    std::unique_ptr<TRef> current;
    std::shared_ptr<T> parent;
    int offset;
public:
    static void setCapacityHint(int _insertCount) {insertCount = _insertCount;}
    MulticastPool(size_t _allocationSize, int _MPIrank);
    ~MulticastPool();

    static void clear();
    static bool filter();

    static std::string getName();

    void memset(int value, CopyType copyType = COPY_TYPE_CE, MemoryPurpose memoryPurpose = MemoryPurpose::MEMORY_SOURCE) {current->memset(value, copyType, memoryPurpose);}
    unsigned long long check(int value, CopyType copyType = COPY_TYPE_CE, int iterations = 0) {return current->check(value, copyType, iterations);}
#if CUDA_VERSION >= 13030
    CUlogicalEndpointId getEndpointId() override { return current->getEndpointId(); }
    size_t getEndpointOffset() override { return current->getEndpointOffset(); }
#endif
};

template class AllocationPool<MultinodeMemoryAllocationUnicast>;
template class AllocationPool<MultinodeMemoryAllocationEGM>;
template class AllocationPool<HostMemoryAllocation>;
template class AllocationPool<MultinodeMemoryAllocationUnicastCFT>;
template class MulticastPool<MultinodeMemoryAllocationMulticast, MultinodeMemoryAllocationMulticastRef>;
template class MulticastPool<MultinodeMemoryAllocationMulticastCFT, MultinodeMemoryAllocationMulticastCFTRef>;
template class AllocationPool<LocalizedMultinodeMemoryAllocationUnicast>;

void clearAllocationPools();

#endif
