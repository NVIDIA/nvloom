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
#include "kernels.cuh"
#include "enums.h"
#include "nvloom.h"

class MemoryAllocation {
public:
    void *ptr = nullptr;
    size_t allocationSize = 0;
    int MPIrank = -1;
    static inline int uniqueIdCounter = 0;
    int uniqueId;

    MemoryAllocation() {
        uniqueId = uniqueIdCounter;
        uniqueIdCounter++;
    };

    ~MemoryAllocation() {};

    // default filter is NOOP. It can be overloaded by classes inheriting it
    static bool filter() {
        return true;
    };

    static std::string getName() {
        return "N/A";
    }

    // copyType and memoryPurpose are optional arguments, and are only used in multicast reduction benchmarks
    virtual void memset(int value, CopyType copyType = COPY_TYPE_CE, MemoryPurpose memoryPurpose = MemoryPurpose::MEMORY_SOURCE);
    // copyType and iterations are optional arguments, and are only used in multicast reduction benchmarks
    virtual unsigned long long check(int value, CopyType copyType = COPY_TYPE_CE, int iterations = 0);
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

class MultinodeMemoryAllocationEGM : public MultinodeMemoryAllocationBase {
public:
    MultinodeMemoryAllocationEGM(size_t _allocationSize, int _MPIrank) : MultinodeMemoryAllocationBase(_allocationSize, _MPIrank, CU_MEM_LOCATION_TYPE_HOST_NUMA) {};
    static bool filter();

    static std::string getName() {
        return "egm";
    }
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

class MultinodeMemoryAllocationMulticast : public MultinodeMemoryAllocationMulticastRef {
private:
    CUmemGenericAllocationHandle handle = {};
    CUmemGenericAllocationHandle multicastHandle = {};
    CUmemFabricHandle fh = {};
    CUmemAllocationHandleType handleType = {};
    CUmulticastObjectProp multicastProp = {};
    CUmemAccessDesc desc = {};
    size_t roundedUpAllocationSize;

public:
    MultinodeMemoryAllocationMulticast(size_t _allocationSize, int _MPIrank);
    ~MultinodeMemoryAllocationMulticast();
    CUresult tryAllocation(size_t _allocationSize, int _MPIrank);
};

class MultinodeMemoryPoolAllocationBase : public MemoryAllocation {
private:
    static inline bool devicePoolsInitialized;
    static inline bool egmPoolsInitialized;
    static inline std::vector<CUmemoryPool> device_pools;
    static inline std::vector<CUmemoryPool> egm_pools;
    std::vector<CUmemFabricHandle> fh_vector;
    CUmemLocationType mem_location;
    CUmemPoolProps poolProps = { };
    CUmemAllocationHandleType handleType = {};
    CUmemPoolPtrExportData data;
    CUmemAccessDesc desc = {};
public:
    MultinodeMemoryPoolAllocationBase(size_t _allocationSize, int _MPIrank, CUmemLocationType location);
    virtual ~MultinodeMemoryPoolAllocationBase();
    virtual std::vector<CUmemoryPool> initPoolsLazy();
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
};

class MulticastPool : public MemoryAllocation {
private:
    static inline std::multimap<size_t,
        std::pair<std::shared_ptr<MultinodeMemoryAllocationMulticast>, int> > pool;
    static const int defaultInsertCount = 1;
    static inline int insertCount = defaultInsertCount;
    std::unique_ptr<MultinodeMemoryAllocationMulticastRef> current;
    std::shared_ptr<MultinodeMemoryAllocationMulticast> parent;
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
};

template class AllocationPool<MultinodeMemoryAllocationUnicast>;
template class AllocationPool<MultinodeMemoryAllocationEGM>;
template class AllocationPool<HostMemoryAllocation>;

void clearAllocationPools();

#endif
