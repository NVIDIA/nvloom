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

#ifndef LIB_H_
#define LIB_H_

#include <vector>
#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <sstream>
#include <map>
#include <mpi.h>
#include <memory>
#include "allocators.h"
#include "enums.h"
#include "error.h"
#include "kernels.cuh"

constexpr unsigned long long WARMUP_ITERATIONS = 1;
constexpr unsigned long long DEFAULT_ITERATIONS = 16;

constexpr unsigned long long DEFAULT_LATENCY_ITERATIONS = 40000;
constexpr unsigned long long WARMUP_LATENCY_ITERATIONS = 256;
constexpr size_t DEFAULT_BUFFER_SIZE_MIB = 512;
constexpr size_t DEFAULT_TMA_CHUNK_SIZE_KB = 192;

constexpr unsigned long long LATENCY_JUMP_SIZE = 128;

class MPIWrapper {
private:
    int worldSize;
    int worldRank;

    MPIWrapper() {
        int mpiInitalized;
        MPI_Initialized(&mpiInitalized);
        if (!mpiInitalized) {
            MPI_Init(NULL, NULL);
        }

        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    }

public:
    static MPIWrapper& instance() {
        static MPIWrapper mpiWrapper;
        return mpiWrapper;
    }

    static int getWorldSize() {
        return instance().worldSize;
    }
    static int getWorldRank() {
        return instance().worldRank;
    }
};

class Copy {
public:
    std::shared_ptr<MemoryAllocation> dst;
    std::shared_ptr<MemoryAllocation> src;
    std::shared_ptr<AllocationPool<HostMemoryAllocation>> counter;
    CopyDirection copyDirection;
    CopyType copyType;
    int executingMPIrank;
    int iterations;

    Copy(std::shared_ptr<MemoryAllocation> _dst, std::shared_ptr<MemoryAllocation> _src, CopyDirection _copyDirection, CopyType _copyType, int _iterations = DEFAULT_ITERATIONS);
};

// Latency is a special kind of copy operation
// The destination MemoryAllocation is an empty allocation, only indicating which node is executing the benchmark
// The source MemoryAllocation is the memory to which access is being benchmarked
// We calculate the size of the calculation to fit all the iterations in
template <typename T>
class Latency : public Copy {
public:
    Latency(int _memoryLocation, int _executingMPIrank, unsigned long long _iterations = DEFAULT_LATENCY_ITERATIONS) :
        Copy(std::make_shared<MemoryAllocation>(_executingMPIrank),
             std::make_shared<T>((_iterations + WARMUP_LATENCY_ITERATIONS) * LATENCY_JUMP_SIZE, _memoryLocation),
             COPY_DIRECTION_READ,
             COPY_TYPE_LATENCY,
             _iterations) {}
};

class MPIOutput {
 public:
    template<typename T>
    MPIOutput& operator<<(T input) {
        if (MPIWrapper::getWorldRank() == 0) std::cout << input;
        return *this;
    }

    using StreamType = decltype(std::cout);
    MPIOutput &operator<<(StreamType &(*func)(StreamType &)) {
        if (MPIWrapper::getWorldRank() == 0) {
            func(std::cout);
        }
        return *this;
    }
};
extern MPIOutput OUTPUT;

enum class BenchmarkState {
    NOT_STARTED = 0,
    WARMUP,
    SYNCHRONIZE,
    RUNNING,
    EXTRA_WORK
};

class Benchmark {
private:
    std::vector<Copy> copies;
    std::vector<Copy> filteredCopies;
    std::vector<CUstream> filteredStreams;
    std::vector<CUevent> filteredStartEvents;
    std::vector<CUevent> filteredEndEvents;
    std::vector<std::vector<CUevent>> filteredLoadSustainmentEvents;
    std::vector<int> filteredExecutedIterations;

    std::shared_ptr<AllocationPool<HostMemoryAllocation>> blockingVarHost;
    std::shared_ptr<AllocationPool<MultinodeMemoryAllocationUnicast>> blockingVarDevice;

    std::vector<Copy> getFilteredCopies();

    // Launch loopCount copy iterations, unconditionally
    void doMemcpyWarmup(Copy& copy, CUstream hStream);

    // Launch up to loopCount iterations, accounting for the spinKernel guarded loop
    // Returns the number of iterations actually launched
    unsigned long long doMemcpyInSpinKernel(Copy& copy, CUstream hStream, unsigned long long loopCount);

    // Launch a single copy iteration
    // Designed to be called after the spinKernel is released.
    // NOOP for non-CE copies, as they're not impacted by the spinKernel issue.
    void doMemcpyBeyondSpinKernel(Copy& copy, CUstream hStream);

public:
    Benchmark(std::vector<Copy> _copies);
    ~Benchmark();

    void fillBuffers();
    void verifyBuffers();

    void blockStreams();
    void releaseStreams();

    void doWarmupCopies();
    void scheduleWorkPreBlock();
    void scheduleWorkPostBlock();
    void doLoadSustainment();

    std::vector<double> calculateBandwidths();
};

class NvLoom {
private:
    static inline int localDevice;
    static inline CUdevice localCuDevice;
    static inline CUcontext localCtx;
    static inline int localMultiprocessorCount;
    static inline int localClockRate;
    static inline int localCpuNumaNode;
    static inline std::string localHostname;
    static inline std::map<std::string, std::vector<int> > rackToProcessMap;
    static inline std::vector<int> localSMIds;
    static inline bool spinKernelsEnabled = true;
    static inline int copyKernelBlockCountOverride = 0;
    static inline int tmaChunkSize = DEFAULT_TMA_CHUNK_SIZE_KB * 1024;
    static inline std::vector<Copy> currentCopies;
    static inline std::map<int, std::string> processNameMap;
public:
    static std::vector<double> doBenchmark(std::vector<Copy> copies);

    static int getLocalDevice() { return localDevice; };
    static CUdevice getLocalCuDevice() { return localCuDevice; };
    static CUcontext getLocalCtx() { return localCtx; };
    static int getLocalMultiprocessorCount() { return localMultiprocessorCount; };
    // Number of thread blocks used by copy kernels. Defaults to min(SM count, 128),
    // but can be overridden via setCopyKernelBlockCountOverride().
    static int getCopyKernelBlockCount() { return copyKernelBlockCountOverride > 0 ? copyKernelBlockCountOverride : std::min(localMultiprocessorCount, 128); };
    static int getTmaChunkSize() { return tmaChunkSize; };
    static int getLocalClockRate() { return localClockRate; };
    static int getLocalCpuNumaNode() { return localCpuNumaNode; };
    static std::vector<int>& getLocalSMIds() { return localSMIds; };
    static std::string getLocalHostname() { return localHostname; };
    static unsigned long long getDefaultIterationCount() { return DEFAULT_ITERATIONS; };
    static unsigned long long getDefaultLatencyIterationCount() { return DEFAULT_LATENCY_ITERATIONS; };
    static size_t getDefaultBufferSizeInMiB() { return DEFAULT_BUFFER_SIZE_MIB; };
    static size_t getDefaultTMAChunkSizeInKB() { return DEFAULT_TMA_CHUNK_SIZE_KB; };

    static std::map<std::string, std::vector<int> > getRackToProcessMap() { return rackToProcessMap; };
    static void initialize(int _localDevice, std::map<std::string, std::vector<int> > _rackToProcessMap = {});
    // Enable profiling by disabling spinKernels. This may reduce accuracy of performance results.
    // spinKernels will not be launched, but spinkernel-related memory allocations and memsets will still occur to minimize the changes in behavior.
    static void disableSpinKernels() { spinKernelsEnabled = false; }
    // Override the number of thread blocks launched by copy kernels. A value <= 0 restores the default.
    static void setCopyKernelBlockCountOverride(int count) { copyKernelBlockCountOverride = count; }
    static void setTmaChunkSize(int chunkSize) { tmaChunkSize = chunkSize; };
    static void finalize();
    static bool executingCopies();
    static void setCurrentCopies(std::vector<Copy> _copies) { currentCopies = _copies; };
    static void clearCurrentCopies() { currentCopies.clear(); };
    static std::vector<std::string> dumpCurrentCopies();
    static std::string getProcessName(int rank) { return processNameMap[rank]; };
    static void setProcessName(int rank, std::string name) { processNameMap[rank] = name; };
};

std::string getRackGuid(int device);
std::string trimRackGuid(std::string rackGuid);

bool filterCopyType(CopyType copyType);

std::string getCopyDirectionName(CopyDirection copyDirection);
CopyDirection getCopyDirection(std::string name);
std::string getCopyTypeName(CopyType copyType);
CopyType getCopyType(std::string name);

std::string getDriverVersion();
std::string getCudaVersion();
int getMajorDriverVersion();

#endif