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

class Copy {
public:
    std::shared_ptr<MemoryAllocation> dst;
    std::shared_ptr<MemoryAllocation> src;
    CopyDirection copyDirection;
    CopyType copyType;
    int executingMPIrank;
    int iterations;

    Copy(std::shared_ptr<MemoryAllocation> _dst, std::shared_ptr<MemoryAllocation> _src, CopyDirection _copyDirection, CopyType _copyType, int _iterations = DEFAULT_ITERATIONS) :
        dst(_dst),
        src(_src),
        copyDirection(_copyDirection),
        copyType(_copyType),
        iterations(_iterations) {

        if (copyDirection == COPY_DIRECTION_READ) {
            executingMPIrank = dst->MPIrank;
        } else {
            executingMPIrank = src->MPIrank;
        }
    }
};

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

    // Launch loopCount copy iterations, unconditionally
    static void doMemcpy(CopyType copyType, CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream, unsigned long long loopCount);

    // Launch up to loopCount iterations, accounting for the spinKernel guarded loop
    // Returns the number of iterations actually launched
    static unsigned long long doMemcpyInSpinKernel(CopyType copyType, CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream, unsigned long long loopCount);

    // Launch a single copy iteration
    // Designed to be called after the spinKernel is released.
    // NOOP for non-CE copies, as they're not impacted by the spinKernel issue.
    static void doMemcpyBeyondSpinKernel(CopyType copyType, CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream);

public:
    static std::vector<double> doBenchmark(std::vector<Copy> copies);

    static int getLocalDevice() { return localDevice; };
    static CUdevice getLocalCuDevice() { return localCuDevice; };
    static CUcontext getLocalCtx() { return localCtx; };
    static int getLocalMultiprocessorCount() { return localMultiprocessorCount; };
    static int getLocalClockRate() { return localClockRate; };
    static int getLocalCpuNumaNode() { return localCpuNumaNode; };
    static std::string getLocalHostname() { return localHostname; };
    static unsigned long long getDefaultIterationCount() { return DEFAULT_ITERATIONS; };

    static std::map<std::string, std::vector<int> > getRackToProcessMap() { return rackToProcessMap; };
    static void initialize(int _localDevice, std::map<std::string, std::vector<int> > _rackToProcessMap = {});
    static void finalize();
};

std::string getRackGuid(int device);
std::string trimRackGuid(std::string rackGuid);

#endif