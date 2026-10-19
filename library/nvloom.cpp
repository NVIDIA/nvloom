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

#include <algorithm>
#include <iostream>
#include <numeric>
#include <iomanip>
#include <string.h>
#include <unistd.h>
#include <set>
#include "error.h"
#include "kernels.cuh"
#include "nvloom.h"
#include <cuda.h>
#include <cudaTypedefs.h>
#define NVML_NO_UNVERSIONED_FUNC_DEFS
#include <nvml.h>

Copy::Copy(std::shared_ptr<MemoryAllocation> _dst, std::shared_ptr<MemoryAllocation> _src, CopyDirection _copyDirection, CopyType _copyType, int _iterations) :
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

    counter = std::make_shared<AllocationPool<HostMemoryAllocation>>(sizeof(int), executingMPIrank);
    if (executingMPIrank == MPIWrapper::getWorldRank()) {
        *(int *) counter->ptr = (int) BenchmarkState::NOT_STARTED;
    }
}

static auto getCopiesWithUniqueSources(std::vector<Copy> copies) {
    auto comp = [](Copy a, Copy b) {return a.src.get() < b.src.get();};
    std::set<Copy, decltype(comp) > uniqueSources(comp);
    for (auto &copy : copies) {
        uniqueSources.insert(copy);
    }
    return uniqueSources;
}

static auto getCopiesWithUniqueDestinations(std::vector<Copy> copies) {
    auto comp = [](Copy a, Copy b) {return a.dst.get() < b.dst.get();};
    std::set<Copy, decltype(comp) > uniqueDestinations(comp);
    for (auto &copy : copies) {
        uniqueDestinations.insert(copy);
    }
    return uniqueDestinations;
}

static bool isEventDone(CUevent event) {
    CUresult result = cuEventQuery(event);
    if (result == CUDA_SUCCESS) {
        return true;
    } else if (result == CUDA_ERROR_NOT_READY) {
        return false;
    }
    CU_ASSERT(result);
    return false;
}

Benchmark::Benchmark(std::vector<Copy> _copies) : copies(_copies) {
    ASSERT(copies.size() > 0);
    filteredCopies = getFilteredCopies();
    filteredStreams.resize(filteredCopies.size());
    filteredStartEvents.resize(filteredCopies.size());
    filteredEndEvents.resize(filteredCopies.size());
    filteredLoadSustainmentEvents.resize(filteredCopies.size());
    filteredExecutedIterations.resize(filteredCopies.size());

    NvLoom::setCurrentCopies(filteredCopies);

    int ceCopies = 0;
    int nonCeCopies = 0;
    for (auto &copy : filteredCopies) {
        if (copy.copyType == COPY_TYPE_CE) {
            ceCopies++;
        } else if (copy.copyType == COPY_TYPE_LOCALIZED_CE) {
#if CUDA_VERSION >= 13040
            // Localized CE copies are split into one copy per locality domain, count each only once
            if (std::static_pointer_cast<MemoryChunkAllocation>(copy.src)->localityDomainId == 0) {
                ceCopies++;
            }
#endif
        } else {
            nonCeCopies++;
        }
    }
    if (ceCopies > 1) {
        std::cerr << "WARNING: Process " << MPIWrapper::getWorldRank() << " is executing " << ceCopies << " CE copies at the same time" << std::endl;
    }
    if (nonCeCopies > 1) {
        std::cerr << "WARNING: Process " << MPIWrapper::getWorldRank() << " is executing " << nonCeCopies << " non-CE copies at the same time" << std::endl;
    }

    blockingVarHost = std::make_shared<AllocationPool<HostMemoryAllocation>>(sizeof(int), copies[0].executingMPIrank);
    blockingVarDevice = std::make_shared<AllocationPool<MultinodeMemoryAllocationUnicast>>(sizeof(int), copies[0].executingMPIrank);

    for (int i = 0; i < filteredCopies.size(); i++) {
        CU_ASSERT(cuStreamCreate(&filteredStreams[i], CU_STREAM_NON_BLOCKING));

        CU_ASSERT(cuEventCreate(&filteredStartEvents[i], CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&filteredEndEvents[i], CU_EVENT_DEFAULT));

        filteredLoadSustainmentEvents[i].resize(2);
        for (int j = 0; j < 2; j++) {
            CU_ASSERT(cuEventCreate(&filteredLoadSustainmentEvents[i][j], CU_EVENT_DEFAULT));
        }
    }

    if (MPIWrapper::getWorldRank() == copies[0].executingMPIrank) {
        *((int *) (blockingVarHost->ptr)) = 0;
        CU_ASSERT(cuMemsetD32((CUdeviceptr) blockingVarDevice->ptr, 0, 1));
    }
}

Benchmark::~Benchmark() {
    for (int i = 0; i < filteredCopies.size(); i++) {
        CU_ASSERT(cuEventDestroy(filteredStartEvents[i]));
        CU_ASSERT(cuEventDestroy(filteredEndEvents[i]));

        for (int j = 0; j < 2; j++) {
            CU_ASSERT(cuEventDestroy(filteredLoadSustainmentEvents[i][j]));
        }

        CU_ASSERT(cuStreamDestroy(filteredStreams[i]));
    }

    NvLoom::clearCurrentCopies();
}

std::vector<Copy> Benchmark::getFilteredCopies() {
    std::vector<Copy> result;
    for (auto copy : copies) {
        if (MPIWrapper::getWorldRank() == copy.executingMPIrank) {
            result.push_back(copy);
        }
    }
    return result;
}

void Benchmark::verifyBuffers() {
    for (auto &copy: getCopiesWithUniqueDestinations(copies)) {
        // TODO this doesn't take into account the extra work for multicast reductions
        ASSERT(0 == copy.dst->check(copy.src->uniqueId, copy.copyType, WARMUP_ITERATIONS + copy.iterations));
    }
}

void Benchmark::fillBuffers() {
    for (auto &copy: getCopiesWithUniqueSources(copies)) {
        copy.src->memset(copy.src->uniqueId, copy.copyType, MemoryPurpose::MEMORY_SOURCE);
    }
    for (auto &copy: getCopiesWithUniqueDestinations(copies)) {
        copy.dst->memset(copy.dst->uniqueId, copy.copyType, MemoryPurpose::MEMORY_DESTINATION);
    }
}

void Benchmark::doMemcpyWarmup(Copy& copy, CUstream hStream) {
    if (copy.copyType == COPY_TYPE_CE || copy.copyType == COPY_TYPE_LOCALIZED_CE) {
        for (int iter = 0; iter < WARMUP_ITERATIONS; iter++) {
            CU_ASSERT(cuMemcpyAsync((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream));
        }
    } else if (copy.copyType == COPY_TYPE_SM) {
        copyKernel((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, WARMUP_ITERATIONS);
    } else if (copy.copyType == COPY_TYPE_MULTICAST_WRITE) {
        copyKernelMulticast((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, WARMUP_ITERATIONS);
    } else if (copy.copyType == COPY_TYPE_MULTICAST_LD_REDUCE) {
        copyKernelMulticastLdReduce((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, WARMUP_ITERATIONS);
    } else if (copy.copyType == COPY_TYPE_MULTICAST_RED_ALL || copy.copyType == COPY_TYPE_MULTICAST_RED_SINGLE) {
        copyKernelMulticastRed((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, WARMUP_ITERATIONS);
    } else if (copy.copyType == COPY_TYPE_LATENCY) {
        pointerChase((CUdeviceptr) copy.src->ptr, copy.src->allocationSize, WARMUP_LATENCY_ITERATIONS, 0, 0, hStream);
    } else if (copy.copyType == COPY_TYPE_TMA) {
        copyKernelTma((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, WARMUP_ITERATIONS);
    } else if (copy.copyType == COPY_TYPE_TMA_MULTICAST_WRITE) {
        copyKernelMulticastTma((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, WARMUP_ITERATIONS);
    } else if (copy.copyType == COPY_TYPE_TMA_MULTICAST_RED_ALL || copy.copyType == COPY_TYPE_TMA_MULTICAST_RED_SINGLE) {
        copyKernelMulticastRedTma((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, WARMUP_ITERATIONS);
#if CUDA_VERSION >= 13030
    } else if (copy.copyType == COPY_TYPE_CFT_READ) {
        copyKernelCFTRead((CUdeviceptr) copy.dst->ptr, copy.src->getEndpointId(), copy.src->getEndpointOffset(), copy.src->allocationSize, hStream, WARMUP_ITERATIONS);
    } else if (copy.copyType == COPY_TYPE_CFT_WRITE) {
        copyKernelCFTWrite(copy.dst->getEndpointId(), copy.dst->getEndpointOffset(), (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, WARMUP_ITERATIONS);
    } else if (copy.copyType == COPY_TYPE_CFT_WRITE_MULTICAST) {
        copyKernelCFTWriteMulticast(copy.dst->getEndpointId(), copy.dst->getEndpointOffset(), (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, WARMUP_ITERATIONS);
    } else if (copy.copyType == COPY_TYPE_CFT_MULTICAST_RED_SINGLE || copy.copyType == COPY_TYPE_CFT_MULTICAST_RED_ALL) {
        copyKernelCFTReduceMulticast(copy.dst->getEndpointId(), copy.dst->getEndpointOffset(), (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, WARMUP_ITERATIONS);
    } else if (copy.copyType == COPY_TYPE_CFT_MULTICAST_PULLRED) {
        copyKernelCFTPullredMulticast((CUdeviceptr) copy.dst->ptr, copy.src->getEndpointId(), copy.src->getEndpointOffset(), copy.src->allocationSize, hStream, WARMUP_ITERATIONS);
#endif
    } else {
        ASSERT(0);
    }
}

unsigned long long Benchmark::doMemcpyInSpinKernel(Copy& copy, CUstream hStream, unsigned long long loopCount) {
    if (copy.copyType == COPY_TYPE_CE || copy.copyType == COPY_TYPE_LOCALIZED_CE) {
        // We're only launching 128 iterations in the spinkernel guarded loop to avoid possibility of a deadlock
        loopCount = std::min(loopCount, (unsigned long long) 128);
        for (int iter = 0; iter < loopCount; iter++) {
            CU_ASSERT(cuMemcpyAsync((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream));
        }
    } else if (copy.copyType == COPY_TYPE_SM) {
        copyKernel((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, loopCount);
    } else if (copy.copyType == COPY_TYPE_MULTICAST_WRITE) {
        copyKernelMulticast((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, loopCount);
    } else if (copy.copyType == COPY_TYPE_MULTICAST_LD_REDUCE) {
        copyKernelMulticastLdReduce((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, loopCount);
    } else if (copy.copyType == COPY_TYPE_MULTICAST_RED_ALL || copy.copyType == COPY_TYPE_MULTICAST_RED_SINGLE) {
        copyKernelMulticastRed((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, loopCount);
    } else if (copy.copyType == COPY_TYPE_LATENCY) {
        auto smIds = NvLoom::getLocalSMIds();
        unsigned long long iterationsSoFar = WARMUP_LATENCY_ITERATIONS;
        for (int j = 0; j < smIds.size(); j++) {
            unsigned long long iterations = loopCount/smIds.size() + (j < loopCount % smIds.size() ? 1 : 0);
            pointerChase((CUdeviceptr) copy.src->ptr, copy.src->allocationSize, iterations, iterationsSoFar, smIds[j], hStream);
            iterationsSoFar += iterations;
        }
        ASSERT(iterationsSoFar == loopCount + WARMUP_LATENCY_ITERATIONS);
    } else if (copy.copyType == COPY_TYPE_TMA) {
        copyKernelTma((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, loopCount);
    } else if (copy.copyType == COPY_TYPE_TMA_MULTICAST_WRITE) {
        copyKernelMulticastTma((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, loopCount);
    } else if (copy.copyType == COPY_TYPE_TMA_MULTICAST_RED_ALL || copy.copyType == COPY_TYPE_TMA_MULTICAST_RED_SINGLE) {
        copyKernelMulticastRedTma((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, loopCount);
#if CUDA_VERSION >= 13030
    } else if (copy.copyType == COPY_TYPE_CFT_READ) {
        copyKernelCFTRead((CUdeviceptr) copy.dst->ptr, copy.src->getEndpointId(), copy.src->getEndpointOffset(), copy.src->allocationSize, hStream, loopCount);
    } else if (copy.copyType == COPY_TYPE_CFT_WRITE) {
        copyKernelCFTWrite(copy.dst->getEndpointId(), copy.dst->getEndpointOffset(), (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, loopCount);
    } else if (copy.copyType == COPY_TYPE_CFT_WRITE_MULTICAST) {
        copyKernelCFTWriteMulticast(copy.dst->getEndpointId(), copy.dst->getEndpointOffset(), (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, loopCount);
    } else if (copy.copyType == COPY_TYPE_CFT_MULTICAST_RED_SINGLE || copy.copyType == COPY_TYPE_CFT_MULTICAST_RED_ALL) {
        copyKernelCFTReduceMulticast(copy.dst->getEndpointId(), copy.dst->getEndpointOffset(), (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream, loopCount);
    } else if (copy.copyType == COPY_TYPE_CFT_MULTICAST_PULLRED) {
        copyKernelCFTPullredMulticast((CUdeviceptr) copy.dst->ptr, copy.src->getEndpointId(), copy.src->getEndpointOffset(), copy.src->allocationSize, hStream, loopCount);
#endif
    } else {
        ASSERT(0);
    }
    return loopCount;
}

void Benchmark::doMemcpyBeyondSpinKernel(Copy& copy, CUstream hStream) {
    if (copy.copyType == COPY_TYPE_CE || copy.copyType == COPY_TYPE_LOCALIZED_CE) {
        CU_ASSERT(cuMemcpyAsync((CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, hStream));
    }
}

std::vector<double> Benchmark::calculateBandwidths() {
    std::vector<double> filteredBandwidths(filteredCopies.size());
    std::vector<double> results;

    for (int i = 0; i < filteredCopies.size(); i++) {
        float elapsedMs;
        CU_ASSERT(cuEventElapsedTime(&elapsedMs, filteredStartEvents[i], filteredEndEvents[i]));
        if (filteredCopies[i].copyType == COPY_TYPE_LATENCY) {
            filteredBandwidths[i] = (double) elapsedMs * (double) 1e6 / (double)filteredCopies[i].iterations;
        } else {
            filteredBandwidths[i] = (double)filteredCopies[i].src->allocationSize * filteredCopies[i].iterations / (1000.0 * 1000.0 * elapsedMs);
        }
    }

    // exchange bandwidths
    int currentIndex = 0;
    for (auto copy : copies) {
        double exchange = 0.0;
        if (MPIWrapper::getWorldRank() == copy.executingMPIrank) {
            exchange = filteredBandwidths[currentIndex];
            currentIndex++;
        }

        MPI_Bcast(&exchange, 1, MPI_DOUBLE, copy.executingMPIrank, MPI_COMM_WORLD);
        results.push_back(exchange);
    }

    return results;
}

void Benchmark::doWarmupCopies() {
    for (int i = 0; i < filteredCopies.size(); i++) {
        Copy &copy = filteredCopies[i];
        CU_ASSERT(cuStreamWriteValue32(filteredStreams[i], (CUdeviceptr) copy.counter->ptr, (int) BenchmarkState::WARMUP, 0));
        doMemcpyWarmup(copy, filteredStreams[i]);
    }
}

void Benchmark::blockStreams() {
    for (int i = 0; i < filteredCopies.size(); i++) {
        Copy &copy = filteredCopies[i];
        CU_ASSERT(cuStreamWriteValue32(filteredStreams[i], (CUdeviceptr) copy.counter->ptr, (int) BenchmarkState::SYNCHRONIZE, 0));
        // Only rank belonging to copy[0] executes the host part of the spin kernel
        CU_ASSERT(spinKernelMultistage((MPIWrapper::getWorldRank() == copies[0].executingMPIrank) ? (volatile int *) blockingVarHost->ptr : nullptr,
                                        (volatile int *) blockingVarDevice->ptr,
                                        filteredStreams[i]));
    }
}

void Benchmark::releaseStreams() {
    if (MPIWrapper::getWorldRank() == copies[0].executingMPIrank) {
        *((int *) (blockingVarHost->ptr)) = 1;
    }
}

void Benchmark::scheduleWorkPreBlock() {
    for (int i = 0; i < filteredCopies.size(); i++) {
        Copy &copy = filteredCopies[i];
        CU_ASSERT(cuStreamWriteValue32(filteredStreams[i], (CUdeviceptr) copy.counter->ptr, (int) BenchmarkState::RUNNING, 0));
        CU_ASSERT(cuEventRecord(filteredStartEvents[i], filteredStreams[i]));

        filteredExecutedIterations[i] = doMemcpyInSpinKernel(copy, filteredStreams[i], copy.iterations);

        // If we were able to schedule all the iterations, we can record the end event and indicate we're done with the benchmarking work
        if (filteredExecutedIterations[i] == copy.iterations) {
            CU_ASSERT(cuEventRecord(filteredEndEvents[i], filteredStreams[i]));
            CU_ASSERT(cuStreamWriteValue32(filteredStreams[i], (CUdeviceptr) copy.counter->ptr, (int) BenchmarkState::EXTRA_WORK, 0));
        }
    }
}

void Benchmark::scheduleWorkPostBlock() {
    // keep adding copies to streams, one at a time, in a round robin fashion
    bool pendingWork = true;
    while (pendingWork) {
        pendingWork = false;
        for (int i = 0; i < filteredCopies.size(); i++) {
            Copy &copy = filteredCopies[i];
            if (filteredExecutedIterations[i] < copy.iterations) {
                bool isLastIter = (filteredExecutedIterations[i] + 1 == copy.iterations);
                doMemcpyBeyondSpinKernel(copy, filteredStreams[i]);
                filteredExecutedIterations[i]++;
                pendingWork = true;

                // Only record the end event if we scheduled more iterations
                // Otherwise, we might be re-recording the end event scheduled prior to barrier synchronization
                if (filteredExecutedIterations[i] == copy.iterations) {
                    CU_ASSERT(cuEventRecord(filteredEndEvents[i], filteredStreams[i]));
                    CU_ASSERT(cuStreamWriteValue32(filteredStreams[i], (CUdeviceptr) copy.counter->ptr, (int) BenchmarkState::EXTRA_WORK, 0));
                }
            }
        }
    }
}

static bool isLoadSustainmentSupported(Copy& copy) {
    return copy.copyType != COPY_TYPE_MULTICAST_RED_SINGLE && copy.copyType != COPY_TYPE_MULTICAST_RED_ALL &&
        copy.copyType != COPY_TYPE_TMA_MULTICAST_RED_SINGLE && copy.copyType != COPY_TYPE_TMA_MULTICAST_RED_ALL &&
        copy.copyType != COPY_TYPE_MULTICAST_LD_REDUCE && copy.copyType != COPY_TYPE_LATENCY &&
        copy.copyType != COPY_TYPE_CFT_MULTICAST_RED_SINGLE && copy.copyType != COPY_TYPE_CFT_MULTICAST_RED_ALL &&
        copy.copyType != COPY_TYPE_CFT_MULTICAST_PULLRED;
}

static int countCopiesWithLoadSustainmentSupport(std::vector<Copy> &copies) {
    return std::count_if(copies.begin(), copies.end(), isLoadSustainmentSupported);
}

void Benchmark::doLoadSustainment() {
    std::vector<bool> filteredStreamDone(filteredCopies.size());
    std::vector<int> filteredLoadSustainmentPendingEventId(filteredCopies.size());
    const int loadSustainmentIterations = 1;
    for (int i = 0; i < filteredCopies.size(); i++) {
        // Don't schedule extra work for copies that don't support it
        if (!isLoadSustainmentSupported(filteredCopies[i])) {
            continue;
        }
        Copy &copy = filteredCopies[i];
        filteredExecutedIterations[i] += doMemcpyInSpinKernel(copy, filteredStreams[i], loadSustainmentIterations);
        CU_ASSERT(cuEventRecord(filteredLoadSustainmentEvents[i][0], filteredStreams[i]));
        // We start with just one part of extra work in the queue, so we set the pending event id to -1
        filteredLoadSustainmentPendingEventId[i] = -1;
    }

    // We don't schedule extra work for all copies, but we still need to track them
    // Example: if we benchmark latency and additional copies at the same time,
    // we want to keep the streams busy until at least latency benchmark is done
    int pendingStreams = filteredCopies.size();
    MPI_Request barrierRequest;

    // The basic idea here it to keep the streams busy with the extra work.
    // We always want to have one part of extra work "ahead" in the queue.
    // If there's only one part of work in the queue, we add more work to it.
    // In each iteration, we have to:
    // 1. Check if any streams are done with the actual benchmarking work
    // 2. If all streams are done with the actual benchmarking work, we can signal other processes that we're done
    // 3. If all other processes are also done, we can break the loop
    // 4. If other processes are not done, we can add more work to the streams to keep them busy:
    //    - If the actual benchmarking work has just finished, we schedule the next part of the extra work
    //    - Otherwise, we use a round robin fashion to schedule the next part of the extra work
    // 5. We continue this process until all processes are done
    // This is a "busy wait" loop, querying CUDA events and MPI barriers in a loop.
    while (true) {
        if (pendingStreams > 0) {
            for (int i = 0; i < filteredCopies.size(); i++) {
                // Checking if the stream is done with the actual benchmarking work
                // If it is, we can remove it from the pending count
                if (!filteredStreamDone[i] && isEventDone(filteredEndEvents[i])) {
                    filteredStreamDone[i] = true;
                    pendingStreams--;
                }
            }
        }

        // If all streams are done with the actual benchmarking work, we can signal other processes that we're done
        // We only want to signal once, so we set pendingStreams to -1
        if (pendingStreams == 0) {
            MPI_Ibarrier(MPI_COMM_WORLD, &barrierRequest);
            pendingStreams = -1;
        }

        // Once all the streams are done with the actual work, we check whether all other processes are also done
        // If they are, we can break the loop
        if (pendingStreams == -1) {
            int flag = 0;
            MPI_Test(&barrierRequest, &flag, MPI_STATUS_IGNORE);
            if (flag) {
                break;
            }
        }

        for (int i = 0; i < filteredCopies.size(); i++) {
            // Don't schedule extra work for copies that don't support it
            if (!isLoadSustainmentSupported(filteredCopies[i])) {
                continue;
            }
            // If the stream is done with the actual benchmarking work, we can add more work to it to keep the stream busy
            if (filteredStreamDone[i]) {
                // In the first iteration, we immediately schedule the next part of the extra work, as soon as the actual benchmarking work is done
                if (filteredLoadSustainmentPendingEventId[i] == -1) {
                    filteredExecutedIterations[i] += doMemcpyInSpinKernel(filteredCopies[i], filteredStreams[i], loadSustainmentIterations);
                    CU_ASSERT(cuEventRecord(filteredLoadSustainmentEvents[i][1], filteredStreams[i]));
                    filteredLoadSustainmentPendingEventId[i] = 0;
                } else {
                    int eventId = filteredLoadSustainmentPendingEventId[i];
                    // We're reusing two events to keep the stream busy in a double buffering fashion
                    // One of the events is currently executing the extra work, while we can schedule work with the other event
                    int nextEventId = (eventId + 1) % 2;
                    // If one part of the extra work is done, we can schedule the next part
                    if (isEventDone(filteredLoadSustainmentEvents[i][eventId])) {
                        filteredLoadSustainmentPendingEventId[i] = nextEventId;
                        filteredExecutedIterations[i] += doMemcpyInSpinKernel(filteredCopies[i], filteredStreams[i], loadSustainmentIterations);
                        CU_ASSERT(cuEventRecord(filteredLoadSustainmentEvents[i][eventId], filteredStreams[i]));
                    }
                }
            }
        }
    }
}

std::vector<double> NvLoom::doBenchmarkCore(std::vector<Copy> copies) {
    Benchmark benchmarkState(copies);

    benchmarkState.fillBuffers();

    // Make sure all buffers are memseted before proceeding
    CU_ASSERT(cuCtxSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    benchmarkState.doWarmupCopies();

    if (spinKernelsEnabled) {
        benchmarkState.blockStreams();
    }

    benchmarkState.scheduleWorkPreBlock();

    MPI_Barrier(MPI_COMM_WORLD);

    benchmarkState.releaseStreams();

    benchmarkState.scheduleWorkPostBlock();

    // If there's only one copy, we don't need to do extra work
    // If there's no copies with extra work support, we don't need to do extra work
    if (copies.size() > 1 && (countCopiesWithLoadSustainmentSupport(copies) > 0)) {
        benchmarkState.doLoadSustainment();
    }

    // Make sure all the work is finished
    CU_ASSERT(cuCtxSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    benchmarkState.verifyBuffers();

    return benchmarkState.calculateBandwidths();
}

std::vector<double> NvLoom::doBenchmark(std::vector<Copy> copies) {
    if (copies.size() == 0) return {};

#if CUDA_VERSION >= 13040
    std::vector<Copy> allCopies;
    std::vector<int> localityDomainsPerCopy(copies.size(), 1);
    std::map<void*, std::shared_ptr<MemoryChunkAllocation>> srcChunkCache;

    for (int i = 0; i < copies.size(); i++) {
        if (copies[i].copyType == COPY_TYPE_LOCALIZED_CE) {
            int numLocalityDomains = copies[i].src->getNumLocalityDomains();
            ASSERT(numLocalityDomains > 1);
            localityDomainsPerCopy[i] = numLocalityDomains;

            auto srcChunks = copies[i].src->getChunks();
            auto dstChunks = copies[i].dst->getChunks();

            for (int d = 0; d < numLocalityDomains; d++) {
                void *srcPtr = srcChunks[d].ptr;

                // The cache ensures the same physical src chunk always maps to the
                // same MemoryChunkAllocation. This prevents memory access issues.
                if (srcChunkCache.find(srcPtr) == srcChunkCache.end()) {
                    srcChunkCache[srcPtr] = std::make_shared<MemoryChunkAllocation>(srcChunks[d], copies[i].src->MPIrank, copies[i].src);
                }
                auto dstChunk = std::make_shared<MemoryChunkAllocation>(dstChunks[d], copies[i].dst->MPIrank, copies[i].dst);
                allCopies.push_back(Copy(dstChunk, srcChunkCache[srcPtr], copies[i].copyDirection, COPY_TYPE_LOCALIZED_CE, copies[i].iterations));
            }
        }
        else {
            allCopies.push_back(copies[i]);
        }
    }

    std::vector<double> allCopiesResults = doBenchmarkCore(allCopies);

    std::vector<double> results(copies.size());

    int allCopiesIdx = 0;
    for (int i = 0; i < copies.size(); i++) {
        for (int d = 0; d < localityDomainsPerCopy[i]; d++) {
            results[i] += allCopiesResults[allCopiesIdx++];
        }
    }
    return results;
#endif
    return doBenchmarkCore(copies);
}

bool NvLoom::executingCopies() {
    return !currentCopies.empty();
}

std::vector<std::string> NvLoom::dumpCurrentCopies() {
    std::vector<std::string> copies;
    for (int i = 0; i < currentCopies.size(); i++) {
        Copy &copy = currentCopies[i];

        int counter = *(int *) copy.counter->ptr;

        std::string copyStatus = "not started";

        if (counter == (int) BenchmarkState::WARMUP) {
            copyStatus = "warmup";
        }
        if (counter == (int) BenchmarkState::SYNCHRONIZE) {
            copyStatus = "synchronization";
        }
        if (counter == (int) BenchmarkState::RUNNING) {
            copyStatus = "core benchmarking";
        }
        if (counter == (int) BenchmarkState::EXTRA_WORK) {
            copyStatus = "extra work";
        }
        copies.push_back("Copy from process " + std::to_string(copy.src->MPIrank) + " (" + getProcessName(copy.src->MPIrank) + ")" +
        " to process " + std::to_string(copy.dst->MPIrank) + " (" + getProcessName(copy.dst->MPIrank) + ")" +
        " with type " + getCopyTypeName(copy.copyType) +
        " and direction " + getCopyDirectionName(copy.copyDirection) +
        " during " + copyStatus +
        " executed by process " + std::to_string(copy.executingMPIrank) + " (" + getProcessName(copy.executingMPIrank) + ")");
    }
    return copies;
}

static nvmlDevice_t getNvmlDevice(int device) {
    NVML_ASSERT(nvmlInit());
    CU_ASSERT(cuInit(0));

    CUuuid uuid;
    CU_ASSERT(cuDeviceGetUuid_v2(&uuid, device));

    nvmlUUID_t arg;
    arg.version = nvmlUUID_v1;
    arg.type = NVML_UUID_TYPE_BINARY;
    memcpy(&arg.value.bytes, &uuid, sizeof(uuid));

    nvmlDevice_t nvmlDev;
    nvmlReturn_t result = nvmlDeviceGetHandleByUUIDV(&arg, &nvmlDev);
    if (result != NVML_SUCCESS) {
        char pciBusId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
        CU_ASSERT(cuDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device));
        NVML_ASSERT(nvmlDeviceGetHandleByPciBusId_v2(pciBusId, &nvmlDev));
    }

    return nvmlDev;
}

static nvmlGpuFabricInfoV_t getGpuFabricInfo(int device) {
    nvmlGpuFabricInfoV_t fabricInfo { .version = nvmlGpuFabricInfo_v2 };
    fabricInfo.state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;
    NVML_ASSERT(nvmlDeviceGetGpuFabricInfoV(getNvmlDevice(device), &fabricInfo));
    return fabricInfo;
}

#if CUDA_VERSION >= 13040
// The clique API was introduced in CUDA 13.4, so it's queried through cuGetProcAddress
// to keep nvloom working on older drivers.
static bool loadCliqueApi(PFN_cuDeviceGetCliqueCount_v13040 &pfnGetCliqueCount, PFN_cuDeviceGetCliqueInfo_v13040 &pfnGetCliqueInfo) {
    CUdriverProcAddressQueryResult symbolStatus;

    if (CUDA_SUCCESS != cuGetProcAddress("cuDeviceGetCliqueCount", (void **) &pfnGetCliqueCount, 13040, CU_GET_PROC_ADDRESS_DEFAULT, &symbolStatus) ||
        symbolStatus != CU_GET_PROC_ADDRESS_SUCCESS) {
        return false;
    }

    if (CUDA_SUCCESS != cuGetProcAddress("cuDeviceGetCliqueInfo", (void **) &pfnGetCliqueInfo, 13040, CU_GET_PROC_ADDRESS_DEFAULT, &symbolStatus) ||
        symbolStatus != CU_GET_PROC_ADDRESS_SUCCESS) {
        return false;
    }

    return true;
}

// Queries the fabric cliques the local device is part of, and stores their IDs indexed by clique type
static void queryCliqueIds() {
    auto &cliqueIds = NvLoom::getLocalCliqueIds();
    cliqueIds.clear();

    PFN_cuDeviceGetCliqueCount_v13040 pfnGetCliqueCount = nullptr;
    PFN_cuDeviceGetCliqueInfo_v13040 pfnGetCliqueInfo = nullptr;

    if (!loadCliqueApi(pfnGetCliqueCount, pfnGetCliqueInfo)) {
        // Pre-13.4 drivers only expose a single clique ID through NVML, which describes
        // pointer sharing. Logical endpoint cliques are not reported by NVML.
        auto fabricInfo = getGpuFabricInfo(NvLoom::getLocalDevice());
        if (fabricInfo.state != NVML_GPU_FABRIC_STATE_NOT_SUPPORTED) {
            cliqueIds[CU_CLIQUE_TYPE_UNICAST_POINTER] = fabricInfo.cliqueId;
            cliqueIds[CU_CLIQUE_TYPE_MULTICAST_POINTER] = fabricInfo.cliqueId;
        }
        return;
    }

    size_t count = 0;
    CU_ASSERT(pfnGetCliqueCount(&count, NvLoom::getLocalCuDevice()));
    if (count == 0) {
        return;
    }

    std::vector<CUcliqueInfo> cliqueInfo(count);
    CU_ASSERT(pfnGetCliqueInfo(&cliqueInfo[0], &count, NvLoom::getLocalCuDevice()));
    ASSERT(count <= cliqueInfo.size());

    for (size_t i = 0; i < count; i++) {
        // A device is expected to be part of at most one clique of each type
        ASSERT(cliqueIds.count(cliqueInfo[i].type) == 0);
        cliqueIds[cliqueInfo[i].type] = cliqueInfo[i].id;
    }
}

// Exchanges the local fabric clique IDs across the MPI world, and records for each clique type
// whether all processes are part of the same clique. Allocators whose clique type isn't shared by
// every process are waived in their filter(), which waives every testcase built on them.
static void exchangeCliqueIds() {
    auto &cliqueSharingMap = NvLoom::getCliqueSharingMap();
    cliqueSharingMap.clear();

    // Each clique type contributes a presence flag followed by the clique ID
    constexpr int entriesPerType = 2;
    constexpr int numTypes = sizeof(ALL_CLIQUE_TYPES) / sizeof(ALL_CLIQUE_TYPES[0]);
    const int worldSize = MPIWrapper::getWorldSize();

    std::vector<unsigned int> localEntries(numTypes * entriesPerType, 0);
    for (int i = 0; i < numTypes; i++) {
        unsigned int cliqueId;
        if (NvLoom::getLocalCliqueId(ALL_CLIQUE_TYPES[i], cliqueId)) {
            localEntries[i * entriesPerType] = 1;
            localEntries[i * entriesPerType + 1] = cliqueId;
        }
    }

    std::vector<unsigned int> exchangedEntries(worldSize * numTypes * entriesPerType);
    MPI_Allgather(&localEntries[0], numTypes * entriesPerType, MPI_UNSIGNED, &exchangedEntries[0], numTypes * entriesPerType, MPI_UNSIGNED, MPI_COMM_WORLD);

    for (int i = 0; i < numTypes; i++) {
        int processesReporting = 0;
        unsigned int firstReportedId = 0;
        bool idsMatch = true;

        for (int rank = 0; rank < worldSize; rank++) {
            const unsigned int *remoteEntry = &exchangedEntries[(rank * numTypes + i) * entriesPerType];
            if (remoteEntry[0] == 0) {
                continue;
            }
            if (processesReporting == 0) {
                firstReportedId = remoteEntry[1];
            }
            processesReporting++;
            idsMatch = idsMatch && (remoteEntry[1] == firstReportedId);
        }

        if (processesReporting == 0) {
            // Nobody reports this clique type, so there's nothing to conclude from it.
            // This is the case for logical endpoint cliques on the NVML fallback path.
            cliqueSharingMap[ALL_CLIQUE_TYPES[i]] = CLIQUE_SHARING_UNKNOWN;
        } else if (processesReporting == worldSize && idsMatch) {
            cliqueSharingMap[ALL_CLIQUE_TYPES[i]] = CLIQUE_SHARING_SHARED;
        } else {
            cliqueSharingMap[ALL_CLIQUE_TYPES[i]] = CLIQUE_SHARING_SPLIT;
        }
    }
}

#endif

static int checkCliques() {
    nvmlGpuFabricInfoV_t fabricInfo = getGpuFabricInfo(NvLoom::getLocalDevice());

    // allowing running without MNNVL for development purposes
    if (fabricInfo.state == NVML_GPU_FABRIC_STATE_NOT_SUPPORTED) {
        OUTPUT << "WARNING: MNNVL fabric not available. Only single node operation available." << std::endl;
    }

    auto clusterUuidArray = std::string(MPIWrapper::getWorldSize() * NVML_GPU_FABRIC_UUID_LEN, 0);
    MPI_Allgather(&fabricInfo.clusterUuid, NVML_GPU_FABRIC_UUID_LEN, MPI_UNSIGNED_CHAR, &clusterUuidArray[0], NVML_GPU_FABRIC_UUID_LEN, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);

    for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
        if (0 != memcmp(&fabricInfo.clusterUuid, &clusterUuidArray[i * NVML_GPU_FABRIC_UUID_LEN], NVML_GPU_FABRIC_UUID_LEN)) {
            std::cerr << "Process " << MPIWrapper::getWorldRank() << " clusterUuid=" << ((unsigned long *)&fabricInfo.clusterUuid)[0] << ";" << ((unsigned long *)&fabricInfo.clusterUuid)[1] <<
                " is different than process " << i << " clusterUuid=" << ((unsigned long *)&clusterUuidArray[i * NVML_GPU_FABRIC_UUID_LEN])[0] << ";" <<  ((unsigned long *)&clusterUuidArray[i * NVML_GPU_FABRIC_UUID_LEN])[1] << std::endl;
            ASSERT(0);
        }
    }

#if CUDA_VERSION >= 13040
    // Every copy method needs plain unicast pointers, so a world split across unicast pointer
    // cliques has nothing left to benchmark. Clique types nobody reports aren't split, which
    // keeps non-MNNVL systems running.
    if (NvLoom::isCliqueSplit(CU_CLIQUE_TYPE_UNICAST_POINTER)) {
        unsigned int cliqueId;
        bool reportedLocally = NvLoom::getLocalCliqueId(CU_CLIQUE_TYPE_UNICAST_POINTER, cliqueId);
        std::cerr << "Process " << MPIWrapper::getWorldRank();
        if (reportedLocally) {
            std::cerr << " belongs to " << getCliqueTypeName(CU_CLIQUE_TYPE_UNICAST_POINTER) << " clique " << cliqueId
                      << ", while other processes belong to different ones";
        } else {
            std::cerr << " reports no " << getCliqueTypeName(CU_CLIQUE_TYPE_UNICAST_POINTER) << " clique, while other processes do";
        }
        std::cerr << std::endl;
        ASSERT(0);
    }
#else
    auto cliqueIdArray = std::vector<unsigned int>(MPIWrapper::getWorldSize());
    MPI_Allgather(&fabricInfo.cliqueId, 1, MPI_UNSIGNED, &cliqueIdArray[0], 1, MPI_UNSIGNED, MPI_COMM_WORLD);

    for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
        if (cliqueIdArray[i] != fabricInfo.cliqueId) {
            std::cerr << "Process " << MPIWrapper::getWorldRank() << " cliqueId=" << fabricInfo.cliqueId << " is different than process " << i << " cliqueId=" << cliqueIdArray[i] << std::endl;
            ASSERT(0);
        }
    }
#endif

    return 0;
}

static void checkSystemConsistency() {
    const int propertiesCount = 4;
    std::vector<int> properties(propertiesCount);
    CU_ASSERT(cuDeviceGetAttribute(&properties[0], CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, NvLoom::getLocalCuDevice()));
    CU_ASSERT(cuDeviceGetAttribute(&properties[1], CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED, NvLoom::getLocalCuDevice()));
    CU_ASSERT(cuDeviceGetAttribute(&properties[2], CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, NvLoom::getLocalCuDevice()));
    CU_ASSERT(cuDriverGetVersion(&properties[3]));

    std::vector<std::string> propertyNames = { "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED",
                                               "CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED",
                                               "CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED",
                                               "CUDA_VERSION"};

    std::vector<int> propertiesExchange(MPIWrapper::getWorldSize() * propertiesCount);
    MPI_Allgather(&properties[0], propertiesCount, MPI_INT, &propertiesExchange[0], propertiesCount, MPI_INT, MPI_COMM_WORLD);

    for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
        for (int j = 0; j < propertiesCount; j++) {
            if (properties[j] != propertiesExchange[i * propertiesCount + j]) {
                std::cerr << "Property " << propertyNames[j] << " is inconsistent between processes " << MPIWrapper::getWorldRank() << " and " << i << ": " << properties[j] << " != " << propertiesExchange[i * propertiesCount + j] << std::endl;
                std::cerr << "Aborting due to system inconsistency" << std::endl;
                ASSERT(0);
            }
        }
    }

    std::string nvmlDriverVersion = std::string(NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE, 0);
    NVML_ASSERT(nvmlSystemGetDriverVersion(&nvmlDriverVersion[0], NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE));

    std::string nvmlDriverVersionExchange(MPIWrapper::getWorldSize() * NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE, 0);
    MPI_Allgather(&nvmlDriverVersion[0], NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE, MPI_CHAR, &nvmlDriverVersionExchange[0], NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE, MPI_CHAR, MPI_COMM_WORLD);

    for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
        if (0 != memcmp(&nvmlDriverVersion[0], &nvmlDriverVersionExchange[i * NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE], NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE)) {
            std::cerr << "Driver version is inconsistent between processes " << MPIWrapper::getWorldRank() << " and " << i << ": " << nvmlDriverVersion << " != " << std::string(&nvmlDriverVersionExchange[i * NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE], NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE) << std::endl;
            std::cerr << "Aborting due to system inconsistency" << std::endl;
            ASSERT(0);
        }
    }
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
    NVML_ASSERT(nvmlInit());

    CU_ASSERT(cuDeviceGet(&localCuDevice, localDevice));
    CU_ASSERT(cuDevicePrimaryCtxRetain(&localCtx, localCuDevice));
    CU_ASSERT(cuCtxSetCurrent(localCtx));

    CU_ASSERT(cuDeviceGetAttribute(&localCpuNumaNode, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, localCuDevice));
    CU_ASSERT(cuDeviceGetAttribute(&localMultiprocessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, localCuDevice));
    CU_ASSERT(cuDeviceGetAttribute(&localClockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, localCuDevice));

    localSMIds = getSmIds();
    localHostname = getHostname();

    preloadKernels(localDevice);
    MPIWrapper::getWorldRank();
#if CUDA_VERSION >= 13040
    queryCliqueIds();
    exchangeCliqueIds();
#endif
    checkCliques();
    checkSystemConsistency();
};

void NvLoom::finalize() {
    int initialized;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Finalize();
    }
};

MPIOutput OUTPUT;

static int getMajorComputeCapability() {
    int major;
    CU_ASSERT(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, NvLoom::getLocalDevice()));
    return major;
}

// Returns compute capability as major * 10 + minor, e.g. 107 for 10.7
static int getComputeCapability() {
    int minor;
    CU_ASSERT(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, NvLoom::getLocalDevice()));
    return getMajorComputeCapability() * 10 + minor;
}

bool filterCopyType(CopyType copyType) {
    switch (copyType) {
        case COPY_TYPE_CE:
            return true;
        case COPY_TYPE_LOCALIZED_CE:
            return getComputeCapability() >= 107;
        case COPY_TYPE_SM:
            return true;
        // If we can allocate multicast buffers, we can run multicast copies
        case COPY_TYPE_MULTICAST_WRITE:
            return true;
        case COPY_TYPE_MULTICAST_LD_REDUCE:
            return true;
        case COPY_TYPE_MULTICAST_RED_ALL:
            return true;
        case COPY_TYPE_MULTICAST_RED_SINGLE:
            return true;
        case COPY_TYPE_LATENCY:
            return true;
        case COPY_TYPE_TMA:
            return getMajorComputeCapability() >= 9;
        case COPY_TYPE_TMA_MULTICAST_WRITE:
        case COPY_TYPE_TMA_MULTICAST_RED_ALL:
        case COPY_TYPE_TMA_MULTICAST_RED_SINGLE:
            return tmaMulticastSupported() && (getMajorComputeCapability() >= 9);
        case COPY_TYPE_CFT_READ:
        case COPY_TYPE_CFT_WRITE:
        case COPY_TYPE_CFT_WRITE_MULTICAST:
        case COPY_TYPE_CFT_MULTICAST_RED_SINGLE:
        case COPY_TYPE_CFT_MULTICAST_RED_ALL:
        case COPY_TYPE_CFT_MULTICAST_PULLRED:
            return cftSupported() && (getMajorComputeCapability() >= 9);
    }
    return false;
}

#if CUDA_VERSION >= 13040
std::string getCliqueTypeName(CUcliqueType cliqueType) {
    switch (cliqueType) {
        case CU_CLIQUE_TYPE_UNICAST_POINTER: return "unicast pointer";
        case CU_CLIQUE_TYPE_MULTICAST_POINTER: return "multicast pointer";
        case CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT: return "unicast logical endpoint";
        case CU_CLIQUE_TYPE_MULTICAST_LOGICAL_ENDPOINT: return "multicast logical endpoint";
    }
    throw std::runtime_error("Invalid clique type");
}
#endif

std::string getCopyDirectionName(CopyDirection copyDirection) {
    if (copyDirection == COPY_DIRECTION_READ) return "read";
    if (copyDirection == COPY_DIRECTION_WRITE) return "write";
    throw std::runtime_error("Invalid copy direction");
}

CopyDirection getCopyDirection(std::string name) {
    if (name == "read") return COPY_DIRECTION_READ;
    if (name == "write") return COPY_DIRECTION_WRITE;
    throw std::runtime_error("Invalid copy direction");
}

std::string getCopyTypeName(CopyType copyType) {
    if (copyType == COPY_TYPE_CE) return "ce";
    if (copyType == COPY_TYPE_LOCALIZED_CE) return "localized_ce";
    if (copyType == COPY_TYPE_SM) return "sm";
    if (copyType == COPY_TYPE_MULTICAST_WRITE) return "mc";
    if (copyType == COPY_TYPE_MULTICAST_LD_REDUCE) return "mc_ld_reduce";
    if (copyType == COPY_TYPE_MULTICAST_RED_ALL || copyType == COPY_TYPE_MULTICAST_RED_SINGLE) return "mc_red";
    if (copyType == COPY_TYPE_LATENCY) return "latency";
    if (copyType == COPY_TYPE_TMA) return "tma";
    if (copyType == COPY_TYPE_TMA_MULTICAST_WRITE) return "tma_mc";
    if (copyType == COPY_TYPE_TMA_MULTICAST_RED_ALL) return "tma_mc_red_all";
    if (copyType == COPY_TYPE_TMA_MULTICAST_RED_SINGLE) return "tma_mc_red_single";
    if (copyType == COPY_TYPE_CFT_READ) return "cft";
    if (copyType == COPY_TYPE_CFT_WRITE) return "cft";
    if (copyType == COPY_TYPE_CFT_WRITE_MULTICAST) return "cft_mc";
    if (copyType == COPY_TYPE_CFT_MULTICAST_RED_SINGLE || copyType == COPY_TYPE_CFT_MULTICAST_RED_ALL) return "cft_mc_red";
    if (copyType == COPY_TYPE_CFT_MULTICAST_PULLRED) return "cft_mc_pullred";
    throw std::runtime_error("Invalid copy type");
}

CopyType getCopyType(std::string name) {
    if (name == "ce") return COPY_TYPE_CE;
    if (name == "localized_ce") return COPY_TYPE_LOCALIZED_CE;
    if (name == "sm") return COPY_TYPE_SM;
    if (name == "mc") return COPY_TYPE_MULTICAST_WRITE;
    if (name == "mc_ld_reduce") return COPY_TYPE_MULTICAST_LD_REDUCE;
    if (name == "mc_red") return COPY_TYPE_MULTICAST_RED_ALL;
    if (name == "latency") return COPY_TYPE_LATENCY;
    if (name == "tma") return COPY_TYPE_TMA;
    if (name == "tma_mc") return COPY_TYPE_TMA_MULTICAST_WRITE;
    if (name == "tma_mc_red_all") return COPY_TYPE_TMA_MULTICAST_RED_ALL;
    if (name == "tma_mc_red_single") return COPY_TYPE_TMA_MULTICAST_RED_SINGLE;
    if (name == "cft_read") return COPY_TYPE_CFT_READ;
    if (name == "cft_write") return COPY_TYPE_CFT_WRITE;
    if (name == "cft") return COPY_TYPE_CFT_READ;
    if (name == "cft_mc") return COPY_TYPE_CFT_WRITE_MULTICAST;
    if (name == "cft_mc_red_single") return COPY_TYPE_CFT_MULTICAST_RED_SINGLE;
    if (name == "cft_mc_red_all") return COPY_TYPE_CFT_MULTICAST_RED_ALL;
    if (name == "cft_mc_pullred") return COPY_TYPE_CFT_MULTICAST_PULLRED;
    throw std::runtime_error("Invalid copy type");
}

std::string getDriverVersion() {
    NVML_ASSERT(nvmlInit());
    char driverVersion[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
    NVML_ASSERT(nvmlSystemGetDriverVersion(driverVersion, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE));
    return driverVersion;
}

std::string getCudaVersion() {
    int driverVersion = 0;
    CU_ASSERT(cuDriverGetVersion(&driverVersion));
    int majorVersion = driverVersion / 1000;
    int minorVersion = (driverVersion % 1000) / 10;
    return std::to_string(majorVersion) + "." + std::to_string(minorVersion);
}

int getMajorDriverVersion() {
    // NVML returns driver version in the format "X.Y"
    // Trim it down and only return the major version
    std::string driverVersion = getDriverVersion();
    return std::stoi(driverVersion.substr(0, driverVersion.find('.')));
}
