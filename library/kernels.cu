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

#include "kernels.cuh"
#include "error.h"
#include "nvloom.h"
#include <curand_kernel.h>

const unsigned int numThreadPerBlock = 512;

template<typename T>
using write_to_memory = void (*)(T*, T);

template<typename T>
using reduce_from_memory = void (*)(T*, T*);

template<typename T>
__device__ void write_to_regular_memory(T *dst, T val) {
    *dst = val;
}

template<typename T>
__device__ void write_to_multicast_memory(T *dst, T val) {
#if __CUDA_ARCH__ >= 900
    static_assert(sizeof(T) == 4, "");
    asm ("multimem.st.weak.global.b32 [%0], %1;" : : "l"(dst), "r"(val) : "memory" );
#endif
}

template<typename T>
__device__ void reduce_from_multicast_ld_reduce(T *dst, T *val) {
#if __CUDA_ARCH__ >= 900
    static_assert(sizeof(T) == 4, "");
    uint result;
    asm ("multimem.ld_reduce.weak.global.add.u32 %0, [%1];" : "=r"(result) : "l"(val) : "memory");
    *dst = result;
#endif
}

template<typename T>
__device__ void reduce_from_multicast_red(T *dst, T *val) {
#if __CUDA_ARCH__ >= 900
    static_assert(sizeof(T) == 4, "");
    asm ("multimem.red.global.add.u32 [%0], %1;" : "+l"(dst) : "r"(*val) : "memory");
#endif
}

template<typename T, write_to_memory<T> write>
__global__ void stridingMemcpyKernel(unsigned int totalThreadCount, unsigned long long loopCount, T* dst, T* src, size_t sizeInElement) {
    T *dstEnd = dst + sizeInElement;
    size_t chunkSizeInElement = sizeInElement / totalThreadCount;

    size_t globalThreadId = blockDim.x * blockIdx.x + threadIdx.x;
    dst += globalThreadId;
    src += globalThreadId;

    // Calculate where to end the big pipelined copy
    size_t bigChunkSizeInElement = chunkSizeInElement / 12;
    T *dstBigEnd = dst + (bigChunkSizeInElement * 12) * totalThreadCount;

    for (unsigned int i = 0; i < loopCount; i++) {
        T* cdst = dst;
        T* csrc = src;

        while (cdst < dstBigEnd) {
            T pipe_0 = *csrc; csrc += totalThreadCount;
            T pipe_1 = *csrc; csrc += totalThreadCount;
            T pipe_2 = *csrc; csrc += totalThreadCount;
            T pipe_3 = *csrc; csrc += totalThreadCount;
            T pipe_4 = *csrc; csrc += totalThreadCount;
            T pipe_5 = *csrc; csrc += totalThreadCount;
            T pipe_6 = *csrc; csrc += totalThreadCount;
            T pipe_7 = *csrc; csrc += totalThreadCount;
            T pipe_8 = *csrc; csrc += totalThreadCount;
            T pipe_9 = *csrc; csrc += totalThreadCount;
            T pipe_10 = *csrc; csrc += totalThreadCount;
            T pipe_11 = *csrc; csrc += totalThreadCount;

            write(cdst, pipe_0); cdst += totalThreadCount;
            write(cdst, pipe_1); cdst += totalThreadCount;
            write(cdst, pipe_2); cdst += totalThreadCount;
            write(cdst, pipe_3); cdst += totalThreadCount;
            write(cdst, pipe_4); cdst += totalThreadCount;
            write(cdst, pipe_5); cdst += totalThreadCount;
            write(cdst, pipe_6); cdst += totalThreadCount;
            write(cdst, pipe_7); cdst += totalThreadCount;
            write(cdst, pipe_8); cdst += totalThreadCount;
            write(cdst, pipe_9); cdst += totalThreadCount;
            write(cdst, pipe_10); cdst += totalThreadCount;
            write(cdst, pipe_11); cdst += totalThreadCount;
        }

        // Take care of copies that didn't get aligned properly
        while (cdst < dstEnd) {
            write(cdst, *csrc); cdst += totalThreadCount; csrc += totalThreadCount;
        }
    }
}

template<typename T, reduce_from_memory<T> write>
__global__ void simpleMemcpyKernel(unsigned int totalThreadCount, unsigned long long loopCount, T* dst, T* src, size_t sizeInElement) {
    T *dstEnd = dst + sizeInElement;

    size_t globalThreadId = blockDim.x * blockIdx.x + threadIdx.x;
    dst += globalThreadId;
    src += globalThreadId;

    for (unsigned int i = 0; i < loopCount; i++) {
        T* cdst = dst;
        T* csrc = src;

        while (cdst < dstEnd) {
            write(cdst, csrc); cdst += totalThreadCount; csrc += totalThreadCount;
        }
    }
}

template<typename T, auto kernel>
void launchCopyKernel(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount) {
    unsigned int totalThreadCount = NvLoom::getLocalMultiprocessorCount() * numThreadPerBlock;

    // adjust size to elements (size is multiple of MB, so no truncation here)
    size_t sizeInElement = size / sizeof(T);

    dim3 gridDim(NvLoom::getLocalMultiprocessorCount(), 1, 1);
    dim3 blockDim(numThreadPerBlock, 1, 1);
    kernel <<<gridDim, blockDim, 0, stream>>> (totalThreadCount, loopCount, (T *) dstBuffer, (T *) srcBuffer, sizeInElement);
    CUDA_ASSERT(cudaPeekAtLastError());
}

void copyKernel(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount) {
    launchCopyKernel<uint4, stridingMemcpyKernel<uint4, write_to_regular_memory> >(dstBuffer, srcBuffer, size, stream, loopCount);
}

void copyKernelMulticast(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount) {
    launchCopyKernel<uint, stridingMemcpyKernel<uint, write_to_multicast_memory> >(dstBuffer, srcBuffer, size, stream, loopCount);
}

void copyKernelMulticastLdReduce(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount) {
    launchCopyKernel<uint, simpleMemcpyKernel<uint, reduce_from_multicast_ld_reduce<uint> > >(dstBuffer, srcBuffer, size, stream, loopCount);
}

void copyKernelMulticastRed(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount) {
    launchCopyKernel<uint, simpleMemcpyKernel<uint, reduce_from_multicast_red<uint> > >(dstBuffer, srcBuffer, size, stream, loopCount);
}

__global__ void spinKernelDeviceMultistage(volatile int *latch1, volatile int *latch2, const unsigned long long timeoutClocks) {
    if (latch1) {
        register unsigned long long endTime = clock64() + timeoutClocks;
        while (!*latch1) {
            if (timeoutClocks != ~0ULL && clock64() > endTime) {
                return;
            }
        }

        *latch2 = 1;
    }

    register unsigned long long endTime = clock64() + timeoutClocks;
    while (!*latch2) {
        if (timeoutClocks != ~0ULL && clock64() > endTime) {
            break;
        }
    }
}

// Implement a 2-stage spin kernel for multi-node synchronization.
// One of the host nodes releases the first latch. Subsequently,
// the second latch is released, that is polled by all other devices
// latch1 argument is optional. If defined, kernel will spin on it until released, and then will release latch2.
// latch2 argument is mandatory. Kernel will spin on it until released.
// timeoutMs argument applies to each stage separately.
// However, since each kernel will spin on only one stage, total runtime is still limited by timeoutMs
CUresult spinKernelMultistage(volatile int *latch1, volatile int *latch2, CUstream stream, unsigned long long timeoutMs) {
    ASSERT(latch2 != nullptr);

    unsigned long long timeoutClocks = NvLoom::getLocalClockRate() * timeoutMs;
    spinKernelDeviceMultistage<<<1, 1, 0, stream>>>(latch1, latch2, timeoutClocks);
    CUDA_ASSERT(cudaPeekAtLastError());

    return CUDA_SUCCESS;
}

__global__ void patternFillKernel(uint* dst, int seed, size_t bufferSize, int groupId, int groupSize) {
    unsigned long long threadId = blockDim.x * blockIdx.x + threadIdx.x;
    size_t totalThreadCount = gridDim.x * blockDim.x;
    char* dstEnd = ((char *) dst) + bufferSize;
    dst += threadId;

    curandStateXORWOW_t state;
    curand_init(seed, 0, threadId, &state);

    for (int i = 0; i < groupId; i++) {
        curand(&state);
    }

    while ((char *) dst < dstEnd) {
        *dst = curand(&state);
        dst += totalThreadCount;

        for (int i = 0; i < groupSize - 1; i++) {
            curand(&state);
        }
    }
}

void memsetBuffer(void *ptr, int seed, size_t size, CUstream stream, int groupId, int groupSize) {
    dim3 gridDim(NvLoom::getLocalMultiprocessorCount(), 1, 1);
    dim3 blockDim(numThreadPerBlock, 1, 1);
    patternFillKernel<<<gridDim, blockDim, 0, stream>>>((uint *)ptr, seed, size, groupId, groupSize);
    CUDA_ASSERT(cudaPeekAtLastError());
}

void zeroOutBuffer(void *ptr, size_t size, CUstream stream) {
    CU_ASSERT(cuMemsetD8Async((CUdeviceptr) ptr, 0, size, stream));
}

void memsetBuffer(void *ptr, int seed, size_t size, CUstream stream, CopyType copyType, MemoryPurpose memoryPurpose) {
    if (copyType == COPY_TYPE_MULTICAST_LD_REDUCE) {
        memsetBuffer(ptr, seed, size, stream, MPIWrapper::getWorldRank(), MPIWrapper::getWorldSize());
    } else if (copyType == COPY_TYPE_MULTICAST_RED_ALL) {
        if (memoryPurpose == MemoryPurpose::MEMORY_SOURCE) {
            memsetBuffer(ptr, seed, size, stream, MPIWrapper::getWorldRank(), MPIWrapper::getWorldSize());
        } else {
            zeroOutBuffer(ptr, size, stream);
        }
    } else if (copyType == COPY_TYPE_MULTICAST_RED_SINGLE) {
        if (memoryPurpose == MemoryPurpose::MEMORY_SOURCE) {
            memsetBuffer(ptr, seed, size, stream, 0, 1);
        } else {
            zeroOutBuffer(ptr, size, stream);
        }
    } else {
        memsetBuffer(ptr, seed, size, stream, 0, 1);
    }
}

__global__ void patternCheckKernel(uint* buffer, int seed, size_t bufferSize, unsigned long long *errorCount, int groupSize, int multiplier) {
    uint* originalBuffer = buffer;
    unsigned long long threadId = blockDim.x * blockIdx.x + threadIdx.x;
    size_t totalThreadCount = gridDim.x * blockDim.x;
    char* bufferEnd = ((char *) buffer) + bufferSize;
    buffer += threadId;

    curandStateXORWOW_t state;
    curand_init(seed, 0, threadId, &state);

    while ((char *) buffer < bufferEnd) {
        uint expectedValue = 0;

        for (int i = 0; i < groupSize; i++) {
            // overflow for uint is well defined
            expectedValue += curand(&state);
        }

        expectedValue *= multiplier;

        uint actualValue = *buffer;
        if (actualValue != expectedValue) {
            printf("Error found at byte offset %llu: expected %u but got %u\n", (char *) buffer - (char *) originalBuffer, expectedValue, actualValue);
            atomicAdd(errorCount, 1);
            // Only report one error per thread to avoid spamming prints
            break;
        }
        buffer += totalThreadCount;
    }
}

unsigned long long checkBuffer(void *ptr, int seed, size_t size, CUstream stream, int groupSize, int multiplier = 1) {
    unsigned long long *errorCount;
    CU_ASSERT(cuMemAlloc((CUdeviceptr *) &errorCount, sizeof(*errorCount)));
    CU_ASSERT(cuMemsetD8((CUdeviceptr) errorCount, 0, sizeof(*errorCount)));

    dim3 gridDim(NvLoom::getLocalMultiprocessorCount(), 1, 1);
    dim3 blockDim(numThreadPerBlock, 1, 1);
    patternCheckKernel<<<gridDim, blockDim, 0, stream>>>((uint *)ptr, seed, size, errorCount, groupSize, multiplier);
    CUDA_ASSERT(cudaPeekAtLastError());
    CU_ASSERT(cuStreamSynchronize(stream));

    unsigned long long errorCountCopy;
    CU_ASSERT(cuMemcpy((CUdeviceptr) &errorCountCopy, (CUdeviceptr) errorCount, sizeof(errorCountCopy)));

    CU_ASSERT(cuMemFree((CUdeviceptr) errorCount));

    return errorCountCopy;
}

unsigned long long checkBuffer(void *ptr, int seed, size_t size, CUstream stream, CopyType copyType, int iterations) {
    if (copyType == COPY_TYPE_MULTICAST_LD_REDUCE) {
        return checkBuffer(ptr, seed, size, stream, MPIWrapper::getWorldSize(), 1);
    } else if (copyType == COPY_TYPE_MULTICAST_RED_ALL) {
        return checkBuffer(ptr, seed, size, stream, MPIWrapper::getWorldSize(), iterations);
    } else if (copyType == COPY_TYPE_MULTICAST_RED_SINGLE) {
        return checkBuffer(ptr, seed, size, stream, 1, iterations);
    } else {
        return checkBuffer(ptr, seed, size, stream, 1);
    }
}

void preloadKernels(int localDevice) {
    cudaFuncAttributes unused;
    cudaSetDevice(localDevice);
    cudaFuncGetAttributes(&unused, &stridingMemcpyKernel<uint, write_to_multicast_memory<uint> >);
    cudaFuncGetAttributes(&unused, &stridingMemcpyKernel<uint4, write_to_regular_memory<uint4> >);
    cudaFuncGetAttributes(&unused, &simpleMemcpyKernel<uint, reduce_from_multicast_ld_reduce<uint> >);
    cudaFuncGetAttributes(&unused, &simpleMemcpyKernel<uint, reduce_from_multicast_red<uint> >);
    cudaFuncGetAttributes(&unused, &spinKernelDeviceMultistage);
    cudaFuncGetAttributes(&unused, &patternFillKernel);
    cudaFuncGetAttributes(&unused, &patternCheckKernel);
}
