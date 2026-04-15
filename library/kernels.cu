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
#include <cuda/ptx>
#include <algorithm>
#include <cuda_awbarrier_primitives.h>
#include <nvml.h>

constexpr unsigned int defaultNumThreadPerBlock = 512;
constexpr unsigned int multicastNumThreadPerBlock = 1024;

template<typename T>
using write_to_memory = void (*)(T*, T);

template<typename T>
using reduce_from_memory = void (*)(T*, T*);

template<typename T>
__device__ void write_to_regular_memory(T *dst, T val) {
    *dst = val;
}

template<>
__device__ void write_to_regular_memory<uint4>(uint4 *dst, uint4 val) {
    *dst = {val.x, val.y, val.z, val.w};
}

template<typename T>
__device__ void write_to_multicast_memory(T *dst, T val) {
#if __CUDA_ARCH__ >= 900
    cuda::ptx::multimem_st(cuda::ptx::sem_weak, dst, val);
#endif
}

template<typename T>
__device__ void reduce_from_multicast_ld_reduce(T *dst, T *val) {
#if __CUDA_ARCH__ >= 900
    *dst = cuda::ptx::multimem_ld_reduce(cuda::ptx::sem_weak, cuda::ptx::op_add, val);
#endif
}

template<typename T>
__device__ void reduce_from_multicast_red(T *dst, T *val) {
#if __CUDA_ARCH__ >= 900
    cuda::ptx::multimem_red(cuda::ptx::sem_relaxed, cuda::ptx::scope_cta, cuda::ptx::op_add, dst, *val);
#endif
}

template<typename T, write_to_memory<T> write, int UNROLL_FACTOR = 12>
__global__ void stridingMemcpyKernel(unsigned int totalThreadCount, unsigned long long loopCount, T* dst, T* src, size_t sizeInElement) {
    size_t chunkSizeInElement = sizeInElement / totalThreadCount;

    size_t globalThreadId = blockDim.x * blockIdx.x + threadIdx.x;
    dst += globalThreadId;
    src += globalThreadId;
    size_t maxElementIndex = sizeInElement - globalThreadId;

    // Calculate where to end the big pipelined copy
    size_t bigChunkSizeInElement = chunkSizeInElement / UNROLL_FACTOR;
    for (unsigned int i = 0; i < loopCount; i++) {
        for (unsigned long long j = 0; j < bigChunkSizeInElement; j++) {
            unsigned long long offset = j * UNROLL_FACTOR * totalThreadCount;

            T pipe[UNROLL_FACTOR];
            #pragma unroll
            for (int k = 0; k < UNROLL_FACTOR; ++k) {
                pipe[k] = src[offset + k * totalThreadCount];
            }
            #pragma unroll
            for (int k = 0; k < UNROLL_FACTOR; ++k) {
                write(dst + offset + k * totalThreadCount, pipe[k]);
            }
        }

        // Take care of copies that didn't get aligned properly and remainder elements
        size_t start_offset = bigChunkSizeInElement * UNROLL_FACTOR * totalThreadCount;
        for (size_t offset = start_offset; offset < maxElementIndex; offset += totalThreadCount) {
            write(dst + offset, src[offset]);
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
void launchCopyKernel(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount, unsigned int numThreadPerBlock) {
    int numBlocks = std::min(NvLoom::getLocalMultiprocessorCount(), 128);

    unsigned int totalThreadCount = numBlocks * numThreadPerBlock;

    // adjust size to elements (size is multiple of MB, so no truncation here)
    size_t sizeInElement = size / sizeof(T);

    cudaLaunchConfig_t config = {0};
    config.gridDim = dim3(numBlocks, 1, 1);
    config.blockDim = dim3(numThreadPerBlock, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute cudaAttr;
    cudaAttr.id = cudaLaunchAttributeNvlinkUtilCentricScheduling;
    cudaAttr.val.nvlinkUtilCentricScheduling = 1;
    config.attrs = &cudaAttr;
    config.numAttrs = 1;

    CUDA_ASSERT(cudaLaunchKernelEx(&config, kernel, totalThreadCount, loopCount, (T *) dstBuffer, (T *) srcBuffer, sizeInElement));
}

// TMA kernels
#define TMA_CHUNK_SIZE_BYTES 32768

#if __CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1
#define TMA_MULTICAST_SUPPORTED 1
#elif __CUDACC_VER_MAJOR__ > 13
#define TMA_MULTICAST_SUPPORTED 1
#else
#define TMA_MULTICAST_SUPPORTED 0
#endif

template <int chunk_size_bytes>
struct SMEM {
  alignas(16) char buf[chunk_size_bytes];
  alignas(16) __mbarrier_t bar;
};

#if __CUDA_ARCH__ >= 900
inline __device__ void cp_async_bulk_global_to_shared(void *dest, void * src, __mbarrier_t *barrier, int size)
{
  uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dest));
  uint64_t gmem_ptr = static_cast<uint64_t>(__cvta_generic_to_global(src));
  uint32_t smem_barrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));

  asm volatile(
    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
    :
    : "r"(smem_ptr),
      "l"(gmem_ptr),
      "r"(size),
      "r"(smem_barrier_ptr)
    : "memory");
}

inline __device__ __mbarrier_token_t barrier_arrive1_tx(__mbarrier_t *barrier, uint32_t expected_tx_count)
{
    __mbarrier_token_t token;

    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
                 : "=l"(token)
                 : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))), "r"(expected_tx_count)
                 : "memory");
    return token;
}

inline __device__ bool barrier_try_wait_token(__mbarrier_t *barrier, __mbarrier_token_t token)
{
  // This function returns a bool, so that software can retry.
  //
  //  The HW only provides best-effort waiting support. The wait time is limited
  //  by the HW capability, after which a fail occurs, in which case the SW is
  //  responsible for retrying.
  int __ready;
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "mbarrier.try_wait.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
               "selp.b32 %0, 1, 0, p;\n\t"
               "}"
               : "=r"(__ready)
               : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))),
                 "l"(token)
               : "memory");
  return __ready;
}

inline __device__ void cp_async_bulk_commit_group()
{
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

inline __device__ void cp_async_bulk_wait_all_read()
{
  // The optional .read modifier indicates that the waiting has to be done until
  // all the bulk async operations in the specified bulk async-group have
  // completed reading from their source locations.
  asm volatile("cp.async.bulk.wait_group.read 0; \n" ::: "memory");
}

inline __device__ void cp_async_bulk_wait_all()
{
  asm volatile("cp.async.bulk.wait_group 0; \n" ::: "memory");
}
#endif // __CUDA_ARCH__ >= 900

using cp_async_bulk_shared_to_global_func = void (*)(void*, void*, int);

inline __device__ void cp_async_bulk_shared_to_global(void *dest, void * src, int size)
{
#if __CUDA_ARCH__ >= 900
  uint64_t dest_gmem_ptr = static_cast<uint64_t>(__cvta_generic_to_global(dest));
  uint32_t src_smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));

  asm volatile(
    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
    :
    : "l"(dest_gmem_ptr),
      "r"(src_smem_ptr),
      "r"(size)
    : "memory");
#endif
}

inline __device__ void cp_async_bulk_shared_to_global_multicast(void *dest, void * src, int size)
{
#if __CUDA_ARCH__ >= 900
#if TMA_MULTICAST_SUPPORTED
  uint64_t dest_gmem_ptr = static_cast<uint64_t>(__cvta_generic_to_global(dest));
  uint32_t src_smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));

  asm volatile(
    "multimem.cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
    :
    : "l"(dest_gmem_ptr),
      "r"(src_smem_ptr),
      "r"(size)
    : "memory");
#endif
#endif
}

inline __device__ void cp_async_bulk_shared_to_global_multicast_red(void *dest, void * src, int size)
{
#if __CUDA_ARCH__ >= 900
#if TMA_MULTICAST_SUPPORTED
  uint64_t dest_gmem_ptr = static_cast<uint64_t>(__cvta_generic_to_global(dest));
  uint32_t src_smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));

  asm volatile(
    "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], %2;\n"
    :
    : "l"(dest_gmem_ptr),
      "r"(src_smem_ptr),
      "r"(size)
    : "memory");
#endif
#endif
}

template<size_t bytesChunk, cp_async_bulk_shared_to_global_func write>
__global__ void memcpyKernelTma(unsigned long long loopCount, char* dst, char* src, size_t size) {
#if __CUDA_ARCH__ >= 900
    assert(size % TMA_CHUNK_SIZE_BYTES == 0 && "TMA memcpy kernels require copy size to be a multiple of TMA_CHUNK_SIZE_BYTES");
    extern __shared__ char smem_bytes[];
    using SMEM_T = SMEM<bytesChunk>;
    SMEM_T &smem = reinterpret_cast<SMEM_T&>(smem_bytes);
    __mbarrier_init(&smem.bar, 1);

    assert(blockDim.x == 1 && "There is zero reason to run this kernel with more than 1 thread / block.");

    size_t block_stride = size_t(bytesChunk) * size_t(blockIdx.x);
    size_t grid_stride = size_t(gridDim.x) * size_t(bytesChunk);
    for (unsigned long long loop = 0; loop < loopCount; loop++) {
        for (size_t i = block_stride; i < size; i += grid_stride) {
            size_t bytesToCopy = bytesChunk;
            if (i + bytesToCopy > size) {
                bytesToCopy = size - i;
            }

            cp_async_bulk_global_to_shared(&smem.buf[0], src + i, &smem.bar, bytesToCopy);
            __mbarrier_token_t token = barrier_arrive1_tx(&smem.bar, bytesToCopy);

            // Wait for previous transfer. Retry in case of false. (waiting can fail)
            while (!barrier_try_wait_token(&smem.bar, token)) {}
            // Data in smem_buf is ready.

            // Write it out to peer global memory
            write(dst + i, &smem.buf[0], bytesToCopy);

            // Commits previously issued TMA instruction from POV of this thread. Read to SMEM have not yet completed.
            cp_async_bulk_commit_group();

            // This guarantees that all of the SMEM read requests have completed, even if the actual dest might be in flight
            cp_async_bulk_wait_all_read(); // NOTE: .read modifier
        }
    }

    __threadfence_system();
#endif
}

template<cp_async_bulk_shared_to_global_func write>
void launchCopyKernelTma(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount) {
    int numBlocks = std::min(NvLoom::getLocalMultiprocessorCount(), 128);

    cudaLaunchConfig_t config = {0};
    config.gridDim = dim3(numBlocks, 1, 1);
    config.blockDim = dim3(1, 1, 1);
    config.dynamicSmemBytes = sizeof(SMEM<TMA_CHUNK_SIZE_BYTES>);
    config.stream = stream;
    cudaLaunchAttribute cudaAttr;
    cudaAttr.id = cudaLaunchAttributeNvlinkUtilCentricScheduling;
    cudaAttr.val.nvlinkUtilCentricScheduling = 1;
    config.attrs = &cudaAttr;
    config.numAttrs = 1;

    CUDA_ASSERT(cudaLaunchKernelEx(&config, memcpyKernelTma<TMA_CHUNK_SIZE_BYTES, write>, loopCount, (char *) dstBuffer, (char *) srcBuffer, size));
}

void copyKernel(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount) {
    launchCopyKernel<uint4, stridingMemcpyKernel<uint4, write_to_regular_memory> >(dstBuffer, srcBuffer, size, stream, loopCount, defaultNumThreadPerBlock);
}

void copyKernelTma(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount) {
    launchCopyKernelTma<cp_async_bulk_shared_to_global>(dstBuffer, srcBuffer, size, stream, loopCount);
}

void copyKernelMulticastTma(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount) {
    launchCopyKernelTma<cp_async_bulk_shared_to_global_multicast>(dstBuffer, srcBuffer, size, stream, loopCount);
}

void copyKernelMulticastRedTma(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount) {
    launchCopyKernelTma<cp_async_bulk_shared_to_global_multicast_red>(dstBuffer, srcBuffer, size, stream, loopCount);
}

void copyKernelMulticast(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount) {
    launchCopyKernel<uint, stridingMemcpyKernel<uint, write_to_multicast_memory> >(dstBuffer, srcBuffer, size, stream, loopCount, multicastNumThreadPerBlock);
}

void copyKernelMulticastLdReduce(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount) {
    launchCopyKernel<uint, simpleMemcpyKernel<uint, reduce_from_multicast_ld_reduce<uint> > >(dstBuffer, srcBuffer, size, stream, loopCount, multicastNumThreadPerBlock);
}

void copyKernelMulticastRed(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount) {
    launchCopyKernel<uint, simpleMemcpyKernel<uint, reduce_from_multicast_red<uint> > >(dstBuffer, srcBuffer, size, stream, loopCount, multicastNumThreadPerBlock);
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
    dim3 blockDim(defaultNumThreadPerBlock, 1, 1);
    patternFillKernel<<<gridDim, blockDim, 0, stream>>>((uint *)ptr, seed, size, groupId, groupSize);
    CUDA_ASSERT(cudaPeekAtLastError());
}

void zeroOutBuffer(void *ptr, size_t size, CUstream stream) {
    CU_ASSERT(cuMemsetD8Async((CUdeviceptr) ptr, 0, size, stream));
}

void memsetBuffer(void *ptr, int seed, size_t size, CUstream stream, CopyType copyType, MemoryPurpose memoryPurpose) {
    if (copyType == COPY_TYPE_MULTICAST_LD_REDUCE) {
        memsetBuffer(ptr, seed, size, stream, MPIWrapper::getWorldRank(), MPIWrapper::getWorldSize());
    } else if (copyType == COPY_TYPE_MULTICAST_RED_ALL || copyType == COPY_TYPE_TMA_MULTICAST_RED_ALL) {
        if (memoryPurpose == MemoryPurpose::MEMORY_SOURCE) {
            memsetBuffer(ptr, seed, size, stream, MPIWrapper::getWorldRank(), MPIWrapper::getWorldSize());
        } else {
            zeroOutBuffer(ptr, size, stream);
        }
    } else if (copyType == COPY_TYPE_MULTICAST_RED_SINGLE || copyType == COPY_TYPE_TMA_MULTICAST_RED_SINGLE) {
        if (memoryPurpose == MemoryPurpose::MEMORY_SOURCE) {
            memsetBuffer(ptr, seed, size, stream, 0, 1);
        } else {
            zeroOutBuffer(ptr, size, stream);
        }
    } else if (copyType == COPY_TYPE_LATENCY) {
        pointerChaseFill(ptr, size, stream);
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
    // Cache the error count pointer to avoid reallocating it on every call
    // We can't use AllocationPool here because checkBuffer is not called uniconditionally by every single process
    // Otherwise the uniqueId of memory allocations would be different for each process, and the check would fail
    static unsigned long long *errorCount;
    if (errorCount == nullptr) {
        CU_ASSERT(cuMemAlloc((CUdeviceptr *) &errorCount, sizeof(*errorCount)));
    }
    CU_ASSERT(cuMemsetD8((CUdeviceptr) errorCount, 0, sizeof(*errorCount)));

    dim3 gridDim(NvLoom::getLocalMultiprocessorCount(), 1, 1);
    dim3 blockDim(defaultNumThreadPerBlock, 1, 1);
    patternCheckKernel<<<gridDim, blockDim, 0, stream>>>((uint *)ptr, seed, size, errorCount, groupSize, multiplier);
    CUDA_ASSERT(cudaPeekAtLastError());
    CU_ASSERT(cuStreamSynchronize(stream));

    unsigned long long errorCountCopy;
    CU_ASSERT(cuMemcpy((CUdeviceptr) &errorCountCopy, (CUdeviceptr) errorCount, sizeof(errorCountCopy)));

    return errorCountCopy;
}

unsigned long long checkBuffer(void *ptr, int seed, size_t size, CUstream stream, CopyType copyType, int iterations) {
    if (copyType == COPY_TYPE_MULTICAST_LD_REDUCE) {
        return checkBuffer(ptr, seed, size, stream, MPIWrapper::getWorldSize(), 1);
    } else if (copyType == COPY_TYPE_MULTICAST_RED_ALL || copyType == COPY_TYPE_TMA_MULTICAST_RED_ALL) {
        return checkBuffer(ptr, seed, size, stream, MPIWrapper::getWorldSize(), iterations);
    } else if (copyType == COPY_TYPE_MULTICAST_RED_SINGLE || copyType == COPY_TYPE_TMA_MULTICAST_RED_SINGLE) {
        return checkBuffer(ptr, seed, size, stream, 1, iterations);
    } else if (copyType == COPY_TYPE_LATENCY) {
        // Latency measurement is read-only, so there's nothing to verify
        return 0;
    } else {
        return checkBuffer(ptr, seed, size, stream, 1);
    }
}

__global__ void pointerChaseFillKernel(size_t* dst, size_t bufferSize) {
    unsigned long long threadId = blockDim.x * blockIdx.x + threadIdx.x;
    size_t totalThreadCount = gridDim.x * blockDim.x;
    char* dstEnd = ((char *) dst) + bufferSize;
    dst += threadId;

    while ((char *) dst < dstEnd) {
        *dst = LATENCY_JUMP_SIZE;
        dst += totalThreadCount;
    }
}

void pointerChaseFill(void *ptr, size_t size, CUstream stream) {
    dim3 gridDim(NvLoom::getLocalMultiprocessorCount(), 1, 1);
    dim3 blockDim(defaultNumThreadPerBlock, 1, 1);
    pointerChaseFillKernel<<<gridDim, blockDim, 0, stream>>>((size_t *)ptr, size);
    CUDA_ASSERT(cudaPeekAtLastError());
}

__global__ void pointerChaseKernel(size_t* dst, unsigned long long iterations, size_t offset, int targetBlock) {
    if (cuda::ptx::get_sreg_smid() != targetBlock) {
        return;
    }

    size_t *ptr = (size_t *) ((char *) dst + offset);
    for (unsigned long long i = 0; i < iterations; i++) {
        ptr = (size_t *) ((char *) ptr + *ptr);
    }

    if (ptr == 0) {
        __trap();
    }
}

void pointerChase(CUdeviceptr ptr, size_t size, unsigned long long iterations, size_t previouslyExecutedIterations, int targetBlock, CUstream stream) {
    dim3 gridDim(NvLoom::getLocalMultiprocessorCount(), 1, 1);
    dim3 blockDim(1, 1, 1);
    pointerChaseKernel<<<gridDim, blockDim, 0, stream>>>((size_t *)ptr, iterations, previouslyExecutedIterations * LATENCY_JUMP_SIZE, targetBlock);
    CUDA_ASSERT(cudaPeekAtLastError());
}

__global__ void getSmIdsKernel(int* smIds) {
    smIds[blockIdx.x] = cuda::ptx::get_sreg_smid();
}

std::vector<int> getSmIds() {
    // we can't use NvLoom::getLocalMultiprocessorCount() here because this function is called during NvLoom::initialize()
    int smCount;
    CU_ASSERT(cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, NvLoom::getLocalDevice()));

    std::vector<int> smIds(smCount);
    int *smIdsDevice;
    CU_ASSERT(cuMemAlloc((CUdeviceptr *) &smIdsDevice, sizeof(int) * smCount));
    CU_ASSERT(cuMemsetD8((CUdeviceptr) smIdsDevice, 0, sizeof(int) * smCount));

    dim3 gridDim(smCount, 1, 1);
    dim3 blockDim(1, 1, 1);
    getSmIdsKernel<<<gridDim, blockDim>>>(smIdsDevice);
    CUDA_ASSERT(cudaPeekAtLastError());
    CU_ASSERT(cuCtxSynchronize());

    CU_ASSERT(cuMemcpy((CUdeviceptr) smIds.data(), (CUdeviceptr) smIdsDevice, sizeof(int) * smCount));
    CU_ASSERT(cuMemFree((CUdeviceptr) smIdsDevice));

    std::sort(smIds.begin(), smIds.end());

    return smIds;
}

void preloadKernels(int localDevice) {
    cudaFuncAttributes unused;
    cudaSetDevice(localDevice);
    cudaFuncGetAttributes(&unused, &stridingMemcpyKernel<uint, write_to_multicast_memory<uint> >);
    cudaFuncGetAttributes(&unused, &stridingMemcpyKernel<uint4, write_to_regular_memory<uint4> >);
    cudaFuncGetAttributes(&unused, &simpleMemcpyKernel<uint, reduce_from_multicast_ld_reduce<uint> >);
    cudaFuncGetAttributes(&unused, &simpleMemcpyKernel<uint, reduce_from_multicast_red<uint> >);
    cudaFuncGetAttributes(&unused, &spinKernelDeviceMultistage);
    cudaFuncGetAttributes(&unused, &pointerChaseFillKernel);
    cudaFuncGetAttributes(&unused, &pointerChaseKernel);
    cudaFuncGetAttributes(&unused, &patternFillKernel);
    cudaFuncGetAttributes(&unused, &patternCheckKernel);
    cudaFuncGetAttributes(&unused, &getSmIdsKernel);
    cudaFuncGetAttributes(&unused, &memcpyKernelTma<TMA_CHUNK_SIZE_BYTES, cp_async_bulk_shared_to_global>);
    cudaFuncGetAttributes(&unused, &memcpyKernelTma<TMA_CHUNK_SIZE_BYTES, cp_async_bulk_shared_to_global_multicast>);
    cudaFuncGetAttributes(&unused, &memcpyKernelTma<TMA_CHUNK_SIZE_BYTES, cp_async_bulk_shared_to_global_multicast_red>);
}

// This utility function has to be defined in a .cu file because it uses nvcc compiler macros
std::string getNvccVersion() {
    return std::to_string(__CUDACC_VER_MAJOR__) + "." + std::to_string(__CUDACC_VER_MINOR__);
}

bool tmaMulticastSupported() {
    return TMA_MULTICAST_SUPPORTED;
}
