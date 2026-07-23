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

#ifndef KERNELS_CUH_
#define KERNELS_CUH_

#include <cuda.h>
#include <string>
#include <vector>
#include "enums.h"

inline constexpr size_t MAX_SMEM_CAPACITY_SIZE = 227ULL * 1024ULL;

const unsigned long long DEFAULT_SPIN_KERNEL_TIMEOUT_MS = 10000ULL;   // 10 seconds

void copyKernel(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount);
void copyKernelTma(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount);
void copyKernelMulticast(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount);
void copyKernelMulticastTma(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount);

void pointerChase(CUdeviceptr ptr, size_t size, unsigned long long iterations, size_t previouslyExecutedIterations, int targetBlock, CUstream stream);
void copyKernelMulticastLdReduce(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount);
void copyKernelMulticastRed(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount);
void copyKernelMulticastRedTma(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount);
#if CUDA_VERSION >= 13030
void copyKernelCFTRead(CUdeviceptr dstBuffer, CUlogicalEndpointId srcEndpointId, size_t srcEndpointOffset, size_t size, CUstream stream, unsigned long long loopCount);
void copyKernelCFTWrite(CUlogicalEndpointId dstEndpointId, size_t dstEndpointOffset, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount);
void copyKernelCFTWriteMulticast(CUlogicalEndpointId dstEndpointId, size_t dstEndpointOffset, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount);
void copyKernelCFTReduceMulticast(CUlogicalEndpointId dstEndpointId, size_t dstEndpointOffset, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount);
void copyKernelCFTPullredMulticast(CUdeviceptr dstBuffer, CUlogicalEndpointId srcEndpointId, size_t srcEndpointOffset, size_t size, CUstream stream, unsigned long long loopCount);

#endif

bool tmaMulticastSupported();
bool cftSupported();

CUresult spinKernelMultistage(volatile int *latch1, volatile int *latch2, CUstream stream, unsigned long long timeoutMs = DEFAULT_SPIN_KERNEL_TIMEOUT_MS);

unsigned long long checkBuffer(void *ptr, int value, size_t size, CUstream stream, CopyType copyType, int iterations);

void memsetBuffer(void *ptr, int value, size_t size, CUstream stream, CopyType copyType, MemoryPurpose memoryPurpose);
void memsetBufferMulticast(void *ptr, int value, size_t size, CUstream stream, CopyType copyType);
void pointerChaseFill(void *ptr, size_t size, CUstream stream);

void preloadKernels(int localDevice);

std::string getNvccVersion();

std::vector<int> getSmIds();

#endif  // KERNELS_CUH_
