/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

const unsigned long long DEFAULT_SPIN_KERNEL_TIMEOUT_MS = 10000ULL;   // 10 seconds

void copyKernel(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount);
void copyKernelMulticast(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount);

CUresult spinKernelMultistage(volatile int *latch1, volatile int *latch2, CUstream stream, unsigned long long timeoutMs = DEFAULT_SPIN_KERNEL_TIMEOUT_MS);

unsigned long long checkBuffer(void *ptr, int value, size_t size, CUstream stream);
unsigned long long checkBufferMulticast(void *ptr, int value, size_t size, CUstream stream);

void memsetBuffer(void *ptr, int value, size_t size, CUstream stream);
void memsetBufferMulticast(void *ptr, int value, size_t size, CUstream stream);

void preloadKernels(int localDevice);

#endif  // KERNELS_CUH_
