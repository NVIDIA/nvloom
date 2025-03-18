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

#ifndef ERROR_H_
#define ERROR_H_

#include <cuda.h>
#include <iostream>
#include <sstream>
#include "nvloom.h"

#define ASSERT(x) do { \
    if (!(x)) { \
        std::cerr << "[" << NvLoom::getLocalHostname() << ":" << NvLoom::getLocalDevice() << "]: " << "ASSERT in expression " << #x << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__  << std::endl; \
        std::exit(1); \
    }  \
} while ( 0 )

#define CU_ASSERT(x) do { \
    CUresult cuResult = (x); \
    if ((cuResult) != CUDA_SUCCESS) { \
        const char *errDescStr, *errNameStr; \
        cuGetErrorString(cuResult, &errDescStr); \
        cuGetErrorName(cuResult, &errNameStr); \
        std::cerr << "[" << NvLoom::getLocalHostname() << ":" << NvLoom::getLocalDevice() << "]: " << "[" << errNameStr << "] " << errDescStr << " in expression " << #x << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__ << std::endl; \
        std::exit(1); \
    }  \
} while(0)

#define CUDA_ASSERT(x) do { \
    cudaError_t cudaErr = (x); \
    if ((cudaErr) != cudaSuccess) { \
        std::cerr << "[" << NvLoom::getLocalHostname() << ":" << NvLoom::getLocalDevice() << "]: " << "[" << cudaGetErrorName(cudaErr) << "] " << cudaGetErrorString(cudaErr) << " in expression " << #x << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__ << std::endl; \
        std::exit(1); \
    }  \
} while ( 0 )

#define NVML_ASSERT(x) do { \
    nvmlReturn_t nvmlResult = (x); \
    if ((nvmlResult) != NVML_SUCCESS) { \
        std::cerr << "[" << NvLoom::getLocalHostname() << ":" << NvLoom::getLocalDevice() << "]: " << "[" << nvmlErrorString(nvmlResult) << "] in expression " << #x << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__ << std::endl; \
        std::exit(1); \
    }  \
} while ( 0 )


#endif  // ERROR_H_
