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

#ifndef ERROR_H_
#define ERROR_H_

#include <cuda.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "nvloom.h"

#define BASIC_CU_ASSERT(x) do { \
    CUresult cuResult = (x); \
    if ((cuResult) != CUDA_SUCCESS) { \
        const char *errDescStr, *errNameStr; \
        cuGetErrorString(cuResult, &errDescStr); \
        cuGetErrorName(cuResult, &errNameStr); \
        std::cerr << "[" << errNameStr << "] " << errDescStr << " in expression " << #x << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__ << std::endl; \
        std::exit(1); \
    }  \
} while(0)

static std::vector<std::string> getCudaLogs() {
    std::vector<std::string> messages;
#if CUDA_VERSION >= 12090
    size_t messageSize = 25601;
    std::string messagesRaw;
    messagesRaw.resize(messageSize);

    CUresult (*cuLogsDumpToMemory_ptr)(CUlogIterator*, char*, size_t*, unsigned int);
    CUdriverProcAddressQueryResult result;
    BASIC_CU_ASSERT(cuGetProcAddress("cuLogsDumpToMemory", (void **) &cuLogsDumpToMemory_ptr, 12090, CU_GET_PROC_ADDRESS_DEFAULT, &result));

    if (result == CU_GET_PROC_ADDRESS_SUCCESS) {
        BASIC_CU_ASSERT((*cuLogsDumpToMemory_ptr)(NULL, messagesRaw.data(), &messageSize, 0));
        messagesRaw.resize(messageSize);
        std::istringstream messagesStringstream(messagesRaw);
        std::string message;

        while(std::getline(messagesStringstream, message, '\n')) {
            messages.push_back(message);
        }
    }
#endif
    return messages;
}

#undef BASIC_CU_ASSERT

#define ASSERT(x) do { \
    if (!(x)) { \
        std::cerr << "[" << NvLoom::getLocalHostname() << ":" << NvLoom::getLocalDevice() << "]: " << "ASSERT in expression " << #x << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__  << std::endl; \
        std::exit(1); \
    }  \
} while (0)

#define CU_ASSERT(x) do { \
    CUresult cuResult = (x); \
    if ((cuResult) != CUDA_SUCCESS) { \
        const char *errDescStr, *errNameStr; \
        cuGetErrorString(cuResult, &errDescStr); \
        cuGetErrorName(cuResult, &errNameStr); \
        for (auto message : getCudaLogs()) {std::cerr << "[" << NvLoom::getLocalHostname() << ":" << NvLoom::getLocalDevice() << "]: " << message << std::endl;}; \
        std::cerr << "[" << NvLoom::getLocalHostname() << ":" << NvLoom::getLocalDevice() << "]: " << "[" << errNameStr << "] " << errDescStr << " in expression " << #x << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__ << std::endl; \
        std::exit(1); \
    }  \
} while(0)

#define CUDA_ASSERT(x) do { \
    cudaError_t cudaErr = (x); \
    if ((cudaErr) != cudaSuccess) { \
        for (auto message : getCudaLogs()) {std::cerr << "[" << NvLoom::getLocalHostname() << ":" << NvLoom::getLocalDevice() << "]: " << message << std::endl;}; \
        std::cerr << "[" << NvLoom::getLocalHostname() << ":" << NvLoom::getLocalDevice() << "]: " << "[" << cudaGetErrorName(cudaErr) << "] " << cudaGetErrorString(cudaErr) << " in expression " << #x << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__ << std::endl; \
        std::exit(1); \
    }  \
} while (0)

#define NVML_ASSERT(x) do { \
    nvmlReturn_t nvmlResult = (x); \
    if ((nvmlResult) != NVML_SUCCESS) { \
        std::cerr << "[" << NvLoom::getLocalHostname() << ":" << NvLoom::getLocalDevice() << "]: " << "[" << nvmlErrorString(nvmlResult) << "] in expression " << #x << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__ << std::endl; \
        std::exit(1); \
    }  \
} while (0)


#endif  // ERROR_H_
