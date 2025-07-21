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
#ifndef ENUMS_H_
#define ENUMS_H_

enum CopyDirection {
    COPY_DIRECTION_WRITE = 0,
    COPY_DIRECTION_READ,
};

enum CopyType {
    COPY_TYPE_CE = 0,
    COPY_TYPE_SM,
    COPY_TYPE_MULTICAST_WRITE,
    COPY_TYPE_MULTICAST_LD_REDUCE,
    // Single GPU executing multimem.red instructions (multicast_one_to_all)
    // Those two types execute the same kernels, but they require different verification algorithm
    COPY_TYPE_MULTICAST_RED_SINGLE,
    // All GPUs executing multimem.red instructions (multicast_all_to_all)
    COPY_TYPE_MULTICAST_RED_ALL,
};

enum MemoryPurpose {
    MEMORY_SOURCE = 0,
    MEMORY_DESTINATION,
};

#endif  // ENUMS_H_
