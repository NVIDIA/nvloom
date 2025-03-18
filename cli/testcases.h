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

#ifndef TESTCASES_H_
#define TESTCASES_H_

#include <memory>
#include <map>

extern bool richOutput;

enum AllocatorStrategy {
    ALLOCATOR_STRATEGY_UNIQUE = 0,
    ALLOCATOR_STRATEGY_REUSE,
};

class Testcase {
public:
    virtual void run(size_t copySize) = 0;
    virtual bool filter() = 0;

    virtual void filterRun(size_t copySize) {
        if (filter()) {
            run(copySize);
        } else {
            OUTPUT << "testcase WAIVED" << std::endl;
        }
    }

    virtual std::string getName() {
        return "not implemented";
    }
};

// Most testcases have a very simple filter, based only on source and destination allocators
template <typename dstAllocator, typename srcAllocator>
class TestcaseDstSrc : public Testcase {
public:
    bool filter() {
        return MPIWrapper::getWorldSize() > 1 && srcAllocator::filter() && dstAllocator::filter();
    }
};

std::tuple<std::map<std::string, std::unique_ptr<Testcase> >, std::map<std::string, std::vector<std::string> > > buildTestcases(AllocatorStrategy strategy);

#endif  // UTIL_H_
