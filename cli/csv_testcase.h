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

#ifndef CSV_TESTCASE_H
#define CSV_TESTCASE_H

#include "testcases.h"

class AllocationDescription {
public:
    std::string type;
    int location;
    size_t bufferSize;

    AllocationDescription(std::string type, int location, size_t bufferSize) : type(type), location(location), bufferSize(bufferSize) {}
};

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
std::shared_ptr<MemoryAllocation> getAllocation(AllocationDescription desc) {
    if (desc.type == "device") {
        return std::make_shared<unicastAllocator>(desc.bufferSize, desc.location);
    } else if (desc.type == "egm") {
        return std::make_shared<egmAllocator>(desc.bufferSize, desc.location);
    } else if (desc.type == "multicast") {
        return std::make_shared<multicastAllocator>(desc.bufferSize, desc.location);
    } else if (desc.type == "cft") {
        return std::make_shared<cftAllocator>(desc.bufferSize, desc.location);
    } else if (desc.type == "cft_multicast") {
        return std::make_shared<cftMulticastAllocator>(desc.bufferSize, desc.location);
    } else if (desc.type == "localized") {
        return std::make_shared<localizedAllocator>(desc.bufferSize, desc.location);
    }
    throw std::runtime_error("Invalid allocation type: " + desc.type);
}

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
bool filterAllocation(const AllocationDescription& desc) {
    if (desc.type == "device") {
        return unicastAllocator::filter();
    } else if (desc.type == "egm") {
        return egmAllocator::filter();
    } else if (desc.type == "multicast") {
        return multicastAllocator::filter();
    } else if (desc.type == "cft") {
        return cftAllocator::filter();
    } else if (desc.type == "cft_multicast") {
        return cftMulticastAllocator::filter();
    } else if (desc.type == "localized") {
        return localizedAllocator::filter();
    }
    throw std::runtime_error("Invalid allocation type: " + desc.type);
}

class CopyDescription {
public:
    AllocationDescription src;
    AllocationDescription dst;
    CopyDirection direction;
    CopyType type;
    double result;
    bool waived = false;
    std::string comment;
    size_t iterationCount;

    CopyDescription(AllocationDescription src, AllocationDescription dst, size_t iterationCount, CopyDirection direction, CopyType type, std::string comment) : src(src), dst(dst), direction(direction), type(type), iterationCount(iterationCount), comment(comment) {}
};

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
bool filterCopy(const CopyDescription& desc) {
    if (!filterCopyType(desc.type)) {
        return false;
    }
    if (!filterAllocation<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>(desc.src)) {
        return false;
    }
    // Latency only uses the source memory type
    if (desc.type == COPY_TYPE_LATENCY) {
        return true;
    }
    return filterAllocation<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>(desc.dst);
}

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
Copy getLatency(CopyDescription desc) {
    if (desc.src.type == "device") {
        return Latency<unicastAllocator>(desc.src.location, desc.dst.location, desc.iterationCount);
    } else if (desc.src.type == "egm") {
        return Latency<egmAllocator>(desc.src.location, desc.dst.location, desc.iterationCount);
    } else {
        throw std::runtime_error("Invalid allocation type for latency: " + desc.src.type);
    }
}

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
Copy getCopy(CopyDescription desc) {
    if (desc.type == COPY_TYPE_LATENCY) {
        return getLatency<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>(desc);
    } else {
        std::shared_ptr<MemoryAllocation> srcAlloc = getAllocation<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>(desc.src);
        std::shared_ptr<MemoryAllocation> dstAlloc = getAllocation<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>(desc.dst);
        return Copy(dstAlloc, srcAlloc, desc.direction, desc.type, desc.iterationCount);
    }
}

typedef std::map<int, std::vector<CopyDescription>> CopyDescriptionMap;

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
class csv_pattern : public Testcase {
private:
    std::string csvHeader = "measurement_id, bandwidth/latency, gpu_src_id, gpu_dst_id, iteration_count, mem_src_type, mem_dst_type, write/read, ce/sm/multicast, buffer_size, comment, result [GB/s] [ns]";

    void validateGpuId(int gpuId);
    std::string getWord(std::istringstream& ss);
    int getInt(std::string str, std::string field);
    std::pair<int, CopyDescription> parseCSVLine(const std::string& line);
    CopyDescriptionMap parseCSV(const std::string& filePath);
    std::string getCSVOutputString(const CopyDescriptionMap& copyDescriptionMap);
    void writeCsvToFile(const std::string& csvOutputFile, const std::string& csvOutputString);
    void outputResults(const CopyDescriptionMap& copyDescriptionMap);
public:
    void run(size_t copySize) override;
    std::string getName() override;
    bool filter() override;
};

#endif // CSV_TESTCASE_H
