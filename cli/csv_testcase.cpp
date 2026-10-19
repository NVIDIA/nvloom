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

#include "nvloom.h"
#include "csv_testcase.h"
#include "util.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
void csv_pattern<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>::run(size_t copySize) {
    CopyDescriptionMap copyDescriptionMap = parseCSV(csvInputFile);

    for (auto& benchmarkStep : copyDescriptionMap) {
        clearAllocationPools();

        auto& copyDescriptionVector = benchmarkStep.second;

        // All copies in a measurement run simultaneously, so waive the whole measurement
        // if any of its copy or memory types is unsupported on any process
        int supported = 1;
        for (const auto& copyDescription: copyDescriptionVector) {
            if (!filterCopy<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>(copyDescription)) {
                supported = 0;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &supported, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (!supported) {
            for (auto& copyDescription: copyDescriptionVector) {
                copyDescription.waived = true;
            }
            continue;
        }

        std::vector<Copy> copies;

        for (const auto& copyDescription: copyDescriptionVector) {
            copies.push_back(getCopy<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>(copyDescription));
        }

        auto benchmarkResults = NvLoom::doBenchmark(copies);

        for (size_t i = 0; i < copyDescriptionVector.size(); i++) {
            copyDescriptionVector[i].result = benchmarkResults[i];
        }
    }

    outputResults(copyDescriptionMap);
}

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
void csv_pattern<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>::outputResults(const CopyDescriptionMap& copyDescriptionMap) {
    std::string csvOutputString = getCSVOutputString(copyDescriptionMap);

    OUTPUT << csvOutputString << std::endl;

    if (!csvOutputFile.empty()) {
        writeCsvToFile(csvOutputFile, csvOutputString);
    }
}

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
std::string csv_pattern<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>::getName() {
    return "csv_pattern";
}

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
bool csv_pattern<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>::filter() {
    return !csvInputFile.empty();
}

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
void csv_pattern<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>::validateGpuId(int gpuId) {
    if (gpuId < 0) {
        throw std::runtime_error("GPU ID can't be negative: " + std::to_string(gpuId));
    }
    if (gpuId >= MPIWrapper::getWorldSize()) {
        throw std::runtime_error("GPU ID has to be less than the number of processes: " + std::to_string(gpuId) + " >= " + std::to_string(MPIWrapper::getWorldSize()));
    }
}

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
std::string csv_pattern<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>::getWord(std::istringstream& ss) {
    std::string word;
    std::getline(ss, word, ',');
    word = trimWhitespace(word);
    word = toLower(word);
    return word;
}

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
int csv_pattern<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>::getInt(std::string str, std::string field) {
    try {
        return std::stoi(str);
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error("Invalid " + field + ": " + str);
    }
}

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
std::pair<int, CopyDescription> csv_pattern<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>::parseCSVLine(const std::string& line) {
    std::istringstream ss(line);

    // Parse line as "measurement_id, bandwidth/latency, gpu_src_id, gpu_dst_id, iteration_count, mem_src_type, mem_dst_type, write/read, ce/sm/multicast, buffer_size, comment, result [GB/s] [ns]"
    std::string measurementIdStr = getWord(ss);
    std::string bandwidthLatencyStr = getWord(ss);
    std::string gpuSrcStr = getWord(ss);
    std::string gpuDstStr = getWord(ss);
    std::string iterationCountStr = getWord(ss);
    std::string memSrcStr = getWord(ss);
    std::string memDstStr = getWord(ss);
    std::string writeReadStr = getWord(ss);
    std::string ceSmMulticastStr = getWord(ss);
    std::string bufferszStr = getWord(ss);
    std::string commentStr = getWord(ss);

    int measurementId;
    int gpuSrc;
    int gpuDst;
    size_t buffersz;
    size_t iterationCount;

    measurementId = getInt(measurementIdStr, "measurement ID");
    gpuSrc = getInt(gpuSrcStr, "GPU source ID");
    gpuDst = getInt(gpuDstStr, "GPU destination ID");

    CopyType type;
    if (bandwidthLatencyStr == "latency") {
        type = COPY_TYPE_LATENCY;
    } else if (bandwidthLatencyStr == "bandwidth") {
        type = getCopyType(ceSmMulticastStr);
    } else {
        throw std::runtime_error("Invalid bandwidth/latency type: " + bandwidthLatencyStr);
    }

    if (type == COPY_TYPE_MULTICAST_RED_ALL ||
        type == COPY_TYPE_MULTICAST_RED_SINGLE ||
        type == COPY_TYPE_MULTICAST_LD_REDUCE ||
        type == COPY_TYPE_TMA_MULTICAST_RED_ALL ||
        type == COPY_TYPE_TMA_MULTICAST_RED_SINGLE ||
        type == COPY_TYPE_CFT_MULTICAST_RED_SINGLE ||
        type == COPY_TYPE_CFT_MULTICAST_RED_ALL ||
        type == COPY_TYPE_CFT_MULTICAST_PULLRED) {
        throw std::runtime_error("Multicast reductions are not supported for CSV input testcase");
    }

    CopyDirection direction;
    if (type == COPY_TYPE_LATENCY) {
        // ignoring direction for latency testcase
        direction = COPY_DIRECTION_WRITE;
    } else {
        direction = getCopyDirection(writeReadStr);
    }

    if (type == COPY_TYPE_CFT_READ || type == COPY_TYPE_CFT_WRITE) {
        // CFT copy types have different methods for read and write, so we need to override the type based on the direction
        if (direction == COPY_DIRECTION_WRITE) {
            type = COPY_TYPE_CFT_WRITE;
        } else if (direction == COPY_DIRECTION_READ) {
            type = COPY_TYPE_CFT_READ;
        } else {
            throw std::runtime_error("Invalid direction for CFT copy type: " + writeReadStr);
        }

        if (gpuSrc == gpuDst) {
            throw std::runtime_error("CFT copy types are not supported for self-copy");
        }

        // Override the memory type to cft for CFT copy types
        // CFT memory reports its name as "device", so we need to override it to "cft"
        if (memSrcStr == "device") {
            memSrcStr = "cft";
        }
        if (memDstStr == "device") {
            memDstStr = "cft";
        }
    }

    if (type == COPY_TYPE_CFT_WRITE_MULTICAST) {
        if (direction != COPY_DIRECTION_WRITE) {
            throw std::runtime_error("cft_mc copy type requires write direction");
        }

        if (memDstStr != "multicast" && memDstStr != "cft_multicast") {
            throw std::runtime_error("cft_mc copy type requires multicast destination memory");
        }

        // CFT multicast benchmarking requires CFT multicast allocations
        if (memDstStr == "multicast") {
            memDstStr = "cft_multicast";
        }
    }

    if (type == COPY_TYPE_LOCALIZED_CE) {
        // Localized CE reports its name as "device", so we need to override it to "localized"
        if (memSrcStr == "device") {
            memSrcStr = "localized";
        }
        if (memDstStr == "device") {
            memDstStr = "localized";
        }
    }

    if (bufferszStr == "default") {
        buffersz = 1024 * 1024 * NvLoom::getDefaultBufferSizeInMiB(); // default is in MiB
    } else if (bufferszStr == "ignored") {
        if (type != COPY_TYPE_LATENCY) {
            throw std::runtime_error("Buffer size cannot be ignored for bandwidth benchmark");
        }
        buffersz = 0;
    } else {
        if (bufferszStr.empty()) {
            throw std::runtime_error("Buffer size field must be populated in CSV row");
        }
        if (std::isdigit(bufferszStr.back())) {
            buffersz = getSizeT(bufferszStr, "buffer size");
        } else {
            buffersz = getSizeT(bufferszStr.substr(0, bufferszStr.size() - 1), "buffer size");

            if (bufferszStr.back() == 'k') {
                buffersz *= 1024;
            } else if (bufferszStr.back() == 'm') {
                buffersz *= 1024 * 1024;
            } else if (bufferszStr.back() == 'g') {
                buffersz *= 1024 * 1024 * 1024;
            } else {
                throw std::runtime_error("Invalid buffer size: " + bufferszStr);
            }
        }
    }

    if (iterationCountStr == "default") {
        if (type == COPY_TYPE_LATENCY) {
            iterationCount = NvLoom::getDefaultLatencyIterationCount();
        } else {
            iterationCount = NvLoom::getDefaultIterationCount();
        }
    } else {
        iterationCount = getSizeT(iterationCountStr, "iteration count");
    }

    validateGpuId(gpuSrc);
    validateGpuId(gpuDst);

    AllocationDescription src(memSrcStr, gpuSrc, buffersz);
    AllocationDescription dst(memDstStr, gpuDst, buffersz);

    return std::make_pair(measurementId, CopyDescription(src, dst, iterationCount, direction, type, commentStr));
}

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
CopyDescriptionMap csv_pattern<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>::parseCSV(const std::string& filePath) {
    std::string line;
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + filePath);
    }

    // Skip the header line
    if (!std::getline(file, line)) {
        throw std::runtime_error("CSV file is empty or missing header: " + filePath);
    }

    CopyDescriptionMap copyDescriptionMap;

    while (std::getline(file, line)) {
        if (trimWhitespace(line).empty()) {
            continue;
        }
        auto [measurementId, copy] = parseCSVLine(line);
        copyDescriptionMap[measurementId].emplace_back(copy);
    }

    return copyDescriptionMap;
}

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
std::string csv_pattern<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>::getCSVOutputString(const CopyDescriptionMap& copyDescriptionMap) {
    std::stringstream ss;
    ss << csvHeader << std::endl;

    for (const auto& [step, stepResults] : copyDescriptionMap) {
        for (int i = 0; i < stepResults.size(); i++) {
            bool isLatency = stepResults[i].type == COPY_TYPE_LATENCY;

            ss << step << ", ";
            ss << (isLatency ? "latency" : "bandwidth") << ", ";
            ss << stepResults[i].src.location << ", ";
            ss << stepResults[i].dst.location << ", ";
            ss << stepResults[i].iterationCount << ", ";
            ss << stepResults[i].src.type << ", ";
            if (isLatency) {
                ss << "ignored, ignored, ignored, ignored, ";
            } else {
                ss << stepResults[i].dst.type << ", ";
                ss << getCopyDirectionName(stepResults[i].direction) << ", ";
                ss << getCopyTypeName(stepResults[i].type) << ", ";
                ss << stepResults[i].src.bufferSize << ", ";
            }
            ss << stepResults[i].comment << ", ";
            if (stepResults[i].waived) {
                ss << "WAIVED";
            } else {
                ss << stepResults[i].result;
            }
            ss << std::endl;
        }
    }

    return ss.str();
}

// write CSV output file -> can be the same as input
template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename cftAllocator, typename cftMulticastAllocator, typename localizedAllocator>
void csv_pattern<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator>::writeCsvToFile(const std::string& csvOutputFile, const std::string& csvOutput) {
    if (MPIWrapper::getWorldRank() != 0) {
        return;
    }

    std::ofstream csv_o_file;
    csv_o_file.open(csvOutputFile);
    if (!csv_o_file.is_open()) {
        throw std::runtime_error("Failed to open output CSV file: " + csvOutputFile);
    }
    csv_o_file << csvOutput;
}

template class csv_pattern<MultinodeMemoryAllocationUnicast, MultinodeMemoryAllocationEGM, MultinodeMemoryAllocationMulticast, MultinodeMemoryAllocationUnicastCFT, MultinodeMemoryAllocationMulticastCFT, LocalizedMultinodeMemoryAllocationUnicast>;
template class csv_pattern<AllocationPool<MultinodeMemoryAllocationUnicast>, AllocationPool<MultinodeMemoryAllocationEGM>, MulticastPool<MultinodeMemoryAllocationMulticast, MultinodeMemoryAllocationMulticastRef>, AllocationPool<MultinodeMemoryAllocationUnicastCFT>, MulticastPool<MultinodeMemoryAllocationMulticastCFT, MultinodeMemoryAllocationMulticastCFTRef>, AllocationPool<LocalizedMultinodeMemoryAllocationUnicast>>;
template class csv_pattern<MultinodeMemoryPoolAllocationUnicast, MultinodeMemoryPoolAllocationEGM, MultinodeMemoryAllocationMulticastCudaPool, MultinodeMemoryAllocationUnicastCFT, MultinodeMemoryAllocationMulticastCFT, LocalizedMultinodeMemoryAllocationUnicast>;
