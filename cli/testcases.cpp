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
#include "util.h"
#include "testcases.h"
#include "csv_testcase.h"

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


std::mt19937 rng;

std::string getCopyCountName(CopyCount copyCount) {
    if (copyCount == COPY_COUNT_UNIDIR) return "unidir";
    if (copyCount == COPY_COUNT_BIDIR) return "bidir";
    throw std::runtime_error("Invalid copy count");
}

CopyCount getCopyCount(std::string name) {
    if (name == "unidir") return COPY_COUNT_UNIDIR;
    if (name == "bidir") return COPY_COUNT_BIDIR;
    throw std::runtime_error("Invalid copy count");
}

int randomSkipNumber(int randomNumber, int numberToSkip) {
    if (randomNumber >= numberToSkip) return randomNumber + 1;
    else return randomNumber;
}

std::vector<int> randomPermutation(int size) {
    std::vector<int> ordering(size);
    std::iota(ordering.begin(), ordering.end(), 0);
    std::shuffle(ordering.begin(), ordering.end(), rng);
    return ordering;
}

template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType>
double doUnidir(int i, int j, size_t copySize) {
    std::shared_ptr<srcAllocator> src = std::make_shared<srcAllocator>(copySize, i);
    std::shared_ptr<dstAllocator> dst = std::make_shared<dstAllocator>(copySize, j);
    Copy copy(dst, src, copyDirection, copyType, iterations);

    auto results = NvLoom::doBenchmark({copy});
    return results[0];
}

template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType>
double doBidir(int i, int j, size_t copySize) {
    std::shared_ptr<srcAllocator> src1 = std::make_shared<srcAllocator>(copySize, i);
    std::shared_ptr<dstAllocator> dst1 = std::make_shared<dstAllocator>(copySize, j);
    Copy copy1(dst1, src1, copyDirection, copyType, iterations);

    std::shared_ptr<srcAllocator> src2 = std::make_shared<srcAllocator>(copySize, j);
    std::shared_ptr<dstAllocator> dst2 = std::make_shared<dstAllocator>(copySize, i);
    Copy copy2(dst2, src2, copyDirection, copyType, iterations);

    auto results = NvLoom::doBenchmark({copy1, copy2});
    return results[0] + results[1];
}

template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType, CopyCount copyCount>
double doUnidirBidirHelper(int i, int j, size_t copySize) {
    if (copyCount == COPY_COUNT_UNIDIR) {
        return doUnidir<dstAllocator, srcAllocator, copyDirection, copyType>(i, j, copySize);
    } else {
        return doBidir<dstAllocator, srcAllocator, copyDirection, copyType>(i, j, copySize);
    }
}

template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType, CopyCount copyCount>
class N_squared_pattern : public TestcaseDstSrc<dstAllocator, srcAllocator, copyType> {
public:
    void run(size_t copySize) {
        OutputMatrix output(getName(), MPIWrapper::getWorldSize(), MPIWrapper::getWorldSize());
        for(int j = 0; j < MPIWrapper::getWorldSize(); j++) {
            for(int i = 0; i < MPIWrapper::getWorldSize(); i++) {
                if (i == j) {
                    output.set(i, j, 0);
                    continue;
                }

                double bandwidth = doUnidirBidirHelper<dstAllocator, srcAllocator, copyDirection, copyType, copyCount>(i, j, copySize);
                output.set(i, j, bandwidth);
            }
        }
    }

    std::string getName() {
        return srcAllocator::getName() + "_to_" + dstAllocator::getName() + "_" + getCopyCountName(copyCount) +
        "_" + getCopyDirectionName(copyDirection) + "_" + getCopyTypeName(copyType);
    }
};

template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType>
class N_squared_pattern_bidir : public TestcaseDstSrc<dstAllocator, srcAllocator, copyType> {
public:
    void run(size_t copySize) {
        OutputMatrix output(getName(), MPIWrapper::getWorldSize(), MPIWrapper::getWorldSize());
        for(int j = 0; j < MPIWrapper::getWorldSize(); j++) {
            for(int i = j; i < MPIWrapper::getWorldSize(); i++) {
                if (i == j) {
                    output.set(i, j, 0);
                    continue;
                }

                double bandwidth = doBidir<dstAllocator, srcAllocator, copyDirection, copyType>(i, j, copySize);
                output.set(i, j, bandwidth);
                output.set(j, i, bandwidth);
            }
        }
    }

    std::string getName() {
        return srcAllocator::getName() + "_to_" + dstAllocator::getName() + "_" + getCopyCountName(COPY_COUNT_BIDIR) +
        "_" + getCopyDirectionName(copyDirection) + "_" + getCopyTypeName(copyType);
    }
};

template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType, IsRandom isRandom>
class bisect_pattern : public TestcaseDstSrc<dstAllocator, srcAllocator, copyType> {
public:
    void run(size_t copySize) {
        std::vector<std::string> rowLabels;
        std::vector<Copy> copies;

        std::vector<int> randomOrdering;
        if (isRandom) {
            randomOrdering = randomPermutation(MPIWrapper::getWorldSize());
        }

        for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
            int peer = (i + MPIWrapper::getWorldSize() / 2) % MPIWrapper::getWorldSize();

            std::shared_ptr<srcAllocator> src = (isRandom) ? std::make_shared<srcAllocator>(copySize, randomOrdering[i]) : std::make_shared<srcAllocator>(copySize, i);
            std::shared_ptr<dstAllocator> dst = (isRandom) ? std::make_shared<dstAllocator>(copySize, randomOrdering[peer]) : std::make_shared<dstAllocator>(copySize, peer);
            copies.push_back(Copy(dst, src, copyDirection, copyType, iterations));

            std::stringstream s;
            if (isRandom) {
                s << randomOrdering[i] << "->" << randomOrdering[peer];
            }
            else {
                s << i << "->" << peer;
            }
            rowLabels.push_back(s.str());
        }

        OutputMatrix output(getName(), 1, MPIWrapper::getWorldSize());
        output.setLabelsY(rowLabels);
        auto results = NvLoom::doBenchmark(copies);

        for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
            output.set(0, i, results[i]);
        }
    }

    std::string getName() {
        return "bisect_" + (isRandom ? std::string("random_") : std::string("")) + srcAllocator::getName() + "_to_" + dstAllocator::getName() + "_" + getCopyDirectionName(copyDirection) + "_" + getCopyTypeName(copyType);
    }
};

template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType>
class fully_random_pattern : public TestcaseDstSrc<dstAllocator, srcAllocator, copyType> {
public:
    void run(size_t copySize) {
        std::vector<std::string> rowLabels;
        std::vector<Copy> copies;

        // Each GPU executes exactly one copy with a uniformly random peer other than itself.
        // A GPU can be the peer of multiple copies.
        std::uniform_int_distribution<int> peerRange(0, MPIWrapper::getWorldSize() - 2);

        for (int gpu = 0; gpu < MPIWrapper::getWorldSize(); gpu++) {
            int peer = peerRange(rng);
            if (peer >= gpu) peer++;

            // gpu is always the executing device: the source for writes,
            // the destination for reads.
            int srcDevice = (copyDirection == COPY_DIRECTION_READ) ? peer : gpu;
            int dstDevice = (copyDirection == COPY_DIRECTION_READ) ? gpu : peer;

            std::shared_ptr<srcAllocator> src = std::make_shared<srcAllocator>(copySize, srcDevice);
            std::shared_ptr<dstAllocator> dst = std::make_shared<dstAllocator>(copySize, dstDevice);
            copies.push_back(Copy(dst, src, copyDirection, copyType, iterations));

            std::stringstream s;
            s << srcDevice << "->" << dstDevice;
            rowLabels.push_back(s.str());
        }

        OutputMatrix output(getName(), 1, copies.size());
        output.setLabelsY(rowLabels);
        auto results = NvLoom::doBenchmark(copies);

        for (int i = 0; i < copies.size(); i++) {
            output.set(0, i, results[i]);
        }
    }

    std::string getName() {
        return "fully_random_" + srcAllocator::getName() + "_to_" + dstAllocator::getName() + "_" + getCopyDirectionName(copyDirection) + "_" + getCopyTypeName(copyType);
    }
};

// Random Permutation - Ring
template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType>
class random_ring : public TestcaseDstSrc<dstAllocator, srcAllocator, copyType> {
public:
    void run(size_t copySize) {
        std::vector<int> ring = randomPermutation(MPIWrapper::getWorldSize());

        std::vector<std::string> rowLabels;
        std::vector<Copy> copies;

        for (int i = 0; i < ring.size(); i++) {
            int peer = (i + 1) % ring.size();

            std::shared_ptr<srcAllocator> src = std::make_shared<srcAllocator>(copySize, ring[i]);
            std::shared_ptr<dstAllocator> dst = std::make_shared<dstAllocator>(copySize, ring[peer]);
            copies.push_back(Copy(dst, src, copyDirection, copyType, iterations));

            std::stringstream s;
            s << ring[i] << "->" << ring[peer];
            rowLabels.push_back(s.str());
        }

        OutputMatrix output(getName(), 1, MPIWrapper::getWorldSize());
        output.setLabelsY(rowLabels);
        auto results = NvLoom::doBenchmark(copies);

        for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
            output.set(0, i, results[i]);
        }
    }

    std::string getName() {
        return "ring_random_" + srcAllocator::getName() + "_to_" + dstAllocator::getName() + "_" + getCopyDirectionName(copyDirection) + "_" + getCopyTypeName(copyType);
    }
};

template <typename dstAllocator, typename srcAllocator, CopyType copyType>
std::vector<Copy> generateAllToOneCopies(int targetDevice, size_t copySize) {
    std::vector<Copy> copies;

    for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
        if (i == targetDevice) continue;
        std::shared_ptr<dstAllocator> dstAlloc = std::make_shared<dstAllocator>(copySize, targetDevice);
        std::shared_ptr<srcAllocator> srcAlloc = std::make_shared<srcAllocator>(copySize, i);
        copies.push_back(Copy(dstAlloc, srcAlloc, COPY_DIRECTION_WRITE, copyType, iterations));
    }

    return copies;
}

template <typename dstAllocator, typename srcAllocator, CopyType copyType>
std::vector<Copy> generateAllFromOneCopies(int targetDevice, size_t copySize) {
    std::vector<Copy> copies;
    std::shared_ptr<srcAllocator> srcAlloc = std::make_shared<srcAllocator>(copySize, targetDevice);

    for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
        if (i == targetDevice) continue;
        std::shared_ptr<dstAllocator> dstAlloc = std::make_shared<dstAllocator>(copySize, i);
        copies.push_back(Copy(dstAlloc, srcAlloc, COPY_DIRECTION_READ, copyType, iterations));
    }

    return copies;
}

template <typename dstAllocator, typename srcAllocator, CopyType copyType, CopyDirection copyDirection>
void doAllToOneGenericHelper(int targetDevice, size_t copySize, OutputMatrix &output, int rowIndex = -1) {
    std::vector<Copy> copies;
    if (copyDirection == COPY_DIRECTION_WRITE) {
        copies = generateAllToOneCopies<dstAllocator, srcAllocator, copyType>(targetDevice, copySize);
    } else {
        copies = generateAllFromOneCopies<dstAllocator, srcAllocator, copyType>(targetDevice, copySize);
    }
    auto results = NvLoom::doBenchmark(copies);
    if (richOutput) {
        int resultIdx = 0;
        for (int src = 0; src < MPIWrapper::getWorldSize(); src++) {
            if (src == targetDevice) continue;
            // all-to-one-rack-aware's output rows do not use target device IDs
            output.set(src, (rowIndex == -1) ? targetDevice : rowIndex, results[resultIdx++]);
        }
    }
    else {
        output.set(0, (rowIndex == -1) ? targetDevice : rowIndex, std::reduce(results.begin(), results.end()));
    }
}

template <typename dstAllocator, typename srcAllocator, CopyType copyType, CopyDirection copyDirection>
class all_to_one_pattern : public TestcaseDstSrc<dstAllocator, srcAllocator, copyType> {
public:
    void run(size_t copySize) {
        int columns = richOutput ? MPIWrapper::getWorldSize() : 1;
        OutputMatrix output(getName(), columns, MPIWrapper::getWorldSize());

        for (int targetDevice = 0; targetDevice < MPIWrapper::getWorldSize(); targetDevice++) {
            doAllToOneGenericHelper<dstAllocator, srcAllocator, copyType, copyDirection>(targetDevice, copySize, output);
        }
    }

    std::string getName() {
        std::string direction = (copyDirection == COPY_DIRECTION_WRITE) ? "to" : "from";
        return "all_" + direction + "_one_" + srcAllocator::getName() + "_to_" + dstAllocator::getName() + "_" + getCopyDirectionName(copyDirection) + "_" + getCopyTypeName(copyType);
    }
};

template <typename dstAllocator, typename srcAllocator, CopyType copyType, CopyDirection copyDirection>
class all_to_one_rack_aware : public TestcaseDstSrc<dstAllocator, srcAllocator, copyType> {
public:
    void run(size_t copySize) {
        int columns = richOutput ? MPIWrapper::getWorldSize() : 1;
        OutputMatrix output(getName(), columns, NvLoom::getRackToProcessMap().size());

        std::vector<std::string> columnLabels;

        for (auto const& elem : NvLoom::getRackToProcessMap()) {
            columnLabels.push_back(elem.first);
        }

        output.setLabelsY(columnLabels);

        int j = 0;
        for (auto& elem : NvLoom::getRackToProcessMap()) {
            int targetDevice = elem.second[0];
            doAllToOneGenericHelper<dstAllocator, srcAllocator, copyType, copyDirection>(targetDevice, copySize, output, j);
            j++;
        }
    }

    std::string getName() {
        std::string direction = (copyDirection == COPY_DIRECTION_WRITE) ? "to" : "from";
        return "rack_aware_all_" + direction + "_one_" + srcAllocator::getName() + "_to_" + dstAllocator::getName() + "_" + getCopyDirectionName(copyDirection) + "_" + getCopyTypeName(copyType);
    }
};

// Multicast CE copy is only supported with CUDA 13.1+
static bool multicastCEFilter(CopyType copyType) {
    if (copyType != COPY_TYPE_CE) {
        return true;
    }

    int driverVersion = 0;
    CU_ASSERT(cuDriverGetVersion(&driverVersion));
    return driverVersion >= 13010;
}

template <typename dstAllocator, typename srcAllocator, CopyType copyType, CopyDirection copyDirection>
class multicast_one_to_all : public TestcaseDstSrc<dstAllocator, srcAllocator, copyType> {
public:
    void run(size_t copySize) {
        OutputMatrix output(getName(), 1, MPIWrapper::getWorldSize(), BUFFERING_DISABLED);
        for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
            std::shared_ptr<srcAllocator> src = std::make_shared<srcAllocator>(copySize, i);
            std::shared_ptr<dstAllocator> dst = std::make_shared<dstAllocator>(copySize, i);
            Copy copy(dst, src, copyDirection, copyType, iterations);

            auto results = NvLoom::doBenchmark({copy});
            output.set(0, i, results[0]);
        }
    }

    std::string getName() {
        std::string direction;
        if constexpr(copyType == COPY_TYPE_MULTICAST_LD_REDUCE || copyType == COPY_TYPE_CFT_MULTICAST_PULLRED) {
            direction = "all_to_one";
        } else {
            direction = "one_to_all";
        }
        return "multicast_" + direction + "_" + getCopyTypeName(copyType);
    }

    bool filter() {
        return multicastCEFilter(copyType) && (MPIWrapper::getWorldSize() > 1) && srcAllocator::filter() && dstAllocator::filter() && filterCopyType(copyType);
    }
};

static void emitMulticastAllToAllOutput(const std::string &name, const std::vector<double> &results) {
    OutputMatrix output(name, richOutput ? results.size() : 1, 1, BUFFERING_DISABLED);
    if (richOutput) {
        for (int i = 0; i < (int) results.size(); i++) {
            output.set(i, 0, results[i]);
        }
    } else {
        output.set(0, 0, std::reduce(results.begin(), results.end()));
    }
}

template <typename dstAllocator, typename srcAllocator, CopyType copyType>
class multicast_all_to_all : public TestcaseDstSrc<dstAllocator, srcAllocator, copyType> {
public:
    void run(size_t copySize) {
        std::vector<Copy> copies;

        dstAllocator::setCapacityHint(MPIWrapper::getWorldSize());

        for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
            std::shared_ptr<srcAllocator> src = std::make_shared<srcAllocator>(copySize, i);
            std::shared_ptr<dstAllocator> dst = std::make_shared<dstAllocator>(copySize, i);
            copies.push_back(Copy(dst, src, COPY_DIRECTION_WRITE, copyType, iterations));
        }

        auto results = NvLoom::doBenchmark(copies);

        emitMulticastAllToAllOutput(getName(), results);
    }

    std::string getName() {
        return "multicast_all_to_all_" + getCopyTypeName(copyType);
    }

    bool filter() {
        return multicastCEFilter(copyType) && (MPIWrapper::getWorldSize() > 1) && srcAllocator::filter() && dstAllocator::filter() && filterCopyType(copyType);
    }
};

template <typename dstAllocator, typename srcAllocator, CopyType copyType>
class multicast_all_to_all_ld_reduce : public TestcaseDstSrc<dstAllocator, srcAllocator, copyType> {
public:
    void run(size_t copySize) {
        std::vector<Copy> copies;

        std::shared_ptr<srcAllocator> src = std::make_shared<srcAllocator>(copySize, 0);

        for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
            std::shared_ptr<dstAllocator> dst = std::make_shared<dstAllocator>(copySize, i);
            copies.push_back(Copy(dst, src, COPY_DIRECTION_READ, copyType, iterations));
        }

        auto results = NvLoom::doBenchmark(copies);

        emitMulticastAllToAllOutput(getName(), results);
    }

    std::string getName() {
        return "multicast_all_to_all_" + getCopyTypeName(copyType);
    }
};

template <typename dstAllocator, typename srcAllocator, CopyType copyType>
class multicast_all_to_all_red : public TestcaseDstSrc<dstAllocator, srcAllocator, copyType> {
public:
    void run(size_t copySize) {
        std::vector<Copy> copies;
        std::shared_ptr<dstAllocator> dst = std::make_shared<dstAllocator>(copySize, 0);

        for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
            std::shared_ptr<srcAllocator> src = std::make_shared<srcAllocator>(copySize, i);
            Copy copy(dst, src, COPY_DIRECTION_WRITE, copyType, iterations);
            copies.push_back(copy);

            // For the multicast_all_to_all_red testcase, we need to make sure that all source have the same uniqueId
            // Otherwise validation will fail
            // Each buffer is still unique, because the contents depend on both uniqueId and GPU_ID
            src->uniqueId = copies[0].src->uniqueId;
        }

        auto results = NvLoom::doBenchmark(copies);
        emitMulticastAllToAllOutput(getName(), results);
    }

    std::string getName() {
        return "multicast_all_to_all_" + getCopyTypeName(copyType);
    }
};

std::vector<int> pickRandomSamplesWithSkip(std::vector<int> elems, int samples, int numberToSkip, std::mt19937 &rng) {
    // This possibly could be binsearch, as the input vector *should* be sorted
    // But let's be careful for now
    auto elemToSkip = std::find(elems.begin(), elems.end(), numberToSkip);
    if (elemToSkip != elems.end()) {
        std::iter_swap(elemToSkip, elems.end() - 1);
        elems.pop_back();
    }

    std::vector<int> pickedElems;
    for (int i = 0; i < samples; i++) {
        if (elems.size() == 0) {
            break;
        }

        std::uniform_int_distribution<std::mt19937::result_type> dist(0, elems.size() - 1);

        int pickedIndex = dist(rng);
        pickedElems.push_back(elems[pickedIndex]);

        std::iter_swap(elems.begin() + pickedIndex, elems.end() - 1);
        elems.pop_back();
    }

    return pickedElems;
}

double getMedian(std::vector<double> &results) {
    if (results.size() == 0) {
        return 0;
    }

    size_t half = results.size() / 2;
    std::nth_element(results.begin(), results.begin() + half, results.end());
    return results[half];
}

template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType, CopyCount copyCount>
class gpu_to_rack : public TestcaseDstSrc<dstAllocator, srcAllocator, copyType> {
public:
    void run(size_t copySize) {
        ASSERT(NvLoom::getRackToProcessMap().size() > 0);

        int columns = NvLoom::getRackToProcessMap().size();
        if (richOutput) {
            columns *= gpuToRackSamples;
        }
        OutputMatrix output(getName(), columns, MPIWrapper::getWorldSize());

        std::vector<std::string> columnLabels;

        for (auto const& elem : NvLoom::getRackToProcessMap()) {
            if (richOutput) {
                for (int i = 0; i < gpuToRackSamples; i++) {
                    columnLabels.push_back(elem.first + "-" + std::to_string(i));
                }
            } else {
                columnLabels.push_back(elem.first);
            }
        }

        output.setLabelsX(columnLabels);

        if (richOutput) {
            std::vector<int> columnSeparators;
            for (int i = 0; i < NvLoom::getRackToProcessMap().size() - 1; i++) {
                columnSeparators.push_back((i + 1) * gpuToRackSamples - 1);
            }
            output.setColumnSeparators(columnSeparators);
        }

        for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
            int j = 0;
            for (auto& elem : NvLoom::getRackToProcessMap()) {
                std::vector<double> results;

                auto peers = pickRandomSamplesWithSkip(elem.second, gpuToRackSamples, i, rng);
                for (auto peerGPU : peers) {
                    // double check that the copy is executed in correct direction
                    double bandwidth = doUnidirBidirHelper<dstAllocator, srcAllocator, copyDirection, copyType, copyCount>(i, peerGPU, copySize);
                    results.push_back(bandwidth);
                }

                if (richOutput) {
                    for (int x = 0; x < results.size(); x++) {
                        std::stringstream note;
                        note << i << "-" << peers[x];
                        output.set(gpuToRackSamples * j + x, i, results[x], note.str());
                    }
                } else {
                    output.set(j, i, getMedian(results));
                }

                j++;
            }
        }
    }

    std::string getName() {
        return "gpu_to_rack_" + srcAllocator::getName() + "_to_" + dstAllocator::getName() + "_" + getCopyCountName(copyCount) +
        "_" + getCopyDirectionName(copyDirection) + "_" + getCopyTypeName(copyType);
    }
};

template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType>
class rack_to_rack_unidir : public TestcaseDstSrc<dstAllocator, srcAllocator, copyType> {
public:
    void run(size_t copySize) {
        OutputMatrix output(getName(), NvLoom::getRackToProcessMap().size(), NvLoom::getRackToProcessMap().size());

        auto rackKeys = getKeys(NvLoom::getRackToProcessMap());
        output.setLabelsX(rackKeys);
        output.setLabelsY(rackKeys);

        int i = 0;
        for (auto const& rack : NvLoom::getRackToProcessMap()) {
            int j = 0;
            for (auto& peerRack : NvLoom::getRackToProcessMap()) {
                if (rack.first == peerRack.first) {
                    j++;
                    continue;
                }

                int peerCount = std::min(rack.second.size(), peerRack.second.size());

                std::vector<Copy> copies;
                for (int iter = 0; iter < peerCount; iter++) {
                    std::shared_ptr<srcAllocator> src = std::make_shared<srcAllocator>(copySize, rack.second[iter]);
                    std::shared_ptr<dstAllocator> dst = std::make_shared<dstAllocator>(copySize, peerRack.second[iter]);
                    copies.push_back(Copy(dst, src, copyDirection, copyType, iterations));
                }

                auto results = NvLoom::doBenchmark(copies);
                output.set(i, j, std::reduce(results.begin(), results.end()));
                j++;
            }
            i++;
        }
    }

    std::string getName() {
        return "rack_to_rack_unidir_" + srcAllocator::getName() + "_to_" + dstAllocator::getName() +
        "_" + getCopyDirectionName(copyDirection) + "_" + getCopyTypeName(copyType);
    }
};

template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType>
class rack_to_rack_bidir : public TestcaseDstSrc<dstAllocator, srcAllocator, copyType> {
public:
    void run(size_t copySize) {
        OutputMatrix output(getName(), NvLoom::getRackToProcessMap().size(), NvLoom::getRackToProcessMap().size());

        auto rackKeys = getKeys(NvLoom::getRackToProcessMap());
        output.setLabelsX(rackKeys);
        output.setLabelsY(rackKeys);

        for (int i = 0; i < rackKeys.size(); i++) {
            for (int j = i; j < rackKeys.size(); j++) {
                if (i == j) {
                    output.set(i, i, 0);
                    continue;
                }

                auto origRack = NvLoom::getRackToProcessMap()[rackKeys[i]];
                auto peerRack = NvLoom::getRackToProcessMap()[rackKeys[j]];

                int peerCount = std::min(origRack.size(), peerRack.size());

                std::vector<Copy> copies;
                for (int iter = 0; iter < peerCount; iter++) {
                    std::shared_ptr<srcAllocator> src1 = std::make_shared<srcAllocator>(copySize, origRack[iter]);
                    std::shared_ptr<dstAllocator> dst1 = std::make_shared<dstAllocator>(copySize, peerRack[iter]);
                    copies.push_back(Copy(dst1, src1, copyDirection, copyType, iterations));

                    std::shared_ptr<srcAllocator> src2 = std::make_shared<srcAllocator>(copySize, peerRack[iter]);
                    std::shared_ptr<dstAllocator> dst2 = std::make_shared<dstAllocator>(copySize, origRack[iter]);
                    copies.push_back(Copy(dst2, src2, copyDirection, copyType, iterations));
                }

                auto results = NvLoom::doBenchmark(copies);
                auto sum = std::reduce(results.begin(), results.end());
                output.set(i, j, sum);
                output.set(j, i, sum);
            }
        }
    }

    std::string getName() {
        return "rack_to_rack_bidir_" + srcAllocator::getName() + "_to_" + dstAllocator::getName() +
        "_" + getCopyDirectionName(copyDirection) + "_" + getCopyTypeName(copyType);
    }
};

template <typename allocator>
class latency : public Testcase {
public:
    void run(size_t copySize) {
        OutputMatrix output(getName(), MPIWrapper::getWorldSize(), MPIWrapper::getWorldSize(), BUFFERING_ENABLED, OutputMatrix::MeasurementType::LATENCY);

        for (int shift = 1; shift < MPIWrapper::getWorldSize(); shift++) {
            std::vector<Copy> latencies;
            for (int executingRank = 0; executingRank < MPIWrapper::getWorldSize(); executingRank++) {
                int memoryLocation = (executingRank + shift) % MPIWrapper::getWorldSize();
                latencies.push_back(Latency<allocator>(memoryLocation, executingRank, latencyIterations));
            }

            auto results = NvLoom::doBenchmark(latencies);

            for (int executingRank = 0; executingRank < results.size(); executingRank++) {
                int memoryLocation = (executingRank + shift) % MPIWrapper::getWorldSize();
                output.set(memoryLocation, executingRank, results[executingRank]);
            }
        }
    }

    std::string getName() {
        return "latency_" + allocator::getName();
    }

    bool filter() {
        return allocator::filter();
    }
};

static void addTestcase(
        std::map<std::string, std::vector<std::string> > &suites,
        std::string suiteName,
        std::map<std::string, std::unique_ptr<Testcase> > &testcases,
        std::unique_ptr<Testcase> &&testcase) {

    if (suiteName.size() > 0) {
        suites[suiteName].push_back(testcase->getName());
    }

    ASSERT(testcases.count(testcase->getName()) == 0);

    testcases[testcase->getName()] = std::move(testcase);
}

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator, typename localizedAllocator, typename cftAllocator, typename cftMulticastAllocator>
std::tuple<std::map<std::string, std::unique_ptr<Testcase> >, std::map<std::string, std::vector<std::string> > > buildTestcasesLower() {
    std::map<std::string, std::unique_ptr<Testcase> > testcases;
    std::map<std::string, std::vector<std::string> > suites;

    // pairwise
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern<localizedAllocator, localizedAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_LOCALIZED_CE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern<localizedAllocator, localizedAllocator, COPY_DIRECTION_READ,  COPY_TYPE_LOCALIZED_CE, COPY_COUNT_UNIDIR> >());

    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE> >());
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE> >());
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM> >());
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM> >());
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern_bidir<localizedAllocator, localizedAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_LOCALIZED_CE> >());
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern_bidir<localizedAllocator, localizedAllocator, COPY_DIRECTION_READ,  COPY_TYPE_LOCALIZED_CE> >());

    // pairwise TMA
    addTestcase(suites, "pairwise-tma", testcases, std::make_unique<N_squared_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_TMA, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "pairwise-tma", testcases, std::make_unique<N_squared_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_TMA, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "pairwise-tma", testcases, std::make_unique<N_squared_pattern_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_TMA> >());
    addTestcase(suites, "pairwise-tma", testcases, std::make_unique<N_squared_pattern_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_TMA> >());

    // pairwise CFT
    addTestcase(suites, "pairwise-cft", testcases, std::make_unique<N_squared_pattern<cftAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CFT_WRITE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "pairwise-cft", testcases, std::make_unique<N_squared_pattern<unicastAllocator, cftAllocator, COPY_DIRECTION_READ, COPY_TYPE_CFT_READ, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "pairwise-cft", testcases, std::make_unique<N_squared_pattern_bidir<cftAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CFT_WRITE> >());
    addTestcase(suites, "pairwise-cft", testcases, std::make_unique<N_squared_pattern_bidir<unicastAllocator, cftAllocator, COPY_DIRECTION_READ, COPY_TYPE_CFT_READ> >());

    // fabric-stress
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE, NOT_RANDOM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE, NOT_RANDOM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM, NOT_RANDOM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM, NOT_RANDOM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_TMA, NOT_RANDOM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_TMA, NOT_RANDOM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<localizedAllocator, localizedAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_LOCALIZED_CE, NOT_RANDOM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<localizedAllocator, localizedAllocator, COPY_DIRECTION_READ,  COPY_TYPE_LOCALIZED_CE, NOT_RANDOM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE, RANDOM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE, RANDOM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM, RANDOM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM, RANDOM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_TMA, RANDOM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_TMA, RANDOM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<localizedAllocator, localizedAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_LOCALIZED_CE, RANDOM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<localizedAllocator, localizedAllocator, COPY_DIRECTION_READ,  COPY_TYPE_LOCALIZED_CE, RANDOM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<random_ring<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<random_ring<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<random_ring<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<random_ring<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<random_ring<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_TMA> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<random_ring<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_TMA> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<random_ring<localizedAllocator, localizedAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_LOCALIZED_CE> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<random_ring<localizedAllocator, localizedAllocator, COPY_DIRECTION_READ, COPY_TYPE_LOCALIZED_CE> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<fully_random_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<fully_random_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<fully_random_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<fully_random_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<fully_random_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_TMA> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<fully_random_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_TMA> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<fully_random_pattern<localizedAllocator, localizedAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_LOCALIZED_CE> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<fully_random_pattern<localizedAllocator, localizedAllocator, COPY_DIRECTION_READ, COPY_TYPE_LOCALIZED_CE> >());

    // all-to-one
    addTestcase(suites, "all-to-one", testcases, std::make_unique<all_to_one_pattern<unicastAllocator, unicastAllocator, COPY_TYPE_CE, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "all-to-one", testcases, std::make_unique<all_to_one_pattern<unicastAllocator, unicastAllocator, COPY_TYPE_CE, COPY_DIRECTION_READ> >());
    addTestcase(suites, "all-to-one", testcases, std::make_unique<all_to_one_pattern<unicastAllocator, unicastAllocator, COPY_TYPE_SM, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "all-to-one", testcases, std::make_unique<all_to_one_pattern<unicastAllocator, unicastAllocator, COPY_TYPE_SM, COPY_DIRECTION_READ> >());
    addTestcase(suites, "all-to-one", testcases, std::make_unique<all_to_one_pattern<unicastAllocator, unicastAllocator, COPY_TYPE_TMA, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "all-to-one", testcases, std::make_unique<all_to_one_pattern<unicastAllocator, unicastAllocator, COPY_TYPE_TMA, COPY_DIRECTION_READ> >());
    addTestcase(suites, "all-to-one", testcases, std::make_unique<all_to_one_pattern<localizedAllocator, localizedAllocator, COPY_TYPE_LOCALIZED_CE, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "all-to-one", testcases, std::make_unique<all_to_one_pattern<localizedAllocator, localizedAllocator, COPY_TYPE_LOCALIZED_CE, COPY_DIRECTION_READ> >());
    addTestcase(suites, "all-to-one", testcases, std::make_unique<all_to_one_pattern<cftAllocator, unicastAllocator, COPY_TYPE_CFT_WRITE, COPY_DIRECTION_WRITE > >());
    addTestcase(suites, "all-to-one", testcases, std::make_unique<all_to_one_pattern<unicastAllocator, cftAllocator, COPY_TYPE_CFT_READ, COPY_DIRECTION_READ > >());

    // all-to-one rack-aware
    addTestcase(suites, "rack-aware-all-to-one", testcases, std::make_unique<all_to_one_rack_aware<unicastAllocator, unicastAllocator, COPY_TYPE_CE, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "rack-aware-all-to-one", testcases, std::make_unique<all_to_one_rack_aware<unicastAllocator, unicastAllocator, COPY_TYPE_CE, COPY_DIRECTION_READ> >());
    addTestcase(suites, "rack-aware-all-to-one", testcases, std::make_unique<all_to_one_rack_aware<unicastAllocator, unicastAllocator, COPY_TYPE_SM, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "rack-aware-all-to-one", testcases, std::make_unique<all_to_one_rack_aware<unicastAllocator, unicastAllocator, COPY_TYPE_SM, COPY_DIRECTION_READ> >());
    addTestcase(suites, "rack-aware-all-to-one", testcases, std::make_unique<all_to_one_rack_aware<unicastAllocator, unicastAllocator, COPY_TYPE_TMA, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "rack-aware-all-to-one", testcases, std::make_unique<all_to_one_rack_aware<unicastAllocator, unicastAllocator, COPY_TYPE_TMA, COPY_DIRECTION_READ> >());
    addTestcase(suites, "rack-aware-all-to-one", testcases, std::make_unique<all_to_one_rack_aware<localizedAllocator, localizedAllocator, COPY_TYPE_LOCALIZED_CE, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "rack-aware-all-to-one", testcases, std::make_unique<all_to_one_rack_aware<localizedAllocator, localizedAllocator, COPY_TYPE_LOCALIZED_CE, COPY_DIRECTION_READ> >());
    addTestcase(suites, "rack-aware-all-to-one", testcases, std::make_unique<all_to_one_rack_aware<cftAllocator, unicastAllocator, COPY_TYPE_CFT_WRITE, COPY_DIRECTION_WRITE > >());
    addTestcase(suites, "rack-aware-all-to-one", testcases, std::make_unique<all_to_one_rack_aware<unicastAllocator, cftAllocator, COPY_TYPE_CFT_READ, COPY_DIRECTION_READ > >());

    // multicast
    addTestcase(suites, "multicast", testcases, std::make_unique<multicast_one_to_all<multicastAllocator, DeviceMemoryAllocation, COPY_TYPE_MULTICAST_WRITE, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "multicast", testcases, std::make_unique<multicast_all_to_all<multicastAllocator, DeviceMemoryAllocation, COPY_TYPE_MULTICAST_WRITE> >());

    addTestcase(suites, "multicast", testcases, std::make_unique<multicast_one_to_all<multicastAllocator, DeviceMemoryAllocation, COPY_TYPE_CE, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "multicast", testcases, std::make_unique<multicast_all_to_all<multicastAllocator, DeviceMemoryAllocation, COPY_TYPE_CE> >());

    addTestcase(suites, "multicast", testcases, std::make_unique<multicast_one_to_all<DeviceMemoryAllocation, multicastAllocator, COPY_TYPE_MULTICAST_LD_REDUCE, COPY_DIRECTION_READ> >());
    addTestcase(suites, "multicast", testcases, std::make_unique<multicast_all_to_all_ld_reduce<DeviceMemoryAllocation, multicastAllocator, COPY_TYPE_MULTICAST_LD_REDUCE> >());

    addTestcase(suites, "multicast", testcases, std::make_unique<multicast_one_to_all<multicastAllocator, DeviceMemoryAllocation, COPY_TYPE_MULTICAST_RED_SINGLE, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "multicast", testcases, std::make_unique<multicast_all_to_all_red<multicastAllocator, DeviceMemoryAllocation, COPY_TYPE_MULTICAST_RED_ALL> >());

    // multicast TMA
    addTestcase(suites, "multicast-tma", testcases, std::make_unique<multicast_one_to_all<multicastAllocator, DeviceMemoryAllocation, COPY_TYPE_TMA_MULTICAST_WRITE, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "multicast-tma", testcases, std::make_unique<multicast_all_to_all<multicastAllocator, DeviceMemoryAllocation, COPY_TYPE_TMA_MULTICAST_WRITE> >());

    addTestcase(suites, "multicast-tma", testcases, std::make_unique<multicast_one_to_all<multicastAllocator, DeviceMemoryAllocation, COPY_TYPE_TMA_MULTICAST_RED_SINGLE, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "multicast-tma", testcases, std::make_unique<multicast_all_to_all_red<multicastAllocator, DeviceMemoryAllocation, COPY_TYPE_TMA_MULTICAST_RED_ALL> >());

    // multicast CFT
    addTestcase(suites, "multicast-cft", testcases, std::make_unique<multicast_one_to_all<cftMulticastAllocator, DeviceMemoryAllocation, COPY_TYPE_CFT_WRITE_MULTICAST, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "multicast-cft", testcases, std::make_unique<multicast_all_to_all<cftMulticastAllocator, DeviceMemoryAllocation, COPY_TYPE_CFT_WRITE_MULTICAST> >());

    addTestcase(suites, "multicast-cft", testcases, std::make_unique<multicast_one_to_all<cftMulticastAllocator, DeviceMemoryAllocation, COPY_TYPE_CFT_MULTICAST_RED_SINGLE, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "multicast-cft", testcases, std::make_unique<multicast_all_to_all_red<cftMulticastAllocator, DeviceMemoryAllocation, COPY_TYPE_CFT_MULTICAST_RED_ALL> >());

    addTestcase(suites, "multicast-cft", testcases, std::make_unique<multicast_one_to_all<DeviceMemoryAllocation, cftMulticastAllocator, COPY_TYPE_CFT_MULTICAST_PULLRED, COPY_DIRECTION_READ> >());
    addTestcase(suites, "multicast-cft", testcases, std::make_unique<multicast_all_to_all_ld_reduce<DeviceMemoryAllocation, cftMulticastAllocator, COPY_TYPE_CFT_MULTICAST_PULLRED> >());

    // egm
    addTestcase(suites, "egm", testcases, std::make_unique<N_squared_pattern<egmAllocator, egmAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "egm", testcases, std::make_unique<N_squared_pattern<egmAllocator, egmAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "egm", testcases, std::make_unique<N_squared_pattern<egmAllocator, egmAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "egm", testcases, std::make_unique<N_squared_pattern<egmAllocator, egmAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM, COPY_COUNT_UNIDIR> >());

    addTestcase(suites, "egm", testcases, std::make_unique<N_squared_pattern<unicastAllocator, egmAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "egm", testcases, std::make_unique<N_squared_pattern<unicastAllocator, egmAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "egm", testcases, std::make_unique<N_squared_pattern<unicastAllocator, egmAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "egm", testcases, std::make_unique<N_squared_pattern<unicastAllocator, egmAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM, COPY_COUNT_UNIDIR> >());

    // gpu-to-rack randomized sampling
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_TMA, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_TMA, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<cftAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CFT_WRITE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, cftAllocator, COPY_DIRECTION_READ, COPY_TYPE_CFT_READ, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<localizedAllocator, localizedAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_LOCALIZED_CE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<localizedAllocator, localizedAllocator, COPY_DIRECTION_READ, COPY_TYPE_LOCALIZED_CE, COPY_COUNT_UNIDIR> >());


    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE, COPY_COUNT_BIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE, COPY_COUNT_BIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM, COPY_COUNT_BIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM, COPY_COUNT_BIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_TMA, COPY_COUNT_BIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_TMA, COPY_COUNT_BIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<cftAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CFT_WRITE, COPY_COUNT_BIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, cftAllocator, COPY_DIRECTION_READ, COPY_TYPE_CFT_READ, COPY_COUNT_BIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<localizedAllocator, localizedAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_LOCALIZED_CE, COPY_COUNT_BIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<localizedAllocator, localizedAllocator, COPY_DIRECTION_READ, COPY_TYPE_LOCALIZED_CE, COPY_COUNT_BIDIR> >());

    // egm-gpu-to-rack
    addTestcase(suites, "egm-gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, egmAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "egm-gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, egmAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "egm-gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, egmAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "egm-gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, egmAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM, COPY_COUNT_UNIDIR> >());

    addTestcase(suites, "egm-gpu-to-rack", testcases, std::make_unique<gpu_to_rack<egmAllocator, egmAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "egm-gpu-to-rack", testcases, std::make_unique<gpu_to_rack<egmAllocator, egmAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "egm-gpu-to-rack", testcases, std::make_unique<gpu_to_rack<egmAllocator, egmAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "egm-gpu-to-rack", testcases, std::make_unique<gpu_to_rack<egmAllocator, egmAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM, COPY_COUNT_UNIDIR> >());

    // rack-to-rack fabric saturation
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_unidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_unidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_unidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_unidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_unidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_TMA> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_unidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_TMA> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_unidir<localizedAllocator, localizedAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_LOCALIZED_CE> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_unidir<localizedAllocator, localizedAllocator, COPY_DIRECTION_READ, COPY_TYPE_LOCALIZED_CE> >());

    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_TMA> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_TMA> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_bidir<localizedAllocator, localizedAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_LOCALIZED_CE> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_bidir<localizedAllocator, localizedAllocator, COPY_DIRECTION_READ, COPY_TYPE_LOCALIZED_CE> >());

    // csv
    addTestcase(suites, "csv", testcases, std::make_unique<csv_pattern<unicastAllocator, egmAllocator, multicastAllocator, cftAllocator, cftMulticastAllocator, localizedAllocator> >());

    // latency
    addTestcase(suites, "latency", testcases, std::make_unique<latency<unicastAllocator> >());
    addTestcase(suites, "latency", testcases, std::make_unique<latency<egmAllocator> >());

    return {std::move(testcases), std::move(suites)};
}

std::tuple<std::map<std::string, std::unique_ptr<Testcase> >, std::map<std::string, std::vector<std::string> > > buildTestcases(AllocatorStrategy strategy) {
    if (strategy == ALLOCATOR_STRATEGY_UNIQUE) {
        return buildTestcasesLower< MultinodeMemoryAllocationUnicast,
                                    MultinodeMemoryAllocationEGM,
                                    MultinodeMemoryAllocationMulticast,
                                    LocalizedMultinodeMemoryAllocationUnicast,
                                    MultinodeMemoryAllocationUnicastCFT,
                                    MultinodeMemoryAllocationMulticastCFT>();
    } else if (strategy == ALLOCATOR_STRATEGY_REUSE) {
        // like the allocations themselves, the endpoints are reused until the pools are cleared
        UnicastLogicalEndpointPool::setRetainEndpoints(true);

        return buildTestcasesLower< AllocationPool<MultinodeMemoryAllocationUnicast>,
                                    AllocationPool<MultinodeMemoryAllocationEGM>,
                                    MulticastPool<MultinodeMemoryAllocationMulticast, MultinodeMemoryAllocationMulticastRef>,
                                    AllocationPool<LocalizedMultinodeMemoryAllocationUnicast>,
                                    AllocationPool<MultinodeMemoryAllocationUnicastCFT>,
                                    MulticastPool<MultinodeMemoryAllocationMulticastCFT, MultinodeMemoryAllocationMulticastCFTRef>>();
    } else if (strategy == ALLOCATOR_STRATEGY_CUDA_POOLS) {
        return buildTestcasesLower< MultinodeMemoryPoolAllocationUnicast,
                                    MultinodeMemoryPoolAllocationEGM,
                                    MultinodeMemoryAllocationMulticastCudaPool,
                                    LocalizedMultinodeMemoryAllocationUnicast,
                                    MultinodeMemoryAllocationUnicastCFT,
                                    MultinodeMemoryAllocationMulticastCFT>();
    }
    ASSERT(0);
}

std::vector<std::string> getSuiteNames() {
    // The set of suites is identical for every allocator strategy (the suite
    // names are hardcoded in buildTestcasesLower and don't depend on the
    // allocator template parameters), so any strategy enumerates them all.
    auto [testcases, suites] = buildTestcases(ALLOCATOR_STRATEGY_UNIQUE);
    return getKeys(suites);
}
