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

std::string getCopyDirectionName(CopyDirection copyDirection) {
    if (copyDirection == COPY_DIRECTION_READ) return "read";
    if (copyDirection == COPY_DIRECTION_WRITE) return "write";
    throw std::runtime_error("Invalid copy direction");
}

CopyDirection getCopyDirection(std::string name) {
    if (name == "read") return COPY_DIRECTION_READ;
    if (name == "write") return COPY_DIRECTION_WRITE;
    throw std::runtime_error("Invalid copy direction");
}

std::string getCopyTypeName(CopyType copyType) {
    if (copyType == COPY_TYPE_CE) return "ce";
    if (copyType == COPY_TYPE_SM) return "sm";
    if (copyType == COPY_TYPE_MULTICAST_WRITE) return "mc";
    if (copyType == COPY_TYPE_MULTICAST_LD_REDUCE) return "mc_ld_reduce";
    if (copyType == COPY_TYPE_MULTICAST_RED_ALL || copyType == COPY_TYPE_MULTICAST_RED_SINGLE) return "mc_red";
    if (copyType == COPY_TYPE_LATENCY) return "latency";
    throw std::runtime_error("Invalid copy type");
}

CopyType getCopyType(std::string name) {
    if (name == "ce") return COPY_TYPE_CE;
    if (name == "sm") return COPY_TYPE_SM;
    if (name == "mc") return COPY_TYPE_MULTICAST_WRITE;
    if (name == "mc_ld_reduce") return COPY_TYPE_MULTICAST_LD_REDUCE;
    if (name == "mc_red") return COPY_TYPE_MULTICAST_RED_ALL;
    if (name == "latency") return COPY_TYPE_LATENCY;
    throw std::runtime_error("Invalid copy type");
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
class N_squared_pattern : public TestcaseDstSrc<dstAllocator, srcAllocator> {
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
class N_squared_pattern_bidir : public TestcaseDstSrc<dstAllocator, srcAllocator> {
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

template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType>
class bisect_pattern : public TestcaseDstSrc<dstAllocator, srcAllocator> {
public:
    void run(size_t copySize) {
        std::vector<std::string> rowLabels;
        std::vector<Copy> copies;
        for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
            int peer = (i + MPIWrapper::getWorldSize() / 2) % MPIWrapper::getWorldSize();
            std::shared_ptr<srcAllocator> src = std::make_shared<srcAllocator>(copySize, i);
            std::shared_ptr<dstAllocator> dst = std::make_shared<dstAllocator>(copySize, peer);
            copies.push_back(Copy(dst, src, copyDirection, copyType, iterations));

            std::stringstream s;
            s << i << "->" << peer;
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
        return "bisect_" + srcAllocator::getName() + "_to_" + dstAllocator::getName() + "_" + getCopyDirectionName(copyDirection) + "_" + getCopyTypeName(copyType);
    }
};

template <typename dstAllocator, typename srcAllocator, CopyType copyType>
class all_to_one_pattern : public TestcaseDstSrc<dstAllocator, srcAllocator> {
public:
    void run(size_t copySize) {
        OutputMatrix output(getName(), 1, MPIWrapper::getWorldSize());

        for (int targetDevice = 0; targetDevice < MPIWrapper::getWorldSize(); targetDevice++) {
            std::vector<Copy> copies;

            for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
                if (i == targetDevice) continue;
                std::shared_ptr<dstAllocator> dstAlloc = std::make_shared<dstAllocator>(copySize, targetDevice);
                std::shared_ptr<srcAllocator> srcAlloc = std::make_shared<srcAllocator>(copySize, i);
                copies.push_back(Copy(dstAlloc, srcAlloc, COPY_DIRECTION_WRITE, copyType, iterations));
            }

            auto results = NvLoom::doBenchmark(copies);

            output.set(0, targetDevice, std::reduce(results.begin(), results.end()));
        }
    }

    std::string getName() {
        return "all_to_one_" + srcAllocator::getName() + "_to_" + dstAllocator::getName() + "_" + getCopyDirectionName(COPY_DIRECTION_WRITE) + "_" + getCopyTypeName(copyType);
    }
};

template <typename dstAllocator, typename srcAllocator, CopyType copyType>
class all_from_one_pattern : public TestcaseDstSrc<dstAllocator, srcAllocator> {
public:
    void run(size_t copySize) {
        OutputMatrix output(getName(), 1, MPIWrapper::getWorldSize());

        for (int targetDevice = 0; targetDevice < MPIWrapper::getWorldSize(); targetDevice++) {
            std::vector<Copy> copies;
            std::shared_ptr<srcAllocator> srcAlloc = std::make_shared<srcAllocator>(copySize, targetDevice);

            for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
                if (i == targetDevice) continue;
                std::shared_ptr<dstAllocator> dstAlloc = std::make_shared<dstAllocator>(copySize, i);
                copies.push_back(Copy(dstAlloc, srcAlloc, COPY_DIRECTION_READ, copyType, iterations));
            }

            auto results = NvLoom::doBenchmark(copies);

            output.set(0, targetDevice, std::reduce(results.begin(), results.end()));
        }
    }

    std::string getName() {
        return "all_from_one_" + srcAllocator::getName()  + "_to_" + dstAllocator::getName() + "_" + getCopyDirectionName(COPY_DIRECTION_READ) + "_" + getCopyTypeName(copyType);
    }
};


template <typename dstAllocator, typename srcAllocator, CopyType copyType, CopyDirection copyDirection>
class multicast_one_to_all : public TestcaseDstSrc<dstAllocator, srcAllocator> {
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
        if constexpr(copyType == COPY_TYPE_MULTICAST_LD_REDUCE) {
            direction = "all_to_one";
        } else {
            direction = "one_to_all";
        }
        return "multicast_" + direction + "_" + getCopyTypeName(copyType);
    }
};

template <typename dstAllocator, typename srcAllocator, CopyType copyType>
class multicast_all_to_all : public TestcaseDstSrc<dstAllocator, srcAllocator> {
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

        OutputMatrix output(getName(), 1, 1, BUFFERING_DISABLED);
        output.set(0, 0, std::reduce(results.begin(), results.end()));
    }

    std::string getName() {
        return "multicast_all_to_all_" + getCopyTypeName(copyType);
    }
};

template <typename dstAllocator, typename srcAllocator>
class multicast_all_to_all_ld_reduce : public TestcaseDstSrc<dstAllocator, srcAllocator> {
public:
    void run(size_t copySize) {
        std::vector<Copy> copies;

        std::shared_ptr<srcAllocator> src = std::make_shared<srcAllocator>(copySize, 0);

        for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
            std::shared_ptr<dstAllocator> dst = std::make_shared<dstAllocator>(copySize, i);
            copies.push_back(Copy(dst, src, COPY_DIRECTION_READ, COPY_TYPE_MULTICAST_LD_REDUCE, iterations));
        }

        auto results = NvLoom::doBenchmark(copies);

        OutputMatrix output(getName(), 1, 1, BUFFERING_DISABLED);
        output.set(0, 0, std::reduce(results.begin(), results.end()));
    }

    std::string getName() {
        return "multicast_all_to_all_" + getCopyTypeName(COPY_TYPE_MULTICAST_LD_REDUCE);
    }
};

template <typename dstAllocator, typename srcAllocator>
class multicast_all_to_all_red : public TestcaseDstSrc<dstAllocator, srcAllocator> {
public:
    void run(size_t copySize) {
        OutputMatrix output(getName(), 1, 1, BUFFERING_DISABLED);

        std::vector<Copy> copies;
        std::shared_ptr<dstAllocator> dst = std::make_shared<dstAllocator>(copySize, 0);

        for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
            std::shared_ptr<srcAllocator> src = std::make_shared<srcAllocator>(copySize, i);
            Copy copy(dst, src, COPY_DIRECTION_WRITE, COPY_TYPE_MULTICAST_RED_ALL, iterations);
            copies.push_back(copy);

            // For the multicast_all_to_all_red testcase, we need to make sure that all source have the same uniqueId
            // Otherwise validation will fail
            // Each buffer is still unique, because the contents depend on both uniqueId and GPU_ID
            src->uniqueId = copies[0].src->uniqueId;
        }

        auto results = NvLoom::doBenchmark(copies);
        output.set(0, 0, std::reduce(results.begin(), results.end()));
    }

    std::string getName() {
        return "multicast_all_to_all_" + getCopyTypeName(COPY_TYPE_MULTICAST_RED_ALL);
    }
};

int randomSkipNumber(int randomNumber, int numberToSkip) {
    if (randomNumber >= numberToSkip) return randomNumber + 1;
    else return randomNumber;
}

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
        std::uniform_int_distribution<std::mt19937::result_type> dist(0, elems.size() - 1);

        if (elems.size() == 0) {
            break;
        }

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

std::mt19937 init_rng() {
    std::random_device random_dev;
    int seed = random_dev();
    // make sure all processes use the same seed
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::mt19937 rng(seed);
    return rng;
}

template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType, CopyCount copyCount>
class gpu_to_rack : public TestcaseDstSrc<dstAllocator, srcAllocator> {
public:
    void run(size_t copySize) {
        std::mt19937 rng = init_rng();

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

        std::vector<int> columnSeparators;
        for (int i = 0; i < NvLoom::getRackToProcessMap().size() - 1; i++) {
            columnSeparators.push_back((i + 1) * gpuToRackSamples - 1);
        }
        output.setColumnSeparators(columnSeparators);

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
class rack_to_rack_unidir : public TestcaseDstSrc<dstAllocator, srcAllocator> {
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
                for (int i = 0; i < peerCount; i++) {
                    std::shared_ptr<srcAllocator> src = std::make_shared<srcAllocator>(copySize, rack.second[i]);
                    std::shared_ptr<dstAllocator> dst = std::make_shared<dstAllocator>(copySize, peerRack.second[i]);
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
class rack_to_rack_bidir : public TestcaseDstSrc<dstAllocator, srcAllocator> {
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
        OutputMatrix output(getName(), MPIWrapper::getWorldSize(), MPIWrapper::getWorldSize(), BUFFERING_ENABLED, "ns");

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

    testcases[testcase->getName()] = std::move(testcase);
}

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator>
std::tuple<std::map<std::string, std::unique_ptr<Testcase> >, std::map<std::string, std::vector<std::string> > > buildTestcasesLower() {
    std::map<std::string, std::unique_ptr<Testcase> > testcases;
    std::map<std::string, std::vector<std::string> > suites;

    // pairwise
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM, COPY_COUNT_UNIDIR> >());
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM, COPY_COUNT_UNIDIR> >());

    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE> >());
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE> >());
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM> >());
    addTestcase(suites, "pairwise", testcases, std::make_unique<N_squared_pattern_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM> >());

    // fabric-stress
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM> >());
    addTestcase(suites, "fabric-stress", testcases, std::make_unique<bisect_pattern<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM> >());

    // all-to-one
    addTestcase(suites, "all-to-one", testcases, std::make_unique<all_to_one_pattern<unicastAllocator, unicastAllocator, COPY_TYPE_CE> >());
    addTestcase(suites, "all-to-one", testcases, std::make_unique<all_from_one_pattern<unicastAllocator, unicastAllocator, COPY_TYPE_CE> >());
    addTestcase(suites, "all-to-one", testcases, std::make_unique<all_to_one_pattern<unicastAllocator, unicastAllocator, COPY_TYPE_SM> >());
    addTestcase(suites, "all-to-one", testcases, std::make_unique<all_from_one_pattern<unicastAllocator, unicastAllocator, COPY_TYPE_SM> >());

    // multicast
    addTestcase(suites, "multicast", testcases, std::make_unique<multicast_one_to_all<multicastAllocator, DeviceMemoryAllocation, COPY_TYPE_MULTICAST_WRITE, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "multicast", testcases, std::make_unique<multicast_all_to_all<multicastAllocator, DeviceMemoryAllocation, COPY_TYPE_MULTICAST_WRITE> >());

    addTestcase(suites, "multicast", testcases, std::make_unique<multicast_one_to_all<DeviceMemoryAllocation, multicastAllocator, COPY_TYPE_MULTICAST_LD_REDUCE, COPY_DIRECTION_READ> >());
    addTestcase(suites, "multicast", testcases, std::make_unique<multicast_all_to_all_ld_reduce<DeviceMemoryAllocation, multicastAllocator> >());

    addTestcase(suites, "multicast", testcases, std::make_unique<multicast_one_to_all<multicastAllocator, DeviceMemoryAllocation, COPY_TYPE_MULTICAST_RED_SINGLE, COPY_DIRECTION_WRITE> >());
    addTestcase(suites, "multicast", testcases, std::make_unique<multicast_all_to_all_red<multicastAllocator, DeviceMemoryAllocation> >());

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

    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE, COPY_COUNT_BIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE, COPY_COUNT_BIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM, COPY_COUNT_BIDIR> >());
    addTestcase(suites, "gpu-to-rack", testcases, std::make_unique<gpu_to_rack<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM, COPY_COUNT_BIDIR> >());

    // rack-to-rack fabric saturation
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_unidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_unidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_unidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_unidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM> >());

    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_CE> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_CE> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_WRITE, COPY_TYPE_SM> >());
    addTestcase(suites, "rack-to-rack", testcases, std::make_unique<rack_to_rack_bidir<unicastAllocator, unicastAllocator, COPY_DIRECTION_READ, COPY_TYPE_SM> >());

    // csv
    addTestcase(suites, "csv", testcases, std::make_unique<csv_pattern<unicastAllocator, egmAllocator, multicastAllocator> >());

    // latency
    addTestcase(suites, "latency", testcases, std::make_unique<latency<unicastAllocator> >());
    addTestcase(suites, "latency", testcases, std::make_unique<latency<egmAllocator> >());

    return {std::move(testcases), std::move(suites)};
}

std::tuple<std::map<std::string, std::unique_ptr<Testcase> >, std::map<std::string, std::vector<std::string> > > buildTestcases(AllocatorStrategy strategy) {
    if (strategy == ALLOCATOR_STRATEGY_UNIQUE) {
        return buildTestcasesLower< MultinodeMemoryAllocationUnicast,
                                    MultinodeMemoryAllocationEGM,
                                    MultinodeMemoryAllocationMulticast>();
    } else if (strategy == ALLOCATOR_STRATEGY_REUSE) {
        return buildTestcasesLower< AllocationPool<MultinodeMemoryAllocationUnicast>,
                                    AllocationPool<MultinodeMemoryAllocationEGM>,
                                    MulticastPool>();
    } else if (strategy == ALLOCATOR_STRATEGY_CUDA_POOLS) {
        return buildTestcasesLower< MultinodeMemoryPoolAllocationUnicast,
                                    MultinodeMemoryPoolAllocationEGM,
                                    MultinodeMemoryAllocationMulticast>();
    }
    ASSERT(0);
}
