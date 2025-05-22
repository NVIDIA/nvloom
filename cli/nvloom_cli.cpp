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

#include "nvloom.h"
#include "util.h"
#include "testcases.h"

#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>
#include <memory>

#define NVLOOM_VERSION "1.1.0"
#ifndef GIT_COMMIT
#define GIT_COMMIT "unknown"
#endif

bool richOutput = false;
int gpuToRackSamples = 5;

int run_program(int argc, char **argv) {
    boost::program_options::options_description opts("nvloom CLI");
    std::vector<std::string> testcasesToRun;
    std::vector<std::string> suitesToRun;
    std::string allocatorStrategyString;

    bool listTestcases = false;
    int bufferSizeInMiB = 512;

    std::string suitesOptionDescription("Suite(s) to run (by name): all-to-one, egm, fabric-stress, gpu-to-rack, multicast, pairwise, rack-to-rack");
    opts.add_options()
        ("help,h", "Produce help message")
        ("bufferSize,b", boost::program_options::value<int>(&bufferSizeInMiB)->default_value(bufferSizeInMiB), "Buffer size in MiB")
        ("testcase,t", boost::program_options::value<std::vector<std::string>>(&testcasesToRun)->multitoken(), "Testcase(s) to run (by name)")
        ("suite,s", boost::program_options::value<std::vector<std::string>>(&suitesToRun)->multitoken(), suitesOptionDescription.c_str())
        ("listTestcases,l", boost::program_options::bool_switch(&listTestcases)->default_value(listTestcases), "List testcases")
        ("richOutput,r", boost::program_options::bool_switch(&richOutput)->default_value(richOutput), "Rich output")
        ("allocatorStrategy,a", boost::program_options::value<std::string>(&allocatorStrategyString)->default_value("reuse"), "Allocator strategy: choose between unique and reuse")
        ("gpuToRackSamples", boost::program_options::value<int>(&gpuToRackSamples)->default_value(gpuToRackSamples), "Number of per-rack samples to use in gpu_to_rack testcases")
        ;

    boost::program_options::variables_map vm;
    try {
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, opts), vm);
        boost::program_options::notify(vm);
    } catch (const boost::program_options::error& e) {
        std::cerr << "Couldn't parse command line arguments properly:\n";
        std::cerr << e.what() << '\n' << '\n';
        std::cerr << opts << "\n";
        return 1;
    }

    if (vm.count("help")) {
        OUTPUT << opts << "\n";
        return 0;
    }

    OUTPUT << "nvloom_cli " << NVLOOM_VERSION << std::endl;
    OUTPUT << "git commit: " << GIT_COMMIT << std::endl;

    AllocatorStrategy allocatorStrategy;
    if (allocatorStrategyString == "reuse"){
        allocatorStrategy = ALLOCATOR_STRATEGY_REUSE;
    } else if (allocatorStrategyString == "unique") {
        allocatorStrategy = ALLOCATOR_STRATEGY_UNIQUE;
    } else {
        std::cerr << "Unknown value for the allocatorStrategy argument: " << allocatorStrategyString << "\n";
        OUTPUT << opts << "\n";
        return 0;
    }

    OUTPUT << "Allocation strategy: " << allocatorStrategyString << std::endl;

    size_t bufferSizeInB = (size_t) bufferSizeInMiB * 1024 * 1024;

    OUTPUT << "Buffer size: " << bufferSizeInMiB << " MiB" << std::endl;

    auto [testcases, suites] = buildTestcases(allocatorStrategy);

    if (listTestcases) {
        OUTPUT << "Available testcases: " << std::endl;
        for (auto const& [testcaseName, testcaseFunction] : testcases) {
            OUTPUT << testcaseName << std::endl;
        }
        return 0;
    }

    std::map<std::string, std::vector<int> > rackToProcessMap;
    int localDevice = discoverRanks(rackToProcessMap);
    NvLoom::initialize(localDevice, rackToProcessMap);

    std::set<std::string> testcasesToRunSet;

    for (auto suite: suitesToRun) {
        if (suites.count(suite) == 0) {
            std::cerr << "No such suite as \"" << suite << "\"\n";
            return 1;
        }
        for (auto testcase: suites[suite]) {
            testcasesToRunSet.insert(testcase);
        }
    }

    for (auto testcase: testcasesToRun) {
        if (testcases.count(testcase) == 0) {
            std::cerr << "No such testcase as \"" << testcase << "\"\n";
            return 1;
        }
        testcasesToRunSet.insert(testcase);
    }

    if (testcasesToRunSet.size() == 0) {
        for (auto& [testcaseName, testcase] : testcases) {
            testcasesToRunSet.insert(testcaseName);
        }
    }

    for (auto testcase : testcasesToRunSet) {
        OUTPUT << "Running " << testcase << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();
        testcases[testcase]->filterRun(bufferSizeInB);
        clearAllocationPools();
        auto endTime = std::chrono::high_resolution_clock::now();
        OUTPUT << "ExecutionTime " << testcase << " " << std::chrono::duration<double>(endTime - startTime).count() << " s" << std::endl;
        OUTPUT << "Done " << testcase << std::endl;
        OUTPUT << std::endl;
    }

    return 0;
}

int main(int argc, char **argv) {
    std::cout << std::unitbuf;
    int ret = run_program(argc, argv);

    NvLoom::finalize();
    return ret;
}
