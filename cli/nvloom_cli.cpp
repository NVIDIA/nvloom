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

#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>
#include <memory>

#define NVLOOM_VERSION "1.2.0"
#ifndef GIT_COMMIT
#define GIT_COMMIT "unknown"
#endif

bool richOutput = false;
int gpuToRackSamples = 5;
int iterations = NvLoom::getDefaultIterationCount();

bool shouldContinue(boost::program_options::variables_map &vm, int iteration, std::chrono::time_point<std::chrono::high_resolution_clock> startTime) {
    if (vm["repeat"].defaulted() && vm["duration"].defaulted()) {
        return false;
    }

    if (!vm["repeat"].defaulted()) {
        return iteration + 1 < vm["repeat"].as<int>();
    }

    if (!vm["duration"].defaulted()) {
        auto duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count();
        return duration < vm["duration"].as<int>();
    }

    ASSERT(0);
    return false;
}

int run_program(int argc, char **argv) {
    boost::program_options::options_description opts("nvloom CLI");
    std::vector<std::string> testcasesToRun;
    std::vector<std::string> suitesToRun;
    std::string allocatorStrategyString;

    bool listTestcases = false;
    int bufferSizeInMiB = 512;
    int repeat = 1;
    int duration = -1;

    std::string suitesOptionDescription("Suite(s) to run (by name): all-to-one, egm, fabric-stress, gpu-to-rack, multicast, pairwise, rack-to-rack");
    opts.add_options()
        ("help,h", "Produce help message")
        ("bufferSize,b", boost::program_options::value<int>(&bufferSizeInMiB)->default_value(bufferSizeInMiB), "Buffer size in MiB")
        ("testcase,t", boost::program_options::value<std::vector<std::string>>(&testcasesToRun)->multitoken(), "Testcase(s) to run (by name)")
        ("suite,s", boost::program_options::value<std::vector<std::string>>(&suitesToRun)->multitoken(), suitesOptionDescription.c_str())
        ("listTestcases,l", boost::program_options::bool_switch(&listTestcases)->default_value(listTestcases), "List testcases")
        ("richOutput,r", boost::program_options::bool_switch(&richOutput)->default_value(richOutput), "Rich output")
        ("allocatorStrategy,a", boost::program_options::value<std::string>(&allocatorStrategyString)->default_value("reuse"), "Allocator strategy: choose between unique, reuse and cudapool")
        ("gpuToRackSamples", boost::program_options::value<int>(&gpuToRackSamples)->default_value(gpuToRackSamples), "Number of per-rack samples to use in gpu_to_rack testcases")
        ("iterations,i", boost::program_options::value<int>(&iterations)->default_value(iterations), "Number of copy iterations within the testcase to run, not including the warmup iteration")
        ("repeat,c", boost::program_options::value<int>(&repeat)->default_value(repeat), "Number of times to repeat each testcase")
        ("duration,d", boost::program_options::value<int>(&duration)->default_value(duration), "Duration of each testcase in seconds")
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

    if (!vm["repeat"].defaulted() && !vm["duration"].defaulted()) {
        std::cerr << "Cannot specify both repeat and duration\n";
        return 1;
    }

    OUTPUT << "nvloom_cli " << NVLOOM_VERSION << std::endl;
    OUTPUT << "git commit: " << GIT_COMMIT << std::endl;

    AllocatorStrategy allocatorStrategy;
    if (allocatorStrategyString == "reuse"){
        allocatorStrategy = ALLOCATOR_STRATEGY_REUSE;
    } else if (allocatorStrategyString == "unique") {
        allocatorStrategy = ALLOCATOR_STRATEGY_UNIQUE;
    } else if (allocatorStrategyString == "cudapool") {
        allocatorStrategy = ALLOCATOR_STRATEGY_CUDA_POOLS;
    } else {
        std::cerr << "Unknown value for the allocatorStrategy argument: " << allocatorStrategyString << "\n";
        OUTPUT << opts << "\n";
        return 0;
    }

    OUTPUT << "Allocation strategy: " << allocatorStrategyString << std::endl;

    size_t bufferSizeInB = (size_t) bufferSizeInMiB * 1024 * 1024;

    OUTPUT << "Buffer size: " << bufferSizeInMiB << " MiB" << std::endl;

    OUTPUT << "Iteration count: " << iterations << std::endl;

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
        int iterationCount = 0;
        auto loopStartTime = std::chrono::high_resolution_clock::now();
        while (true) {
            std::string testcaseName = testcase;
            if (!vm["repeat"].defaulted() || !vm["duration"].defaulted()) {
                testcaseName += "_iter_" + std::to_string(iterationCount);
            }

            OUTPUT << "Running " << testcaseName << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();
            testcases[testcase]->filterRun(bufferSizeInB);
            auto endTime = std::chrono::high_resolution_clock::now();

            bool shouldContinueIteration = shouldContinue(vm, iterationCount, loopStartTime);
            if (!shouldContinueIteration) {
                // We're only clearing the pools on last iteration of the loop
                // But we still want to include the time it took to clear the pools in the output
                clearAllocationPools();
            }

            OUTPUT << "ExecutionTime " << testcaseName << " " << std::chrono::duration<double>(endTime - startTime).count() << " s" << std::endl;
            OUTPUT << "Done " << testcaseName << std::endl;
            OUTPUT << std::endl;

            if (!shouldContinueIteration) {
                break;
            }

            iterationCount++;
        }
    }

    return 0;
}

int main(int argc, char **argv) {
    std::cout << std::unitbuf;
    int ret = run_program(argc, argv);

    NvLoom::finalize();
    return ret;
}
