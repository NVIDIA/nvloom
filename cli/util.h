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

#ifndef UTIL_H_
#define UTIL_H_

#include <numeric>
#include <vector>
#include <algorithm>
#include <iomanip>

// returns which GPU should this process run on
int discoverRanks(std::map<std::string, std::vector<int>> &rackToProcessMap);

class OutputMatrix {
private:
    std::string name;
    int dimX;
    int dimY;
    int currentlyPrinted = 0;
    std::vector<double> data;
    std::vector<bool> initialized;
    std::vector<std::string> notes;
    std::vector<std::string> labelsX;
    std::vector<std::string> labelsY;
    std::vector<int> columnSeparators;

    void fillLabels(std::vector<std::string>& vec, int size) {
        for (int i = 0; i < size; i++) {
            vec.push_back(std::to_string(i));
        }
    }

    void incrementalPrint(int stage);

public:
    OutputMatrix(std::string _name, int _dimX, int _dimY) : name(_name), dimX(_dimX), dimY(_dimY), data(dimX * dimY, 0), initialized(dimX * dimY, false), notes(dimX * dimY, "") {
        fillLabels(labelsX, dimX);
        fillLabels(labelsY, dimY);
    }

    ~OutputMatrix() {
        for (int i = currentlyPrinted; i < dimX * dimY; i++) {
            incrementalPrint(i);
        }

        std::vector<double> filteredData;
        // C++20 has "erase", which does exactly what we want
        // We're on C++17, so let's do it manually
        for (auto elem : data) {
            if (elem != 0) {
                filteredData.push_back(elem);
            } 
        }

        if (filteredData.size() == 0) {
            OUTPUT << std::endl;
            OUTPUT << "MinBW " << name << " N/A" << std::endl;
            OUTPUT << "AvgBW " << name << " N/A" << std::endl;
            OUTPUT << "MaxBW " << name << " N/A" << std::endl;
            OUTPUT << "TotalBW " << name << " N/A" << std::endl;
            OUTPUT << std::endl;
        } else {
            auto sum = std::reduce(filteredData.begin(), filteredData.end());
            auto [min, max] = std::minmax_element(filteredData.begin(), filteredData.end());
            OUTPUT << std::endl;
            OUTPUT << std::fixed << std::setprecision(2);
            OUTPUT << "MinBW " << name << " " << *min << std::endl;
            OUTPUT << "AvgBW " << name << " " << sum / filteredData.size() << std::endl;
            OUTPUT << "MaxBW " << name << " " << *max << std::endl;
            OUTPUT << "TotalBW " << name << " " << sum << std::endl;
            OUTPUT << std::endl;
        }
    }

    void set(int x, int y, double value, std::string note = "") {
        ASSERT(x >= 0);
        ASSERT(x < dimX);
        ASSERT(y >= 0);
        ASSERT(y < dimY);
        
        int id = y * dimX + x;
        data[id] = value;
        notes[id] = note;
        initialized[id] = true;
        if (currentlyPrinted == id) {
            int i = id;
            while (initialized[i] && i < dimX * dimY) {
                incrementalPrint(i);
                currentlyPrinted = i + 1;
                i++;
            }
        }
    }

    double get(int x, int y) {
        ASSERT(x >= 0);
        ASSERT(x < dimX);
        ASSERT(y >= 0);
        ASSERT(y < dimY);
        return data[y * dimX + x]; 
    }

    void setLabelsX(std::vector<std::string> _labelsX) {
        ASSERT(_labelsX.size() == dimX);
        labelsX = _labelsX;
    }

    void setLabelsY(std::vector<std::string> _labelsY) {
        ASSERT(_labelsY.size() == dimY);
        labelsY = _labelsY;
    }

    void setColumnSeparators(std::vector<int> _columnSeparators) {
        columnSeparators = _columnSeparators;
    }
};

#endif  // UTIL_H_
