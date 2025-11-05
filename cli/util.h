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

#ifndef UTIL_H_
#define UTIL_H_

#include <numeric>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <string>
#include <cctype>

// returns which GPU should this process run on
int discoverRanks(std::map<std::string, std::vector<int>> &rackToProcessMap);

enum Buffering {
    BUFFERING_ENABLED,
    BUFFERING_DISABLED
};

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
    Buffering buffering;
    std::string unit;

    void fillLabels(std::vector<std::string>& vec, int size) {
        for (int i = 0; i < size; i++) {
            vec.push_back(std::to_string(i));
        }
    }

    void incrementalPrint(int stage);

public:
    OutputMatrix(std::string _name, int _dimX, int _dimY, Buffering _buffering = BUFFERING_ENABLED, std::string _unit = "GB/s") : name(_name), dimX(_dimX), dimY(_dimY), data(dimX * dimY, 0), initialized(dimX * dimY, false), notes(dimX * dimY, ""), buffering(_buffering), unit(_unit) {
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
            OUTPUT << "Min " << name << " N/A " << unit << std::endl;
            OUTPUT << "Avg " << name << " N/A " << unit << std::endl;
            OUTPUT << "Max " << name << " N/A " << unit << std::endl;
            OUTPUT << "Total " << name << " N/A " << unit << std::endl;
            OUTPUT << std::endl;
        } else {
            auto sum = std::reduce(filteredData.begin(), filteredData.end());
            auto [min, max] = std::minmax_element(filteredData.begin(), filteredData.end());
            OUTPUT << std::endl;
            OUTPUT << std::fixed << std::setprecision(2);
            OUTPUT << "Min " << name << " " << *min << " " << unit << std::endl;
            OUTPUT << "Avg " << name << " " << sum / filteredData.size() << " " << unit << std::endl;
            OUTPUT << "Max " << name << " " << *max << " " << unit << std::endl;
            OUTPUT << "Total " << name << " " << sum << " " << unit << std::endl;
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
        if (buffering == BUFFERING_ENABLED && currentlyPrinted == id) {
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

template <typename T, typename U>
static std::vector<T> getKeys(const std::map<T, U> &map) {
    std::vector<T> keys;
    for (auto &pair : map) {
        keys.push_back(pair.first);
    }
    return keys;
}

static inline std::string trimWhitespace(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c) {return !std::isspace(c);}));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char c) {return !std::isspace(c);}).base(), s.end());
    return s;
}

static inline std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return s;
}

std::string getDriverVersion();
std::string getCudaVersion();

#endif  // UTIL_H_
