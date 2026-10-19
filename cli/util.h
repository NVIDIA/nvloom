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
#include <stdexcept>
#include <fstream>

extern char bandwidthUnit;

// returns which GPU should this process run on
int discoverRanks(std::map<std::string, std::vector<int>> &rackToProcessMap);

enum Buffering {
    BUFFERING_ENABLED,
    BUFFERING_DISABLED
};

class OutputMatrix {
public:
    enum class MeasurementType {
        BANDWIDTH,
        LATENCY,
    };

    class StatisticsStrings {
    public:
        std::string min;
        std::string max;
        std::string avg;
        std::string med;
        std::string total;
    };
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
    MeasurementType measurementType;
    std::string unit;
    double unitMultiplier;
    static inline std::string jsonOutputFile;
    static inline std::ofstream jsonFile;
    static inline int jsonTestCount = 0;

    void fillLabels(std::vector<std::string>& vec, int size) {
        for (int i = 0; i < size; i++) {
            vec.push_back(std::to_string(i));
        }
    }

    void incrementalPrint(int stage);

    void printJson(StatisticsStrings stats);

    std::string getStatisticsString(double value) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << value;
        return ss.str();
    }

    StatisticsStrings getStatisticsStrings() {
        StatisticsStrings stats;
        stats.min = "N/A";
        stats.max = "N/A";
        stats.avg = "N/A";
        stats.med = "N/A";
        stats.total = "N/A";

        std::vector<double> filteredData;
        // C++20 has "erase", which does exactly what we want
        // We're on C++17, so let's do it manually
        for (auto elem : data) {
            if (elem != 0) {
                filteredData.push_back(elem);
            }
        }

        if (filteredData.size() > 0) {
            auto sum = std::reduce(filteredData.begin(), filteredData.end());
            std::sort(filteredData.begin(), filteredData.end());
            size_t n = filteredData.size();
            double median = (n % 2 == 0) ? (filteredData[n/2 - 1] + filteredData[n/2]) / 2.0 : filteredData[n/2];
            stats.min = getStatisticsString(filteredData.front());
            stats.max = getStatisticsString(filteredData.back());
            stats.avg = getStatisticsString(sum / n);
            stats.med = getStatisticsString(median);
            stats.total = getStatisticsString(sum);
        }
        return stats;
    }
public:
    OutputMatrix(std::string _name, int _dimX, int _dimY, Buffering _buffering = BUFFERING_ENABLED, MeasurementType _measurementType = MeasurementType::BANDWIDTH) : name(_name), dimX(_dimX), dimY(_dimY), data(dimX * dimY, 0), initialized(dimX * dimY, false), notes(dimX * dimY, ""), buffering(_buffering), measurementType(_measurementType) {
        fillLabels(labelsX, dimX);
        fillLabels(labelsY, dimY);

        unitMultiplier = 1.0;
        if (measurementType == MeasurementType::BANDWIDTH) {
            if (bandwidthUnit == 'k') {
                unitMultiplier = 1000*1000;
                unit = "KB/s";
            } else if (bandwidthUnit == 'm') {
                unitMultiplier = 1000;
                unit = "MB/s";
            } else if (bandwidthUnit == 'g') {
                unitMultiplier = 1;
                unit = "GB/s";
            } else if (bandwidthUnit == 't') {
                unitMultiplier = (double) 1.0/1000.0;
                unit = "TB/s";
            } else {
                throw std::runtime_error("Invalid bandwidth unit passed to OutputMatrix constructor: " + std::to_string(bandwidthUnit));
            }
        } else if (measurementType == MeasurementType::LATENCY) {
            unit = "ns";
        } else {
            throw std::runtime_error("Invalid measurement type passed to OutputMatrix constructor");
        }
    }

    ~OutputMatrix() {
        for (int i = currentlyPrinted; i < dimX * dimY; i++) {
            incrementalPrint(i);
        }

        auto stats = getStatisticsStrings();
        OUTPUT << std::endl;
        OUTPUT << "Min " << name << " " << stats.min << " " << unit << std::endl;
        OUTPUT << "Avg " << name << " " << stats.avg << " " << unit << std::endl;
        OUTPUT << "Med " << name << " " << stats.med << " " << unit << std::endl;
        OUTPUT << "Max " << name << " " << stats.max << " " << unit << std::endl;
        OUTPUT << "Total " << name << " " << stats.total << " " << unit << std::endl;
        OUTPUT << std::endl;

        printJson(stats);
    }

    void set(int x, int y, double value, std::string note = "") {
        ASSERT(x >= 0);
        ASSERT(x < dimX);
        ASSERT(y >= 0);
        ASSERT(y < dimY);

        value *= unitMultiplier;

        int id = y * dimX + x;
        data[id] = value;
        notes[id] = note;
        initialized[id] = true;
        if (buffering == BUFFERING_ENABLED && currentlyPrinted == id) {
            int i = id;
            while (i < dimX * dimY && initialized[i]) {
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

    static void setJsonOutputFile(std::string _jsonOutputFile);
    static void finalize();
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

static inline size_t getSizeT(std::string str, std::string field) {
    try {
        return std::stoull(str);
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error("Invalid " + field + ": " + str);
    }
}

size_t getBufferSizeInBytes(std::string bufferSizeStr);

#endif  // UTIL_H_
