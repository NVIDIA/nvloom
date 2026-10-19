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

#include <cstring>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cuda.h>
#include <nvml.h>

#include "nvloom.h"
#include "util.h"

#define STRING_LENGTH 128

static std::string getPaddedProcessId(int id) {
    // max printed number will be MPIWrapper::getWorldSize() - 1
    int paddingSize = MPIWrapper::getWorldSize() > 1 ? (int) log10(MPIWrapper::getWorldSize() - 1) + 1 : 1;
    std::stringstream s;
    s << std::setfill(' ') << std::setw(paddingSize) << id;
    return s.str();
}

static std::string jsonEscape(const std::string &input) {
    std::stringstream sstream;
    for (unsigned char c : input) {
        switch (c) {
            case '\"': sstream << "\\\""; break;
            case '\\': sstream << "\\\\"; break;
            case '\b': sstream << "\\b"; break;
            case '\f': sstream << "\\f"; break;
            case '\n': sstream << "\\n"; break;
            case '\r': sstream << "\\r"; break;
            case '\t': sstream << "\\t"; break;
            default:
                if (c < 0x20) {
                    sstream << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int) c
                            << std::dec << std::setw(0) << std::setfill(' ');
                } else {
                    sstream << c;
                }
        }
    }
    return sstream.str();
}

static std::string getDeviceDisplayInfo(int deviceOrdinal) {
    std::stringstream sstream;
    CUdevice dev;
    char name[STRING_LENGTH];
    int busId, deviceId, domainId;

    CU_ASSERT(cuDeviceGet(&dev, deviceOrdinal));
    CU_ASSERT(cuDeviceGetName(name, STRING_LENGTH, dev));
    CU_ASSERT(cuDeviceGetAttribute(&domainId, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev));
    CU_ASSERT(cuDeviceGetAttribute(&busId, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev));
    CU_ASSERT(cuDeviceGetAttribute(&deviceId, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev));
    sstream << name << " (" <<
        std::hex << std::setw(8) << std::setfill('0') << domainId << ":" <<
        std::hex << std::setw(2) << std::setfill('0') << busId << ":" <<
        std::hex << std::setw(2) << std::setfill('0') << deviceId << ")" <<
        std::dec << std::setfill(' ') << std::setw(0);  // reset formatting

    return sstream.str();
}

int discoverRanks(std::map<std::string, std::vector<int>> &rackToProcessMap) {
    char localHostname[STRING_LENGTH] = {};
    gethostname(localHostname, STRING_LENGTH - 1);

    int deviceCount;
    CU_ASSERT(cuInit(0));
    CU_ASSERT(cuDeviceGetCount(&deviceCount));

    // Exchange hostnames
    std::vector<char> hostnameExchange(MPIWrapper::getWorldSize() * STRING_LENGTH);
    MPI_Allgather(localHostname, STRING_LENGTH, MPI_BYTE, &hostnameExchange[0], STRING_LENGTH, MPI_BYTE, MPI_COMM_WORLD);

    int localRank, localDevice;

    // Find local rank based on hostnames
    localRank = 0;
    for (int i = 0; i < MPIWrapper::getWorldRank(); i++) {
        if (strncmp(localHostname, &hostnameExchange[i * STRING_LENGTH], STRING_LENGTH) == 0) {
            localRank++;
        }
    }

    // It's not recommended to run more ranks per node than GPU count, but we want to make sure we handle it gracefully
    localDevice = localRank % deviceCount;

    // Exchange GPU UUIDs to detect processes sharing the same GPU
    CUdevice cuDev;
    CU_ASSERT(cuDeviceGet(&cuDev, localDevice));
    CUuuid localUuid;
    CU_ASSERT(cuDeviceGetUuid_v2(&localUuid, cuDev));

    std::vector<CUuuid> uuidExchange(MPIWrapper::getWorldSize());
    MPI_Allgather(&localUuid, sizeof(CUuuid), MPI_BYTE, &uuidExchange[0], sizeof(CUuuid), MPI_BYTE, MPI_COMM_WORLD);

    // Check for duplicate UUIDs — warn once per pair, only from the lower-ranked process
    for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
        if (i == MPIWrapper::getWorldRank()) continue;
        if (strncmp(localHostname, &hostnameExchange[i * STRING_LENGTH], STRING_LENGTH) == 0
                && memcmp(&localUuid, &uuidExchange[i], sizeof(CUuuid)) == 0
                && MPIWrapper::getWorldRank() < i) {
            std::cout << "WARNING: processes " << MPIWrapper::getWorldRank() << " and " << i
                      << " are sharing the same GPU on " << localHostname
                      << ". Please reduce number of processes or adjust CUDA_VISIBLE_DEVICES." << std::endl;
        }
    }

    // Exchange rack GUIDs
    std::string rackGuid = getRackGuid(localDevice);
    // GUID length is equal for every process
    std::vector<char> rackGuidExchange(MPIWrapper::getWorldSize() * rackGuid.length());
    MPI_Allgather(&rackGuid[0], rackGuid.length(), MPI_BYTE, &rackGuidExchange[0], rackGuid.length(), MPI_BYTE, MPI_COMM_WORLD);

    // Exchange device names
    std::string localDeviceName = getDeviceDisplayInfo(localDevice);
    ASSERT(localDeviceName.size() < STRING_LENGTH);
    localDeviceName.resize(STRING_LENGTH);

    std::vector<char> deviceNameExchange(MPIWrapper::getWorldSize() * STRING_LENGTH, 0);
    MPI_Allgather(&localDeviceName[0], STRING_LENGTH, MPI_BYTE, &deviceNameExchange[0], STRING_LENGTH, MPI_BYTE, MPI_COMM_WORLD);

    // Exchange device ids
    std::vector<int> localDeviceIdExchange(MPIWrapper::getWorldSize(), -1);
    MPI_Allgather(&localDevice, 1, MPI_INT, &localDeviceIdExchange[0], 1, MPI_INT, MPI_COMM_WORLD);

    // Print gathered info
    for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
        char *deviceName = &deviceNameExchange[i * STRING_LENGTH];
        std::string rackGuidName(&rackGuidExchange[i * rackGuid.length()], rackGuid.length());
        OUTPUT << "Process " << getPaddedProcessId(i) << " (" << &hostnameExchange[i * STRING_LENGTH] << "): device " << localDeviceIdExchange[i];
        OUTPUT << ": " << deviceName << "; rackGuid: " << trimRackGuid(rackGuidName) << std::endl;
        rackToProcessMap[trimRackGuid(rackGuidName)].push_back(i);

        std::stringstream processName;
        processName << &hostnameExchange[i * STRING_LENGTH] << ":" << localDeviceIdExchange[i];
        NvLoom::setProcessName(i, processName.str());
    }
    OUTPUT << std::endl;

    return localDevice;
}

void OutputMatrix::incrementalPrint(int stage) {
    if (stage == 0) {
        OUTPUT << "\t";
        for (int i = 0; i < labelsX.size(); i++) {
            OUTPUT << labelsX[i] << "\t";
            if (std::find(columnSeparators.begin(), columnSeparators.end(), i) != columnSeparators.end()) {
                OUTPUT << "| \t";
            }
        }
        OUTPUT << "\n";
    }

    if (stage % dimX == 0) {
        OUTPUT << labelsY[stage / dimX] << "\t";
    }

    if (data[stage] == 0) {
        OUTPUT << "N/A";
    } else {
        OUTPUT << std::fixed << std::setprecision(2) << data[stage];
    }

    if (notes[stage] != "") {
        OUTPUT << ";(" << notes[stage] << ")";
    }

    OUTPUT << "\t";

    if (std::find(columnSeparators.begin(), columnSeparators.end(), stage % dimX) != columnSeparators.end()) {
        OUTPUT << "|\t";
    }

    if (stage % dimX == (dimX - 1)) {
        OUTPUT << "\n";
    }
}

void OutputMatrix::setJsonOutputFile(std::string _jsonOutputFile) {
    jsonOutputFile = _jsonOutputFile;
    if (jsonOutputFile == "") {
        return;
    }

    // Only rank 0 writes the JSON file, mirroring OUTPUT. Otherwise every rank
    // would truncate and interleave writes into the same path.
    if (MPIWrapper::getWorldRank() != 0) {
        return;
    }

    jsonFile = std::ofstream(jsonOutputFile);
    ASSERT(jsonFile.good());

    jsonFile << "{" << std::endl;
    jsonFile << "\"tests\": [" << std::endl;
}

void OutputMatrix::finalize() {
    if (jsonFile.is_open()) {
        jsonFile << "]" << std::endl;
        jsonFile << "}" << std::endl;
        jsonFile.close();
    }
}

void OutputMatrix::printJson(StatisticsStrings stats) {
    if (!jsonFile.is_open()) {
        return;
    }
    if (jsonTestCount > 0) {
        jsonFile << "," << std::endl;
    }
    jsonTestCount++;
    jsonFile << "{" << std::endl;
    jsonFile << "\"name\": \"" << jsonEscape(name) << "\"," << std::endl;
    jsonFile << "\"unit\": \"" << jsonEscape(unit) << "\"," << std::endl;
    jsonFile << "\"avg\": \"" << stats.avg << "\"," << std::endl;
    jsonFile << "\"med\": \"" << stats.med << "\"," << std::endl;
    jsonFile << "\"min\": \"" << stats.min << "\"," << std::endl;
    jsonFile << "\"max\": \"" << stats.max << "\"," << std::endl;
    jsonFile << "\"total\": \"" << stats.total << "\"," << std::endl;

    jsonFile << "\"labelsX\": [" << std::endl;
    for (int i = 0; i < labelsX.size(); i++) {
        jsonFile << "\"" << jsonEscape(labelsX[i]) << "\"";
        if (i < labelsX.size() - 1) {
            jsonFile << ",";
        }
    }
    jsonFile << "]," << std::endl;
    jsonFile << "\"labelsY\": [" << std::endl;
    for (int i = 0; i < labelsY.size(); i++) {
        jsonFile << "\"" << jsonEscape(labelsY[i]) << "\"";
        if (i < labelsY.size() - 1) {
            jsonFile << ",";
        }
    }
    jsonFile << "]," << std::endl;

    jsonFile << "\"results\": [" << std::endl;
    jsonFile << "[";

    for (int i = 0; i < data.size(); i++) {
        if (data[i] == 0) {
            jsonFile << "\"N/A\"";
        } else {
            jsonFile << "\"" << data[i] << "\"";
        }

        if (i % dimX != (dimX - 1)) {
            jsonFile << ",";
        } else {
            jsonFile << "]";
            if (i < data.size() - 1) {
                jsonFile << "," << std::endl << "[";
            }
        }
    }

    jsonFile << "]" << std::endl;

    jsonFile << "}";
}

size_t getBufferSizeInBytes(std::string bufferSizeStr) {
    bufferSizeStr = toLower(bufferSizeStr);
    size_t bufferSizeInBytes;
    if (std::isdigit(bufferSizeStr.back())) {
        bufferSizeInBytes = std::stoull(bufferSizeStr);
    } else {
        bufferSizeInBytes = std::stoull(bufferSizeStr.substr(0, bufferSizeStr.size() - 1));
        if (bufferSizeStr.back() == 'k') {
            bufferSizeInBytes *= (size_t) 1024;
        } else if (bufferSizeStr.back() == 'm') {
            bufferSizeInBytes *= (size_t) 1024 * 1024;
        } else if (bufferSizeStr.back() == 'g') {
            bufferSizeInBytes *= (size_t) 1024 * 1024 * 1024;
        } else if (bufferSizeStr.back() == 't') {
            bufferSizeInBytes *= (size_t) 1024 * 1024 * 1024 * 1024;
        } else {
            throw std::runtime_error("Invalid buffer size: " + bufferSizeStr);
        }
    }
    return bufferSizeInBytes;
}