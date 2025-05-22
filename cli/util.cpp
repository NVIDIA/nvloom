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

#include <cstring>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

#include "nvloom.h"
#include "util.h"

#define STRING_LENGTH 128

static std::string getPaddedProcessId(int id) {
    // max printed number will be MPIWrapper::getWorldSize() - 1
    int paddingSize = (int) log10(MPIWrapper::getWorldSize() - 1) + 1;
    std::stringstream s;
    s << std::setfill(' ') << std::setw(paddingSize) << id;
    return s.str();
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

    // Unconditionally emitting a warning, once per node
    if (localRank == deviceCount) {
        std::cout << "WARNING: there are more processes than GPUs on " << localHostname << ". Please reduce number of processes to match GPU count." << std::endl;
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


