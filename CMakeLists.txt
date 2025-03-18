#[[
SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
]]

cmake_minimum_required(VERSION 3.20)

project(nvloom_cli
    LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE "Release")
endif()

set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS program_options REQUIRED)

set(src
    cli/nvloom_cli.cpp
    cli/util.cpp
    cli/testcases.cpp
    library/allocators.cpp
    library/nvloom.cpp
    library/kernels.cu
)

add_executable(nvloom_cli ${src})
target_include_directories(nvloom_cli PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_CURRENT_SOURCE_DIR}/library .)
target_link_libraries(nvloom_cli cuda nvidia-ml Boost::program_options)

set_target_properties(nvloom_cli PROPERTIES CUDA_ARCHITECTURES "86;90;100")

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(nvloom_cli MPI::MPI_CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMULTINODE")
