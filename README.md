# nvloom

Nvloom is a set of tools designed to scalably test MNNVL fabrics. It consists of a library and command-line tool. 

**Disclaimer: nvloom is still in active and early development. All interfaces might be subject to change in the future.**

## library

The purpose of the library is to simplify integrating MNNVL testing into an application.

Mininium requirements:
- CUDA 12.8.1
- 570.124.06 driver
- MPI
- C++17
- Cmake 3.20

There are two ways to integrate the library into an application

### .so file

The library can be built as a shared library.

`cd library && cmake . && make`

This will build `libnvloom.so` which can be linked from the application. 

The `library` directory can be added to the build system as an include directory.

### source code

Alternatively, the libray source code can be added to an application.

Both `kernels.cu` and `nvloom.cpp` need to be added to the build system.

### library entry point

In the application, including `nvloom.h` will include all necessary headers.

### library design

Can be found in library/README.md.

## command line tool

While the library is designed to be integrated into a multitude of projects, for ease of use a sample command line interface is provided to get started.

Tests are grouped into suites for ease of use.

Current suites:
1. `pairwise` - typical O(n^2), one pair at a time testing
2. `fabric-stress` - testcases where every single GPU is writing/reading to/from some other GPU at the same time
3. `all-to-one` - testcases where multiple GPUs access memory on a single GPU
4. `multicast` - multicast benchmarks
5. `egm` - typical O(n^2) pairwise testing for EGM
6. `gpu-to-rack` - for each GPU bandwidth is measured against a constant number of peer GPUs. Those GPUs are selected in a randomized manner. Median is reported. If one GPU has a slower connection to the fabric, it will show up on all its benchmarks, but it should have minimal impact on measurement of other GPUs. This is designed to replicate data from typical pairwise benchmarks, but in linear runtime. 
7. `rack-to-rack`. This suite focuses on benchmarking rack-to-rack communication, pushing as many simultaneous copies as possible. 
Build it with 

`cmake . && make -j`

Run it with MPI:

`./nvloom_cli`

To run a single suite of tests:

`./nvloom_cli -s SUITE_NAME`

To run a single testcases

`./nvloom_cli -t TESTCASE_NAME`

You can get a list of current testcases with

`./nvloom_cli -l`

# How to benchmark MNNVL domains

For large MNNVL domains, measuring bandwidth between each pair of GPUs can be lengthy. While this project is meant to help with evaluating large MNNVL domains, some testcases are pairwise, executing in O(n^2) time. Such testcases are included to for compatibility with all of `nvbandwidth` testcases, but might not be optimal in run time and could be skipped.

To evaluate a large MNNVL domain, the recommendation is to:

1. Run `gpu-to-rack` suite. For each GPU bandwidth is measured against a constant number of peer GPUs. Those GPUs are selected in a randomized manner. Median is reported. If one GPU has slower connection to the fabric, it will show up on all its benchmarks, but it should have minimal impact on measurement of other GPUs. This is designed to replicate data from typical pairwise benchmarks, but in linear runtime. 
2. Run `fabric-stress` suite. This suite focuses on launching many copies simultaneously to saturate the fabric. Each GPU in the MNNVL domain is executing a copy at the same time.
3. Run `rack-to-rack` suite. This suite focuses on benchmarking rack-to-rack communication, pushing as many simultaneous copies as possible. 

To run those suites:

```
nvloom_cli -s gpu-to-rack rack-to-rack fabric-stress
```

Both of those suites have variants of SM/CE, write/read, etc.

More scalable suites will be added in the future. 

# How to quickly test MNNVL fabric

To quickly test MNNVL functionality, running `bisect_write_sm` is recommended. Every single GPU will write to a remote GPU at the same time, running in roughly O(1) time. This test completes in around 3 seconds on a 72-GPU MNNVL domain, including application startup. 

```
nvloom_cli -t bisect_device_to_device_write_sm
```
