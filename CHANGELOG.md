# Changelog

## [2.2.0] - 2026-10-19

### Added
- Added localized CE benchmarks
- Added CFT all-to-one/all-from-one tests
- Added support for CFT and localized CE benchmarks to the CSV-defined benchmark
- Added option to save benchmark results as JSON file. Experimental feature, subject to change.
- Added --fillPattern option to control how buffers are memset prior to benchmarking
- Added support for cuDeviceGetCliqueInfo to determine whether GPUs are in the same clique for a given operation
- Added a warning if a GPU is trying to execute multiple CE methods or multiple non-CE methods at the same time

### Changed
- CFT handles are now pooled for each CFT testcase, rather than each allocation creating its own CFT handle
- Overhauled the "fully random" fabric stress test to remove the limit of 3 peers
- Sample Dockerfile is now based on CUDA 13.4.1

### Fixed
- Fixed a bug in "fully random" fabric stress test where the test would schedule multiple CE concurrent copies per GPU

## [2.1.0] - 2026-07-23

### Added
- Added support for CUDA Compute Fabric Transport (CFT) programming model with a coverage of unicast and multicast testcases
- Added fully random and random permutation (ring, bisect) fabric stress tests
- Added TMA flavors of all-to-one/bisect/gpu-to-rack/rack-to-rack testcases
- Added --blockCount argument to override block count of copy kernels
- Added --tmaChunkSize argument to override TMA staging buffer size
- Added --richOutput support to all-to-one, rack-aware-all-to-one and multicast_all_to_all testcases
- Added support for CUDA MPS
- Testcases now output median value alongside preexisting min/average/max/sum

### Changed
- Increased default TMA staging buffer size to 196 kB

## [2.0.0] - 2026-04-01

### Added
- Load sustainment algorithm was added to testcases that execute simultaneously multiple copies
- CUDA Pool based multicast memory allocator
- A "bandwidthUnit" option to specify the unit used when reporting bandwidths was added to nvloom_cli.
- Scalable egm-to-rack testcase suite was added to nvloom_cli
- Rack-aware all-to-one testcase suite was added to nvloom_cli
- An option to plot a histogram of the results was added to `plot_heatmaps.py`
- Consistency checks of system settings (such as EGM or driver versions) between processes were added to `nvloom_cli`

### Changed
- [Breaking change] nvloom_cli "bufferSize" argument is now interpreted as bytes, and supports parsing K/M/G/T units
- Copy kernels are now launched with NVLink-centric scheduling
- Asserts were enhanced by adding information about what copies were executing when the error happened
- Asserts were enhanced by adding information about what stage the benchmark was when the error happened
- Tuning of multicast copy kernels
- When running nvloom_cli with CUDA Pool allocators, tests use the newly added CUDA Pool multicast allocator
- Default value of the "gpuToRackSamples" argument was changed from 5 to 3
- Plot_heatmaps.py now uses uv to handle its dependencies using PEP 723
- Plot_heatmaps.py prints what nodes it was run on in the graph title
- Minor performance enhancements

### Fixed
- Fixed a bug where a CSV testcase would ignore specified number of iterations
- Fixed a bug where nvloom_cli could hang when using "duration" option to repeat testcases

### Removed
- Support for CUDA pre-13.0 was removed
- Support for drivers pre-580 was removed

## [1.4.0] - 2025-12-08

### Added
- Benchmarks of multicast broadcast operations executed by Copy Engines
- Unicast and multicast TMA testcases
- Heatmap plotter now has an option to skip plotting data values

### Changed
- All-to-one testcases warn if individual results are significantly assymetric
- Column separators in gpu_to_rack testcases are only printed in "rich output mode"
- Deprecated CUDA 12.X, support will be removed in next release
- Deprecated 570 and 575 drivers, support will be removed in next release

### Fixed
- Row separator handling in heatmap plotter

## [1.3.0] - 2025-10-02

### Added
- CSV defined benchmarks
- Memory access latency benchmarks
- Sample Dockerfile to build nvloom

### Changed
- Retry mechanism for CUDA multicast allocations was removed

### Fixed
- Freeing MNNVL memory did not have enough MPI barriers, leading to race conditions in extremely rare edge-cases
- Benchmarking algorithm sometimes would record "end event" twice. This had no impact on benchmark results.

## [1.2.0] - 2025-07-21

### Added
- Multicast reductions benchmarks
- Option to specify iteration count (-i/--iterations)
- Option to repeat a testcase for a specified number of iterations (-c/--repeat)
- Option to repeat a testcase for a specified number of seconds (-d/--duration)
- CUDA Stream Ordered Memory Allocator was added as a new allocator option (-a cudapool)

### Changed
- Caching multicast allocations is now much faster, thanks to multicast-specific memory pool

## [1.1.0] - 2025-05-22

### Added

- Heatmap plotter
- Support for CUDA Error Log Management
- Retry mechanism for CUDA multicast allocations
- Nvloom_cli argument to set number of samples in gpu-to-rack testcases
- Nvloom_cli now prints its version, git commit it was built from, and specified buffer size
- Nvloom_cli now prints units when reporting results
- Native compilation for sm_103 on CUDA 12.9 toolkits

### Changed

- Expanded README.md
- Rack-to-rack are now both unidir and bidir, and bidir rack-to-rack are symmetry-optimized.

### Fixed

- Bug where requesting allocations over 4 GiB would fail with CUDA_OUT_OF_MEMORY

## [1.0.0] - 2025-03-17

### Added

- Initial release
