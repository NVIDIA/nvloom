# Library design

## Memory allocators

The core class for memory allocations is

```
class MemoryAllocation {
public:
    void *ptr;
    size_t allocationSize;
    int MPIrank;
    static inline int uniqueIdCounter = 0;
    int uniqueId;
}
```

`void *ptr` and `size_t allocationSize` describe the allocated memory.

`int MPIrank` describes the process owning the memory.

`int uniqueIdCounter` and `int uniqueId` are used to provide a unique ID for each allocation, which is needed to uniquely initialize allocations with data.

This class is meant to be a base for inheritance of your custom allocators.

Sample implementations of the class include `MultinodeMemoryAllocationUnicast`, `MultinodeMemoryAllocationEGM`, `MultinodeMemoryAllocationMulticast`, `DeviceMemoryAllocation` and `HostMemoryAllocation`/

## Copy direction

Each copy can be either a "read" or a "write".

When copying memory from GPU A to B, the copy can be executed by either GPU.

If the copy is executed by GPU A, we call it a "write" - because GPU A is issuing writes over Nvlink.

If the copy is executed by GPU B, we call it a "read" - because GPU B is issuing reads over Nvlink.

This choice is captured by the enum `CopyDirection`

```
enum CopyDirection {
    COPY_DIRECTION_WRITE = 0,
    COPY_DIRECTION_READ,
};
```

## Copy type

A copy can be executed either by Copy Engines (CE) or Streaming Multiprocessors (SM).

To execute a copy on CEs, we call `cuMemcpyAsync`. CUDA Driver will then schedule the copy on the copy engines of a given device.

To execute a copy on SMs, we call a memcpy kernel implemented in `kernels.cu`

This choice is captured by the enum `CopyType`

```
enum CopyType {
    COPY_TYPE_CE = 0,
    COPY_TYPE_SM,
}
```

## Copy

A copy is defined by
- destination allocation (`MemoryAllocation` class)
- source allocation (`MemoryAllocation` class)
- copy direction (`CopyDirection` enum)
- copy type (`CopyType` enum)

See the constructor of the Copy class.

Allocations are passed as `std::shared_ptr` to allow for easy reuse of allocations.

```
Copy(std::shared_ptr<MemoryAllocation> _dst, std::shared_ptr<MemoryAllocation> _src, CopyDirection _copyDirection, CopyType _copyType)
```

MPI process and CUDA context executing the copy are determined based on `CopyDirection`.

## Initializing the library

The library needs to be initialized by calling `NvLoom::initialize(int _localDevice, std::map<std::string, std::vector<int> > _rackToProcessMap = {})`.

### Local device

Each process owns one GPU. Your application needs to tell the library which local GPU it owns (`cudaSetDevice` ordering).

Determining which process owns which device is a problem with multiple solutions, each with its own pros and cons. The `nvloom` library is supposed to be as flexible as possible, which is why it leaves the choice to the library user.

Example strategies for determining which process owns which GPU are:
- processes exchange hostnames through MPI and figure out the ordering based on the hostnames (implemented in the `nvloom-cli`)
- applications get the local GPU parameter through command line argument set by the scheduler

### Rack to process map

Optional argument, providing information about NVLink topology.

Each key is a rack name (most likely its GUID, but it's up to the library user).

Each value is a vector of process MPI ranks running in this rack.

## Running a benchmark

A benchmark is executed through the `doBenchmark` method

```
std::vector<double> NvLoom::doBenchmark(std::vector<Copy> copies);
```

It takes a list of copies to simultaneously execute, and returns a list of measured bandwidth for each one of those copies (in GiB/s). The returned list of bandwidth is ordered exactly the same as the list of copies passed as an argument.

To do multiple independent non-simultaneous benchmarks, one needs to call `doBenchmark` once per each benchmark.

`doBenchmark` handles scheduling and measuring bandwidths of all the copies. To schedule copies, it uses spin kernels to release all benchmarks at precisely the same time. Using spin kernels in this manner does not conform with CUDA Programming Model, but it is a valuable tool for benchmarking simultaneous bandwidths.


