# BiCGStab Workspace Implementation

## Overview

The `BiCGStabWorkspace` provides a structured, efficient memory management solution for the BiCGStab iterative solver. It organizes all temporary vectors and scalars in a single contiguous memory allocation, optimizing cache performance and simplifying memory management for persistent kernel implementations.

## Key Features

### 🚀 Performance Optimizations
- **Contiguous Memory Layout**: All 6 vectors (r, r0, p, v, s, t) are allocated in a single block
- **Cache-Friendly Access**: Sequential memory layout improves cache hit rates
- **GPU Memory Alignment**: Optimized for GPU memory access patterns
- **Reduced Allocation Overhead**: Single `cudaMalloc` instead of 6 separate allocations

### 🔧 Developer-Friendly Interface
- **Type-Safe Access**: Named constants for scalar indices (`RHO_NEW`, `ALPHA`, etc.)
- **Host/Device Compatible**: Works in both `__host__` and `__device__` code
- **Easy Integration**: Compatible with existing `LinearOperator<double>` interface
- **Comprehensive Validation**: Built-in validity checks and error handling

### 📝 Memory Layout

```
+------------------+------------------+------------------+------------------+
|  Vectors (6*n)   |  Scalars (7)     |  Output (2)      |   Padding        |
+------------------+------------------+------------------+------------------+
| r,r0,p,v,s,t     | rho,alpha,omega, | iterations,      |  For alignment   |
| (n doubles each) | beta,temp,ts,tt  | residual_norm    |                  |
+------------------+------------------+------------------+------------------+
```

## Usage Examples

### Basic Allocation and Usage

```cpp
#include "Solvers/BiCGStabWorkspace.cuh"

// Allocate workspace for system of size n
int n = 1000;
auto [rawMemory, workspace] = allocateBiCGStabWorkspace(n);

if (workspace.isValid()) {
    // Initialize to zero
    workspace.zero();
    
    // Access vectors directly
    workspace.r[i] = b[i];
    workspace.r0[i] = workspace.r[i];
    
    // Access scalars using named indices
    workspace.scalars[BiCGStabWorkspace::RHO_NEW] = rho_value;
    workspace.scalars[BiCGStabWorkspace::ALPHA] = alpha_value;
    
    // Clean up
    freeBiCGStabWorkspace(rawMemory);
}
```

### Integration with CUBLAS

```cpp
// Works seamlessly with existing CUBLAS operations
cublasHandle_t handle;
cublasCreate(&handle);

// Copy r to r0
cublasDcopy(handle, n, workspace.r, 1, workspace.r0, 1);

// Compute dot product: rho = r0 . r
double rho;
cublasDdot(handle, n, workspace.r0, 1, workspace.r, 1, &rho);

// Store in workspace
workspace.scalars[BiCGStabWorkspace::RHO_NEW] = rho;
```

### Device Kernel Usage

```cpp
__global__ void biCGStabKernel(double* rawWorkspace, int n) {
    BiCGStabWorkspace ws(rawWorkspace, n);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ws.isValid() && tid < n) {
        // Each thread can work on vector elements
        ws.p[tid] = ws.r[tid] + beta * (ws.p[tid] - omega * ws.v[tid]);
    }
    
    // Thread 0 can handle scalar operations
    if (threadIdx.x == 0) {
        ws.scalars[BiCGStabWorkspace::BETA] = new_beta;
    }
}
```

## API Reference

### Constructor
```cpp
BiCGStabWorkspace(double* rawMemory, int size)
```
Initializes workspace from pre-allocated memory block.

### Static Methods
```cpp
static size_t getRequiredSize(int size)  // Calculate memory requirements
```

### Vector Access
```cpp
double* r, *r0, *p, *v, *s, *t;          // Direct access
double* getVector(int index);            // Access by index (0-5)
int getVectorStride();                   // Distance between vectors
```

### Scalar Access
```cpp
double* scalars;                         // Scalar array
static constexpr int RHO_NEW = 0;        // Named indices
static constexpr int ALPHA = 1;
static constexpr int OMEGA = 2;
static constexpr int BETA = 3;
static constexpr int TEMP_DOT = 4;
static constexpr int TS = 5;
static constexpr int TT = 6;
```

### Utility Methods
```cpp
bool isValid();                          // Validate workspace
void zero();                             // Initialize to zero
```

### Helper Functions
```cpp
std::pair<double*, BiCGStabWorkspace> allocateBiCGStabWorkspace(int size);
void freeBiCGStabWorkspace(double* rawMemory);
```

## Performance Benefits

### Before (Separate Allocations)
```cpp
// Current BiCGStab approach
cudaMalloc(&r, n * sizeof(double));   // 6 separate allocations
cudaMalloc(&r0, n * sizeof(double));  // Potentially fragmented memory
cudaMalloc(&p, n * sizeof(double));   // Poor cache locality
// ... (3 more allocations)
```

### After (Workspace)
```cpp
// New workspace approach
auto [raw, ws] = allocateBiCGStabWorkspace(n);  // Single allocation
// Guaranteed contiguous memory layout
// Optimal cache performance
```

## Integration with Existing Code

The workspace is designed to be a drop-in replacement for existing memory management:

1. **Minimal Code Changes**: Existing CUBLAS calls work unchanged
2. **Compatible Interface**: Works with current `LinearOperator<double>` 
3. **Incremental Adoption**: Can be adopted gradually without breaking changes
4. **Performance Transparent**: Same computational complexity, better memory access

## Testing

Comprehensive test suite validates:
- ✅ Memory layout correctness
- ✅ Contiguous vector arrangement  
- ✅ Proper scalar indexing
- ✅ Edge case handling
- ✅ Device/host compatibility
- ✅ Integration with existing interfaces

Run tests with:
```bash
# Compile and run test suite
nvcc -I./include CuDenseSolvers.Tests/kernel.cu -lgtest -o test_runner
./test_runner
```

## Future Enhancements

- [ ] Support for multiple precision types (float, complex)
- [ ] Persistent kernel implementation using this workspace
- [ ] Memory pool allocation for multiple solvers
- [ ] Performance benchmarking suite
- [ ] Integration with other iterative solvers (CG, GMRES)

## See Also

- [BiCGStabWorkspaceExample.cu](BiCGStabWorkspaceExample.cu) - Comprehensive usage examples
- [BiCGStab.cuh](include/Solvers/BiCGStab.cuh) - Original solver implementation
- [SolverTests.cuh](../CuDenseSolvers.Tests/SolverTests.cuh) - Test suite