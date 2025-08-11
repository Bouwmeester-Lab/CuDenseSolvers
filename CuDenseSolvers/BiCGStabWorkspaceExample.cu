/**
 * @file BiCGStabWorkspaceExample.cu
 * @brief Example demonstrating usage of BiCGStabWorkspace for persistent kernel implementation
 * 
 * This example shows how the BiCGStabWorkspace can be integrated with the existing
 * BiCGStab solver for improved memory management and performance.
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Solvers/BiCGStabWorkspace.cuh"
#include "Operators/LinearOperator.h"
#include <cublas_v2.h>

/**
 * @brief Example kernel showing how BiCGStabWorkspace can be used in device code
 * 
 * This demonstrates the workspace being used in a persistent kernel context,
 * where multiple threads can work on different parts of the vectors simultaneously.
 */
__global__ void exampleBiCGStabKernel(double* rawWorkspace, int n, const double* b, double* x) {
    // Each thread block gets its own workspace view
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize workspace from raw memory
    BiCGStabWorkspace ws(rawWorkspace, n);
    
    if (!ws.isValid() || tid >= n) return;
    
    // Example: Initialize residual vector r = b - Ax (simplified)
    // In real implementation, this would involve proper matrix-vector multiplication
    if (tid < n) {
        ws.r[tid] = b[tid]; // Simplified initialization
    }
    
    __syncthreads();
    
    // Example: Access scalars using named indices
    if (threadIdx.x == 0) {
        ws.scalars[BiCGStabWorkspace::RHO_NEW] = 1.0;
        ws.scalars[BiCGStabWorkspace::ALPHA] = 0.0;
        ws.scalars[BiCGStabWorkspace::OMEGA] = 1.0;
    }
    
    __syncthreads();
    
    // Example: Vector operations can be performed by multiple threads
    if (tid < n) {
        // p = r + beta * (p - omega * v)
        double beta = ws.scalars[BiCGStabWorkspace::BETA];
        double omega = ws.scalars[BiCGStabWorkspace::OMEGA];
        
        ws.p[tid] = ws.r[tid] + beta * (ws.p[tid] - omega * ws.v[tid]);
    }
}

/**
 * @brief Host function demonstrating workspace allocation and usage
 */
void exampleHostUsage() {
    const int n = 1000; // System size
    
    // Allocate workspace using helper function
    auto [rawWorkspace, workspace] = allocateBiCGStabWorkspace(n);
    
    if (!workspace.isValid()) {
        printf("Failed to allocate workspace\n");
        return;
    }
    
    printf("Workspace allocated successfully:\n");
    printf("- System size: %d\n", workspace.n);
    printf("- Memory required: %zu bytes\n", BiCGStabWorkspace::getRequiredSize(n));
    printf("- Vector stride: %d doubles\n", workspace.getVectorStride());
    
    // Initialize workspace to zero
    workspace.zero();
    
    // Example: Set some scalar values from host
    double hostScalars[BiCGStabWorkspace::NUM_SCALARS] = {
        1.0, // RHO_NEW
        0.0, // ALPHA
        1.0, // OMEGA
        0.0, // BETA
        0.0, // TEMP_DOT
        0.0, // TS
        0.0  // TT
    };
    
    cudaMemcpy(workspace.scalars, hostScalars, 
               BiCGStabWorkspace::NUM_SCALARS * sizeof(double), 
               cudaMemcpyHostToDevice);
    
    // The workspace can now be used with CUBLAS operations
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Example: Set r0 = r (copy operation)
    cublasDcopy(handle, n, workspace.r, 1, workspace.r0, 1);
    
    // Example: Compute dot product rho = r0 . r
    double rho;
    cublasDdot(handle, n, workspace.r0, 1, workspace.r, 1, &rho);
    
    // Store result in workspace
    cudaMemcpy(&workspace.scalars[BiCGStabWorkspace::RHO_NEW], &rho, 
               sizeof(double), cudaMemcpyHostToDevice);
    
    printf("Example operations completed successfully\n");
    
    // Clean up
    cublasDestroy(handle);
    freeBiCGStabWorkspace(rawWorkspace);
}

/**
 * @brief Performance comparison: old vs new memory layout
 * 
 * This function demonstrates the performance benefits of contiguous memory layout
 * compared to separate allocations.
 */
void performanceComparison() {
    const int n = 100000;
    
    // Method 1: Separate allocations (current BiCGStab approach)
    double *r1, *r0_1, *p1, *v1, *s1, *t1;
    cudaMalloc(&r1, n * sizeof(double));
    cudaMalloc(&r0_1, n * sizeof(double));
    cudaMalloc(&p1, n * sizeof(double));
    cudaMalloc(&v1, n * sizeof(double));
    cudaMalloc(&s1, n * sizeof(double));
    cudaMalloc(&t1, n * sizeof(double));
    
    // Method 2: Workspace (new approach)
    auto [rawWorkspace, workspace] = allocateBiCGStabWorkspace(n);
    
    printf("Memory Layout Comparison:\n");
    printf("Method 1 (separate): 6 allocations, potentially fragmented\n");
    printf("Method 2 (workspace): 1 allocation, guaranteed contiguous\n");
    printf("Workspace vectors span: %p to %p\n", 
           workspace.r, workspace.t + n - 1);
    printf("Memory span: %ld bytes\n", 
           (char*)(workspace.t + n) - (char*)workspace.r);
    
    // Clean up
    cudaFree(r1); cudaFree(r0_1); cudaFree(p1);
    cudaFree(v1); cudaFree(s1); cudaFree(t1);
    freeBiCGStabWorkspace(rawWorkspace);
}

/**
 * @brief Integration example with existing LinearOperator interface
 */
template<typename LinearOp>
void integratedSolverExample(const LinearOp& A, const double* b, double* x) {
    int n = A.size();
    
    // Allocate workspace
    auto [rawWorkspace, ws] = allocateBiCGStabWorkspace(n);
    if (!ws.isValid()) return;
    
    // Initialize workspace
    ws.zero();
    
    // Copy b to device and initialize residual
    cudaMemcpy(ws.r, b, n * sizeof(double), cudaMemcpyHostToDevice);
    
    // Example BiCGStab iteration using workspace
    // (This is a simplified version - real implementation would have full algorithm)
    
    // r0 = r
    cudaMemcpy(ws.r0, ws.r, n * sizeof(double), cudaMemcpyDeviceToDevice);
    
    // v = A * p (using existing LinearOperator interface)
    A.apply(ws.p, ws.v);
    
    // The workspace provides clean, organized access to all vectors
    // and integrates seamlessly with existing CUBLAS and LinearOperator code
    
    printf("Integration example completed for system size %d\n", n);
    
    freeBiCGStabWorkspace(rawWorkspace);
}