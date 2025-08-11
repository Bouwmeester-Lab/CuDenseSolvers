#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstddef>
#include <utility>

/**
 * @brief Workspace structure for BiCGStab solver providing organized memory layout
 * 
 * This struct organizes all temporary vectors and scalars required by the BiCGStab
 * iterative solver in a single contiguous memory allocation for optimal cache 
 * performance and easy memory management.
 * 
 * Memory Layout:
 * +------------------+------------------+------------------+------------------+
 * |  Vectors (6*n)   |  Scalars (7)     |  Output (2)      |   Padding        |
 * +------------------+------------------+------------------+------------------+
 * | r,r0,p,v,s,t     | rho,alpha,omega, | iterations,      |  For alignment   |
 * | (n doubles each) | beta,temp,ts,tt  | residual_norm    |                  |
 * +------------------+------------------+------------------+------------------+
 * 
 * All vectors are contiguous and aligned for optimal GPU memory access.
 */
struct BiCGStabWorkspace {
    // Vector pointers - 6 vectors of size n
    double* r;       ///< Residual vector
    double* r0;      ///< Fixed reference vector
    double* p;       ///< Search direction vector
    double* v;       ///< Matrix-vector product A*p
    double* s;       ///< Intermediate residual vector
    double* t;       ///< Matrix-vector product A*s
    
    // Scalar storage with named indices
    double* scalars; ///< Array of scalar values
    
    // Output storage
    int* iterations;        ///< Number of iterations performed
    double* residual_norm;  ///< Final residual norm
    
    // Problem size
    int n;           ///< Size of the linear system
    
    // Named indices for scalar array access
    static constexpr int RHO_NEW = 0;    ///< Index for rho_new scalar
    static constexpr int ALPHA = 1;      ///< Index for alpha scalar
    static constexpr int OMEGA = 2;      ///< Index for omega scalar
    static constexpr int BETA = 3;       ///< Index for beta scalar
    static constexpr int TEMP_DOT = 4;   ///< Index for temporary dot product
    static constexpr int TS = 5;         ///< Index for ts = s.t
    static constexpr int TT = 6;         ///< Index for tt = t.t
    static constexpr int NUM_SCALARS = 7; ///< Total number of scalars
    
    /**
     * @brief Constructor that initializes workspace from raw memory
     * @param rawMemory Pointer to pre-allocated contiguous memory block
     * @param size Size of the linear system (number of unknowns)
     * 
     * The raw memory must be at least getRequiredSize(size) bytes.
     * Memory layout is organized for optimal cache performance.
     */
    __host__ __device__ 
    BiCGStabWorkspace(double* rawMemory, int size) : n(size) {
        if (rawMemory == nullptr || size <= 0) {
            // Handle error case - in device code we can't throw exceptions
            n = 0;
            r = r0 = p = v = s = t = nullptr;
            scalars = nullptr;
            iterations = nullptr;
            residual_norm = nullptr;
            return;
        }
        
        // Calculate aligned offsets for each section
        double* current = rawMemory;
        
        // Vector storage - 6 vectors of size n, each aligned
        r = current;  current += size;
        r0 = current; current += size;
        p = current;  current += size;
        v = current;  current += size;
        s = current;  current += size;
        t = current;  current += size;
        
        // Scalar storage - aligned to double boundary
        scalars = current;
        current += NUM_SCALARS;
        
        // Output storage - aligned properly
        // Align to int boundary for iterations
        size_t offset = reinterpret_cast<size_t>(current) % sizeof(int);
        if (offset != 0) {
            current = reinterpret_cast<double*>(
                reinterpret_cast<char*>(current) + (sizeof(int) - offset)
            );
        }
        iterations = reinterpret_cast<int*>(current);
        current = reinterpret_cast<double*>(
            reinterpret_cast<int*>(current) + 1
        );
        
        // Residual norm storage
        residual_norm = current;
    }
    
    /**
     * @brief Calculate the total memory size required for the workspace
     * @param size Size of the linear system
     * @return Number of bytes required for the workspace
     */
    __host__ __device__ 
    static size_t getRequiredSize(int size) {
        if (size <= 0) return 0;
        
        // 6 vectors of size 'size' doubles each
        size_t vectorBytes = 6 * size * sizeof(double);
        
        // Scalar storage
        size_t scalarBytes = NUM_SCALARS * sizeof(double);
        
        // Output storage (int + double with potential padding)
        size_t outputBytes = sizeof(int) + sizeof(double) + sizeof(int); // padding
        
        // Total with some padding for alignment
        return vectorBytes + scalarBytes + outputBytes + 64; // 64 bytes extra for alignment
    }
    
    /**
     * @brief Validate that the workspace is properly initialized
     * @return true if workspace is valid, false otherwise
     */
    __host__ __device__ 
    bool isValid() const {
        return (n > 0 && r != nullptr && r0 != nullptr && p != nullptr && 
                v != nullptr && s != nullptr && t != nullptr && 
                scalars != nullptr && iterations != nullptr && 
                residual_norm != nullptr);
    }
    
    /**
     * @brief Get pointer to specific vector by index
     * @param vectorIndex Index of vector (0=r, 1=r0, 2=p, 3=v, 4=s, 5=t)
     * @return Pointer to the requested vector, or nullptr if invalid index
     */
    __host__ __device__ 
    double* getVector(int vectorIndex) const {
        switch (vectorIndex) {
            case 0: return r;
            case 1: return r0;
            case 2: return p;
            case 3: return v;
            case 4: return s;
            case 5: return t;
            default: return nullptr;
        }
    }
    
    /**
     * @brief Initialize all vectors and scalars to zero
     */
    __host__ 
    void zero() const {
        if (!isValid()) return;
        
        // Zero out all vectors
        cudaMemset(r, 0, n * sizeof(double));
        cudaMemset(r0, 0, n * sizeof(double));
        cudaMemset(p, 0, n * sizeof(double));
        cudaMemset(v, 0, n * sizeof(double));
        cudaMemset(s, 0, n * sizeof(double));
        cudaMemset(t, 0, n * sizeof(double));
        
        // Zero out scalars
        cudaMemset(scalars, 0, NUM_SCALARS * sizeof(double));
        
        // Initialize output
        int zero_int = 0;
        double zero_double = 0.0;
        cudaMemcpy(iterations, &zero_int, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(residual_norm, &zero_double, sizeof(double), cudaMemcpyHostToDevice);
    }
    
    /**
     * @brief Get the memory offset between consecutive vectors
     * @return Stride in doubles between vector starts
     */
    __host__ __device__ 
    int getVectorStride() const {
        return n;
    }
    
    /**
     * @brief Get the base address of the vector storage area
     * @return Pointer to the start of vector memory
     */
    __host__ __device__ 
    double* getVectorBase() const {
        return r;
    }
    
    /**
     * @brief Get the total size of vector storage in bytes
     * @return Size in bytes of all vector storage
     */
    __host__ __device__ 
    size_t getVectorStorageSize() const {
        return 6 * n * sizeof(double);
    }
    
    /**
     * @brief Check if two workspaces share the same memory layout
     * @param other Another workspace to compare with
     * @return true if workspaces have compatible layout
     */
    __host__ __device__ 
    bool isCompatibleWith(const BiCGStabWorkspace& other) const {
        return (n == other.n && isValid() && other.isValid());
    }
};

/**
 * @brief Helper function to allocate workspace on device
 * @param size Size of the linear system
 * @return Pair of raw memory pointer and workspace object, or {nullptr, invalid workspace} on failure
 */
__host__ 
inline std::pair<double*, BiCGStabWorkspace> allocateBiCGStabWorkspace(int size) {
    if (size <= 0) {
        return {nullptr, BiCGStabWorkspace(nullptr, 0)};
    }
    
    size_t requiredBytes = BiCGStabWorkspace::getRequiredSize(size);
    double* rawMemory = nullptr;
    
    cudaError_t err = cudaMalloc(&rawMemory, requiredBytes);
    if (err != cudaSuccess || rawMemory == nullptr) {
        return {nullptr, BiCGStabWorkspace(nullptr, 0)};
    }
    
    BiCGStabWorkspace workspace(rawMemory, size);
    if (!workspace.isValid()) {
        cudaFree(rawMemory);
        return {nullptr, BiCGStabWorkspace(nullptr, 0)};
    }
    
    return {rawMemory, workspace};
}

/**
 * @brief Helper function to free workspace allocated by allocateBiCGStabWorkspace
 * @param rawMemory Pointer returned by allocateBiCGStabWorkspace
 */
__host__ 
inline void freeBiCGStabWorkspace(double* rawMemory) {
    if (rawMemory != nullptr) {
        cudaFree(rawMemory);
    }
}