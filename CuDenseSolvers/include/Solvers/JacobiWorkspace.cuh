#pragma once
#include "cuda_runtime.h"

/// <summary>
/// Holds all the data needed for the Jacobi method in a contiguous memory space on the GPU
/// </summary>
struct JacobiWorkspace
{
	/// <summary>
	/// Represents xk_th
	/// </summary>
	double* xnew;
	/// <summary>
	/// Represents xk-1_th vector
	/// </summary>
	double* xprevious;
	/// <summary>
	/// Represents the inverted diagonal.
	/// </summary>
	double* dinv;
	double* omega;
	/// <summary>
	/// Raw memory
	/// </summary>
	double* rawMemory;
	static constexpr int NUM_SCALARS = 1;
	size_t size;
	__host__ __device__ JacobiWorkspace(size_t n) : size(n)
	{
		cudaMalloc(&rawMemory, getRequiredSize(n));
		
		xprevious = rawMemory;
		xnew = rawMemory + n;
		dinv = rawMemory + 2 * n;
		omega = dinv + 1;
	}

	~JacobiWorkspace() {
		cudaFree(rawMemory);
	}

	__device__ __host__ static int getRequiredSize(size_t n)
	{
		return 3 * n * sizeof(double) + NUM_SCALARS * sizeof(double);
	}
};