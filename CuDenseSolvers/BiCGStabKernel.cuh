#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define WARP_SIZE 32
#include "Constants.cuh"

namespace CuDenseSolvers
{
	struct global_variables {
		double* global_dot_r0_r;
		double* global_dot_r0_v;
		double* global_norm_s2;
		double* global_dot_st;
		double* global_dot_tt;
		double* global_norm_r2;
	};

	/// <summary>
	/// Each thread in the block computes a row of the BiCGStab algorithm.
	/// </summary>
	template <typename OperatorA>
	__global__ void BiCGStabKernel(
		double* x,
		const double* b,
		int N,
		double* r,
		double* r0,
		double* p,
		double* v,
		double* s,
		double* t,
		OperatorA A,
		int maxIter,
		double tol,
		double* residualOut,
		int* iterOut,
		global_variables gv)
	{
		// Kernel implementation for the BiCGStab algorithm
		// This will include the logic for the BiCGStab iterations
		// and will use the provided operator A to apply the matrix-vector product.

		// Note: The actual implementation of the BiCGStab algorithm is complex
		// and requires careful handling of vector operations, convergence checks, etc.

		int tid = threadIdx.x + blockIdx.x * blockDim.x; // the row index for this thread

		if (tid >= N) return; // return if thread index exceeds vector size


		// step 1: calculate the initial residual r = b - A*x for this row

		double xi = x[tid]; // current value of x for this row
		double ri = b[tid]; // current value of b for this row
		double Ax = A(tid, x); // apply the operator A to x for this row

		ri -= Ax; // compute the residual for this row

		r[tid] = ri; // store the residual
		r0[tid] = ri; // store the initial residual used as reference

		double rho_old = 1.0, alpha = 1.0, omega = 1.0;
		double rho_new = 0.0, beta = 0.0;

		// let's loop
		for (int k = 0; k < maxIter; ++k)
		{
			__shared__ double dot_r0_r;
			__shared__ double dot_r0_v;
			__shared__ double norm_s;
			__shared__ double dot_st, dot_tt;
			__shared__ double norm_r2;

			// compute rho_new: r_0^T * r
			double ri = r[tid];
			double r0i = r0[tid];
			double local_rho_new = ri * r0i; // local dot product for this thread

			// reduce
			reduceAtomicAdd<BLOCK_SIZE>(local_rho_new, &dot_r0_r, gv.global_dot_r0_r);

		}
	}

	template <unsigned int blockSize>
	__device__ __forceinline__ void reduceAtomicAdd(double local, double* shared_accum, double* global_accum)
	{
		local = blockReduceSum<blockSize>(local); // Reduce within the block
		if (threadIdx.x == 0) {
			atomicAdd(global_accum, local); // Atomic add to global accumulator
		}
		__syncthreads(); // Ensure all threads have completed their reduction

		if (threadIdx.x == 0) {
			// If this is the first thread, write the result to shared memory
			*shared_accum = *global_accum;
		}
		__syncthreads();
	}

	__inline__ __device__
		double warpReduceSum(double val) {
		for (int offset = warpSize / 2; offset > 0; offset /= 2)
			val += __shfl_down_sync(0xffffffff, val, offset);
		return val;
	}

	template <unsigned int blockSize>
	__inline__ __device__
		double blockReduceSum(double val) {
		__shared__ double shared[32]; // One per warp
		int lane = threadIdx.x % warpSize;
		int wid = threadIdx.x / warpSize;

		val = warpReduceSum(val); // intra-warp reduction

		if (lane == 0)
			shared[wid] = val; // write warp result to shared memory

		__syncthreads();

		// First warp reduces all warp results
		val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
		if (wid == 0) val = warpReduceSum(val);

		return val;
	}

}