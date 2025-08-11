#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define WARP_SIZE 32
#include "Constants.cuh"
#include "Solvers/BiCGStabWorkspace.cuh"
#ifdef __INTELLISENSE__ // for Visual Studio IntelliSense https://stackoverflow.com/questions/77769389/intellisense-in-visual-studio-cannot-find-cuda-cooperative-groups-namespace
#define __CUDACC__
#endif // __INTELLISENSE__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#ifdef __INTELLISENSE__
#undef __CUDACC__
#endif // __INTELLISENSE__

namespace cg = cooperative_groups;

namespace CuDenseSolvers
{
	struct PairPlus {
		__device__ double2 operator()(double2 a, double2 b) const {
			return make_double2(a.x + b.x, a.y + b.y);
		}
	};
	template<int BS, class A>
	/// <summary>
	/// Applies the operator A to the vector x, stores the result in y, and computes two dot products with vectors w1 and with y
	/// The dot products are stored in out_dot1 and out_dot2 respectively.
	/// </summary>	
	__device__ inline void grid_apply_and_two_dots(
		cg::grid_group grid, int n, const A& Aop,
		const double* __restrict__ x, double* __restrict__ y,
		const double* __restrict__ w1,
		double* __restrict__ out_dot1, double* __restrict__ out_dot2)
	{
		auto block = cg::this_thread_block();
		int  tid = blockIdx.x * BS + threadIdx.x;
		int  stride = gridDim.x * BS;

		double2 local = make_double2(0.0, 0.0);
		for (int i = tid; i < n; i += stride) {
			double yi = Aop(i, x);
			y[i] = yi;
			if (w1) local.x += w1[i] * yi; // e.g., (s,t)
			if (out_dot2) local.y += yi * yi;    // e.g., (t,t)
		}
		__syncthreads();
		double2 block_sum = cg::reduce(block, local, PairPlus{});
		if (block.thread_rank() == 0) {
			if (out_dot1) atomicAdd(out_dot1, block_sum.x);
			if (out_dot2) atomicAdd(out_dot2, block_sum.y);
		}
		grid.sync();
	}

	/// <summary>
	/// Calculate the residual r = b - A*x and optionally the norm squared of r.
	/// </summary>
	template<int BLOCK_SIZE, class A>
	__device__ inline void grid_residual_and_norm(
		cg::grid_group grid,
		int n,
		const A& Aop,
		const double* __restrict__ x,
		const double* __restrict__ b,
		double* __restrict__ r,
		double* __restrict__ out_norm2, // may be nullptr, make sure it's initialized to 0.0 before calling
		double* __restrict__ out_r0 = nullptr // may be nullptr
	) {
		auto block = cg::this_thread_block();

		const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		const int stride = gridDim.x * BLOCK_SIZE;

		double local = 0.0;

		for (int i = tid; i < n; i += stride) {
			double yi = Aop(i, x);          // generic row action
			double ri = b[i] - yi;
			r[i] = ri;
			if (out_norm2) local += ri * ri; // fuse norm squared
			if (out_r0) out_r0[i] = ri; // store initial residual if requested
		}

		// Block sync sufficient - no inter-block dependencies in computation
		__syncthreads();

		if (out_norm2) {

			// block-wide reduction using CG
			double block_sum = cg::reduce(block, local, cg::plus<double>());
			if (block.thread_rank() == 0) atomicAdd(out_norm2, block_sum);
		}

		// sync before consumers read out_norm2 or r
		grid.sync();
	}

	/// <summary>
	/// Calculate the residual yout = y + alpha*x  and optionally the norm squared of y.
	/// </summary>
	template<int BLOCK_SIZE>
	__device__ inline void grid_axpy_and_norm(
		cg::grid_group grid,
		int n,
		const double* alpha,
		const double* __restrict__ x,
		const double* __restrict__ y,

		double* __restrict__ yout,
		double* __restrict__ out_norm2 // may be nullptr, make sure it's initialized to 0.0 before calling
	) {
		auto block = cg::this_thread_block();

		const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		const int stride = gridDim.x * BLOCK_SIZE;

		double local = 0.0;

		for (int i = tid; i < n; i += stride) {
			yout[i] = y[i] + (*alpha) * x[i]; // y += alpha * x
			if (out_norm2) local += yout[i] * yout[i]; // fuse norm squared
		}

		
		

		if (out_norm2) {
			// Block sync sufficient - no inter-block dependencies in computation
			__syncthreads();
			// block-wide reduction using CG
			double block_sum = cg::reduce(block, local, cg::plus<double>());
			if (block.thread_rank() == 0) atomicAdd(out_norm2, block_sum);
		}

		// sync before consumers read out_norm2 or r
		grid.sync();
	}

	template<int BLOCK_SIZE, class A>
	__device__ inline void grid_apply_and_dot(
		cg::grid_group grid,
		int n,
		const A& Aop,
		const double* __restrict__ x,   // input vector (p or s)
		double* __restrict__ y,         // output vector (v or t)
		const double* __restrict__ w,   // dot partner (e.g. r0 or s); may be nullptr
		double* __restrict__ out_dot    // global scalar; may be nullptr (caller zeroes)
	) {
		auto block = cg::this_thread_block();
		int  tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		int  stride = gridDim.x * BLOCK_SIZE;

		double local = 0.0;
		for (int i = tid; i < n; i += stride) {
			double yi = Aop(i, x);  // (A x)_i
			y[i] = yi;
			if (w && out_dot) local += w[i] * yi;
		}

		if (w && out_dot) {
			double block_sum = cg::reduce(block, local, cg::plus<double>());
			if (block.thread_rank() == 0) atomicAdd(out_dot, block_sum);
		}

		grid.sync();
	}

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
		BiCGStabWorkspace ws, // must be initialized to zero before calling this kernel
		OperatorA A,
		int maxIter,
		double tol,
		double* residualOut,
		int* iterOut)
	{
		// Kernel implementation for the BiCGStab algorithm
		// This will include the logic for the BiCGStab iterations
		// and will use the provided operator A to apply the matrix-vector product.

		// Note: The actual implementation of the BiCGStab algorithm is complex
		// and requires careful handling of vector operations, convergence checks, etc.

		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		if (tid >= N) return; // return if thread index exceeds vector size


		// step 1: calculate the initial residual r = b - A*x for this row

		double xi = x[tid]; // current value of x for this row
		double ri = b[tid]; // current value of b for this row

		double rho_old = 1.0, alpha = 1.0, omega = 1.0;
		double rho_new = 0.0, beta = 0.0;

		// Initialize vectors to zero
		ws.p[tid] = 0.0; // p vector
		ws.v[tid] = 0.0; // v vector
		// no need to sync here since the next line will compute the initial residual and that will sync the whole grid
		auto grid = cg::this_grid(); // get the grid group
		grid_residual_and_norm<BLOCK_SIZE, OperatorA>(grid, N, A, x, b, ws.r, &ws.scalars[ws.RHO_NEW], ws.r0); // compute r = b - A*x and store initial residual in r0

		

		// let's loop
		for (int k = 0; k < maxIter; ++k)
		{
			if (rho_new == 0.0) break; // convergence check

			beta = (rho_new / rho_old) * (alpha / omega);

			// calculate p = r + beta * (p - omega * v)
			ws.p[tid] = ws.r[tid] + beta * (ws.p[tid] - omega * ws.v[tid]);

			// compute v = A * p for this row
			grid.sync(); // ensure all threads have updated p
			
			grid_apply_and_dot<BLOCK_SIZE, OperatorA>(grid, N, A, ws.p, ws.v, ws.r0, &ws.scalars[ws.TEMP_DOT]); // v = A * p, dot product with r0

			alpha = ws.scalars[ws.RHO_NEW] / ws.scalars[ws.TEMP_DOT]; // compute alpha
			double negAlpha = -alpha; // precompute -alpha

			// compute s = r - alpha * v for this row and the norm squared of s
			grid_axpy_and_norm<BLOCK_SIZE>(grid, N, &negAlpha, ws.v, ws.r, ws.s, &ws.scalars[ws.residual_temp]);

			if(ws.scalars[ws.residual_temp] < tol * tol) 
			{   // check convergence on s
				// if the norm of s is small enough, we can finish
				// compute the final x update, x = x + alpha * p;
				if (blockIdx.x == 0 && threadIdx.x == 0) 
				{
					*ws.residual_norm = ws.scalars[ws.residual_temp]; // compute the final residual norm
					if (residualOut) *residualOut = *ws.residual_norm; // store the final residual norm if requested
					if (iterOut) *iterOut = k + 1; // store the number of iterations
				}
				grid_axpy_and_norm<BLOCK_SIZE>(grid, N, &alpha, ws.p, x, x, nullptr); // x += alpha * p
				
				break;
			}

			// compute t = A * s for this row and s . t and t. t at the same time

			grid_apply_and_two_dots<BLOCK_SIZE, OperatorA>(grid, N, A, ws.s, ws.t, ws.s, &ws.scalars[ws.TS], &ws.scalars[ws.TT]);

			omega = ws.scalars[ws.TS] / ws.scalars[ws.TT]; // compute omega

			// update x = x + alpha * p + omega * s
			grid_axpy_and_norm<BLOCK_SIZE>(grid, N, &alpha, ws.p, x, x, nullptr); // x += alpha * p
			grid_axpy_and_norm<BLOCK_SIZE>(grid, N, &omega, ws.s, x, x, nullptr); // x += omega * s



		}
	}

	
}