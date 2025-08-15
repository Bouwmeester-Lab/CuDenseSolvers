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
#include "Common.cuh"

namespace cg = cooperative_groups;

namespace CuDenseSolvers
{
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
		double tol)
	{
		// Kernel implementation for the BiCGStab algorithm
		// This will include the logic for the BiCGStab iterations
		// and will use the provided operator A to apply the matrix-vector product.

		// Note: The actual implementation of the BiCGStab algorithm is complex
		// and requires careful handling of vector operations, convergence checks, etc.

		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		//printf("BiCGStabKernel launched with blockIdx.x = %d, blockDim.x = %d, threadIdx.x = %d, tid = %d\n", blockIdx.x, blockDim.x, threadIdx.x, tid);
		//printf("N = %d, maxIter = %d, tol = %f\n", N, maxIter, tol);
		if (tid >= N) return; // return if thread index exceeds vector size

		//printf("Thread %d is processing row %d\n", tid, tid);
		// step 1: calculate the initial residual r = b - A*x for this row

		//double xi = x[tid]; // current value of x for this row
		//double ri = b[tid]; // current value of b for this row

		double rho_old = 1.0, alpha = 1.0, omega = 1.0;
		double rho_new = 0.0, beta = 0.0;

		// Initialize vectors to zero
		ws.p[tid] = 0.0; // p vector
		ws.v[tid] = 0.0; // v vector
		// no need to sync here since the next line will compute the initial residual and that will sync the whole grid
		auto grid = cg::this_grid(); // get the grid group
#if DEBUG_PRINT
		printf("Thread %d initialized p and v to zero\n", tid);
		printf("Workspace points to %d", ws);
#endif

		grid_residual_and_norm<BLOCK_SIZE, OperatorA>(grid, N, A, x, b, ws.r, &ws.scalars[ws.RHO_NEW], ws.r0); // compute r = b - A*x and store initial residual in r0

		rho_new = ws.scalars[ws.RHO_NEW];

		// let's loop
#if DEBUG_PRINT
		printf("%d: Starting BiCGStab iterations with maxIter = %d and tol = %f\n", tid, maxIter, tol);
		printf("%d: Initial rho_new = %f\n", tid, rho_new);
#endif

		for (int k = 0; k < maxIter; ++k)
		{
			// update local variables
#if DEBUG_PRINT
			printf("%d: Iteration %d - start\n", tid, k);
#endif
			rho_new = ws.scalars[ws.RHO_NEW]; // get the new rho from the scalars
			rho_old = ws.scalars[ws.RHO_NEW]; // get the old rho from the scalars

			if (rho_new == 0.0) break; // convergence check

			beta = (rho_new / rho_old) * (alpha / omega);
#if DEBUG_PRINT
			printf("%d: Iteration %d: rho_new = %f, rho_old = %f, alpha = %f, omega = %f, beta = %f\n", tid, k, rho_new, rho_old, alpha, omega, beta);
#endif
			// calculate p = r + beta * (p - omega * v)
			ws.p[tid] = ws.r[tid] + beta * (ws.p[tid] - omega * ws.v[tid]);

			// compute v = A * p for this row
			grid.sync(); // ensure all threads have updated p
			
#if DEBUG_PRINT
			printf("%d: Calling grid_apply_and_dot for v = A * p\n", tid);
#endif
			grid_apply_and_dot<BLOCK_SIZE, OperatorA>(grid, N, A, ws.p, ws.v, ws.r0, &ws.scalars[ws.TEMP_DOT]); // v = A * p, dot product with r0
			
			alpha = ws.scalars[ws.RHO_NEW] / ws.scalars[ws.TEMP_DOT]; // compute alpha
#if DEBUG_PRINT
			printf("%d: Iteration %d: alpha = %f, TEMP_DOT = %f\n", tid, k, alpha, ws.scalars[ws.TEMP_DOT]);
#endif
			// compute s = r - alpha * v for this row and the norm squared of s
#if DEBUG_PRINT
			printf("%d: Calling grid_axpy_and_norm for s = r - alpha * v\n", tid);
#endif
			grid_axpy_and_norm<BLOCK_SIZE>(grid, N, -alpha, ws.v, ws.r, ws.s, &ws.scalars[ws.RES_TEMP]);

			if(ws.scalars[ws.RES_TEMP] < tol * tol)
			{   
#if DEBUG_PRINT
				printf("%d: Early convergence detected at iteration %d with residual norm %f\n", tid, k, ws.scalars[ws.RES_TEMP]);
#endif
				// check convergence on s
				// if the norm of s is small enough, we can finish
				// compute the final x update, x = x + alpha * p;
				if (blockIdx.x == 0 && threadIdx.x == 0) 
				{
					*ws.residual_norm = ws.scalars[ws.RES_TEMP]; // compute the final residual norm
					*ws.iterations = k + 1; // store the number of iterations
				}
				grid_axpy_and_norm<BLOCK_SIZE>(grid, N, alpha, ws.p, x, x, nullptr); // x += alpha * p
				
				break;
			}

			// compute t = A * s for this row and s . t and t. t at the same time
#if DEBUG_PRINT
			printf("%d: Calling grid_apply_and_two_dots for t = A * s\n", tid);
#endif
			grid_apply_and_two_dots<BLOCK_SIZE, OperatorA>(grid, N, A, ws.s, ws.t, ws.s, &ws.scalars[ws.TS], &ws.scalars[ws.TT]);

			omega = ws.scalars[ws.TS] / ws.scalars[ws.TT]; // compute omega
#if DEBUG_PRINT
			printf("%d: Iteration %d: omega = %f, TS = %f, TT = %f\n", tid, k, omega, ws.scalars[ws.TS], ws.scalars[ws.TT]);
#endif
			// update x = x + alpha * p + omega * s
			grid_axbvpy<BLOCK_SIZE>(N, alpha, omega, ws.p, ws.s, x, x);
#if DEBUG_PRINT
			printf("%d: Updated x = x + alpha * p + omega * s\n", tid);
#endif
			// calculate the residual: r = s - omega * t
			if (blockIdx.x == 0 && threadIdx.x == 0)
			{
				// update the rho:
#if DEBUG_PRINT
				printf("%d: Updating rho_new = %f\n", tid, rho_new);
#endif
				ws.scalars[ws.RHO_OLD] = rho_new; // store the old rho for the next iteration
			}
#if DEBUG_PRINT
			printf("%d: Calling grid_axpy_norm_dot for r = s - omega * t (new residual)\n", tid);
#endif
			grid_axpy_norm_dot<BLOCK_SIZE>(grid, N, -omega, ws.t, ws.s, ws.r0, ws.r, &ws.scalars[ws.RHO_NEW], ws.residual_norm);
#if DEBUG_PRINT
			printf("%d: Updated residual r = s - omega * t\n", tid);
			printf("%d: Iteration %d, residual norm %f - end\n", tid, k, *ws.residual_norm);
#endif
			if (*ws.residual_norm < tol * tol) {
#if DEBUG_PRINT
				printf("%d: Early convergence detected at iteration %d with residual norm %f\n", tid, k, *ws.residual_norm);
#endif
				*ws.iterations = k + 1; // store the number of iterations
				break;
			}
		}
	}

	
}