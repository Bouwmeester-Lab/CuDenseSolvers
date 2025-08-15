#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#ifdef __INTELLISENSE__ // for Visual Studio IntelliSense https://stackoverflow.com/questions/77769389/intellisense-in-visual-studio-cannot-find-cuda-cooperative-groups-namespace
#define __CUDACC__
#endif // __INTELLISENSE__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#ifdef __INTELLISENSE__
#undef __CUDACC__
#endif // __INTELLISENSE__
namespace cg = cooperative_groups;

namespace CuDenseSolvers {


template<int BLOCK_SIZE, typename Op, typename Type>
__device__ Type block_reduce_sum(Type local) {
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<32>(block);       // 32-thread tile

	// 1) reduce within each warp
	Type warp_sum = cg::reduce(warp, local, Op());

	// 2) write one value per warp to shared mem
	__shared__ Type smem[BLOCK_SIZE / 32];
	if (warp.thread_rank() == 0)
		smem[warp.meta_group_rank()] = warp_sum;
	block.sync();

	// 3) first warp reduces the per-warp sums
	Type block_sum = 0.0;
	if (warp.meta_group_rank() == 0) {
		int num_warps = (BLOCK_SIZE + 31) / 32;
		Type v = (warp.thread_rank() < num_warps) ? smem[warp.thread_rank()] : 0.0;
		block_sum = cg::reduce(warp, v, Op());  // result in lane 0 of this warp
	}
	return (warp.meta_group_rank() == 0 && warp.thread_rank() == 0) ? block_sum : 0.0;
}

template<int BLOCK_SIZE, typename Op>
__device__ double2 block_reduce_sum(double2 local) {
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<32>(block);       // 32-thread tile

	// 1) reduce within each warp
	double2 warp_sum = cg::reduce(warp, local, Op());

	// 2) write one value per warp to shared mem
	__shared__ double2 smem[BLOCK_SIZE / 32];
	if (warp.thread_rank() == 0)
		smem[warp.meta_group_rank()] = warp_sum;
	block.sync();

	// 3) first warp reduces the per-warp sums
	double2 block_sum = make_double2(0.0, 0.0);
	if (warp.meta_group_rank() == 0) {
		int num_warps = (BLOCK_SIZE + 31) / 32;
		double2 v = (warp.thread_rank() < num_warps) ? smem[warp.thread_rank()] : make_double2(0.0, 0.0);
		block_sum = cg::reduce(warp, v, Op());  // result in lane 0 of this warp
	}
	return (warp.meta_group_rank() == 0 && warp.thread_rank() == 0) ? block_sum : make_double2(0.0, 0.0);
}

struct PairPlus {
	__device__ double2 operator()(double2 a, double2 b) const {
		return make_double2(a.x + b.x, a.y + b.y);
	}
};

/// <summary>
/// Applies the operator A to the vector x, stores the result in y, and computes two dot products with vectors w1 and with y
/// The dot products are stored in out_dot1 and out_dot2 respectively.
/// </summary>	
/// 
template<int BS, class A>
__device__ inline void grid_apply_and_two_dots(
	cg::grid_group grid, int n,
	const A& Aop,
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
	double2 block_sum = block_reduce_sum<BS, PairPlus>(local);
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
	const A Aop,
	const double* __restrict__ x,
	const double* __restrict__ b,
	double* __restrict__ r,
	double* __restrict__ out_norm2, // may be nullptr, make sure it's initialized to 0.0 before calling
	double* __restrict__ out_r0 = nullptr // may be nullptr
) {
#if DEBUG_PRINT
	printf("grid_residual_and_norm called with n = %d\n", n);
#endif
	bool sync_required = false;
	if (out_norm2 != nullptr && *out_norm2 != 0.0) {
		*out_norm2 = 0.0; // ensure out_norm2 is initialized to 0.0
		sync_required = true; // we need to sync before we can use out_norm2
	}
	if (sync_required) {
#if DEBUG_PRINT
		printf("grid_residual_and_norm syncing before computation\n");
#endif
		grid.sync();
	}

	auto block = cg::this_thread_block();

	const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	const int stride = gridDim.x * BLOCK_SIZE;

	double local = 0.0;

	for (int i = tid; i < n; i += stride) {
#if DEBUG_PRINT
		printf("Thread %d processing row %d\n", tid, i);
#endif
		double yi = Aop(i, x);          // generic row action
#if DEBUG_PRINT
		printf("%d: Aop(i, x) called for i = %d, yi = %f\n", tid, i, yi);
#endif
		double ri = b[i] - yi;
		r[i] = ri;
		if (out_norm2) local += ri * ri; // fuse norm squared
		if (out_r0) out_r0[i] = ri; // store initial residual if requested
	}
#if DEBUG_PRINT
	printf("%d: Value of local after loop: %f\n", tid, local);
#endif
	// Block sync sufficient - no inter-block dependencies in computation
	__syncthreads();

	if (out_norm2) {


		// block-wide reduction using CG
		double block_sum = block_reduce_sum<BLOCK_SIZE, cg::plus<double>, double>(local);
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
	const double alpha,
	const double* __restrict__ x,
	const double* __restrict__ y,

	double* __restrict__ yout,
	double* __restrict__ out_norm2 // may be nullptr, make sure it's initialized to 0.0 before calling
) {
	bool sync_required = false;
	if (out_norm2 != nullptr && *out_norm2 != 0.0) {
		*out_norm2 = 0.0; // ensure out_norm2 is initialized to 0.0
		sync_required = true; // we need to sync before we can use out_norm2
	}
	if (sync_required) {
		grid.sync();
	}

	auto block = cg::this_thread_block();

	const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	const int stride = gridDim.x * BLOCK_SIZE;

	double local = 0.0;

	for (int i = tid; i < n; i += stride) {
		yout[i] = y[i] + (alpha)*x[i]; // y += alpha * x
		if (out_norm2) local += yout[i] * yout[i]; // fuse norm squared
	}




	if (out_norm2) {
		// Block sync sufficient - no inter-block dependencies in computation
		__syncthreads();
		// block-wide reduction using CG
		double block_sum = block_reduce_sum<BLOCK_SIZE, cg::plus<double>, double>(local);
		if (block.thread_rank() == 0) atomicAdd(out_norm2, block_sum);
		// sync before consumers read out_norm2 or r
		grid.sync();
	}
}

void zeroGlobalValue(double* globalValue) {
	*globalValue = 0.0; // ensure the global value is initialized to 0.0
}

/// <summary>
/// Calculate yout = y + alpha*x  and optionally the norm squared of y, and return the dot product of yout with r0
/// </summary>
template<int BLOCK_SIZE>
__device__ inline void grid_axpy_norm_dot(
	cg::grid_group grid,
	int n,
	const double alpha,
	const double* __restrict__ x,
	const double* __restrict__  y,
	const double* __restrict__ r0,

	double* __restrict__ yout,
	double* __restrict__ out_dot, // may be nullptr, make sure it's initialized to 0.0 before calling
	double* __restrict__ out_norm2 // may be nullptr, make sure it's initialized to 0.0 before calling
) {
#if DEBUG_PRINT
	printf("Called.\n");
#endif
	const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	const int stride = gridDim.x * BLOCK_SIZE;
	bool sync_required = false;
	if (out_norm2 != nullptr && *out_norm2 != 0.0) { //
#if DEBUG_PRINT
		printf("%d: grid_axpy_norm_dot: out_norm2 is not zero, initializing to 0.0\n", tid);
		printf("%d: grid_axpy_norm_dot: out_norm2 points to %f\n", tid, out_norm2);
#endif
		if (blockIdx.x == 0 && threadIdx.x == 0)
		{
#if DEBUG_PRINT
			printf("%d: %f", tid, *out_norm2);
#endif
			* out_norm2 = 0.0; // ensure out_norm2 is initialized to 0.0
		}
		sync_required = true; // we need to sync before we can use out_norm2
	}
	//if(out_dot != nullptr && *out_dot != 0.0) {
	//	printf("%d: grid_axpy_norm_dot: out_dot is not zero, initializing to 0.0\n", tid);
	//	*out_dot = 0.0; // ensure out_dot is initialized to 0.0
	//	sync_required = true; // we need to sync before we can use out_dot
	//}
	if (sync_required) {
#if DEBUG_PRINT
		printf("%d: grid_axpy_norm_dot syncing before computation\n", tid);
#endif
		grid.sync();
#if DEBUG_PRINT
		printf("%d: grid_axpy_norm_dot synced\n", tid);
#endif
	}
#if DEBUG_PRINT
	printf("%d: grid_axpy_norm_dot called with n = %d\n", tid, n);
#endif
	auto block = cg::this_thread_block();



	double local = 0.0;
	double local_dot = 0.0; // for the dot product with r0

	for (int i = tid; i < n; i += stride) {
		yout[i] = y[i] + (alpha)*x[i]; // y += alpha * x
		if (out_norm2) local += yout[i] * yout[i]; // fuse norm squared
		if (out_dot) local_dot += yout[i] * r0[i]; // dot product with r0
	}




	if (out_norm2) {
		// Block sync sufficient - no inter-block dependencies in computation
		__syncthreads();
		// block-wide reduction using CG
		double block_sum = block_reduce_sum<BLOCK_SIZE, cg::plus<double>, double>(local);
		if (block.thread_rank() == 0) atomicAdd(out_norm2, block_sum);
	}
	if (out_dot) {
		// Block sync sufficient - no inter-block dependencies in computation
		__syncthreads();
		// block-wide reduction using CG
		double block_sum = block_reduce_sum<BLOCK_SIZE, cg::plus<double>, double>(local_dot);
		if (block.thread_rank() == 0) atomicAdd(out_dot, block_sum);
	}
	if (out_dot != nullptr || out_norm2 != nullptr) {
		// ensure all threads have completed their work before returning
		grid.sync();
	}
}

/// <summary>
/// Calculate the residual yout = y + alpha*x + beta * s  and optionally the norm squared of y.
/// </summary>
template<int BLOCK_SIZE>
__device__ inline void grid_axbvpy(

	int n,
	const double alpha,
	const double beta,
	const double* __restrict__ x,
	const double* __restrict__ s,
	const double* __restrict__ y,
	double* __restrict__ yout
) {

	const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	const int stride = gridDim.x * BLOCK_SIZE;



	for (int i = tid; i < n; i += stride) {
		yout[i] = y[i] + alpha * x[i] + beta * s[i]; // y += alpha * x + beta * s	
	}
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
		double block_sum = block_reduce_sum<BLOCK_SIZE, cg::plus<double>, double>(local);
		if (block.thread_rank() == 0) atomicAdd(out_dot, block_sum);
	}

	grid.sync();
}
}