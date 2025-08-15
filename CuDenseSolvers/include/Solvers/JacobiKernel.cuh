#pragma once
#include "Common.cuh"
#include "JacobiWorkspace.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace CuDenseSolvers {
	/// <summary>
	/// Calculates the update
	/// </summary>
	__global__ void JacobiKernel(double* y, double* b, JacobiWorkspace workspace) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		if (tid < workspace.size) {

			double ri = b[tid] - y[tid];
			workspace.xnew[tid] = workspace.xprevious[tid] + *workspace.omega * workspace.dinv[tid] * ri;
		}
	}
}