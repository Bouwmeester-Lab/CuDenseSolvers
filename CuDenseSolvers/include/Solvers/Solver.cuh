#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Operators/LinearOperator.h"


template <typename T>
class IterativeSolver
{
public:
	virtual ~IterativeSolver() = default;

	/// <summary>
	/// Method to solve the linear system Ax = b using an iterative method.
	/// </summary>
	/// <param name="A">The linear operator to use. This represent the matrix A, but avoids storing the matrix since we only care about the Ax vector which this calculates without storing A.</param>
	/// <param name="b">The RHS of the equation to solve. Vector of size N</param>
	/// <param name="x"></param>
	/// <param name="maxIterations"></param>
	/// <param name="tolerance"></param>
	virtual int solve(const T* b, T* x, int maxIterations, double tolerance, cudaStream_t stream = cudaStreamPerThread) = 0;
	// Method to get the number of iterations performed
	virtual int getNumIterations() const = 0;
	// Method to get the residual norm
	virtual double getResidualNorm() const = 0;
	// virtual void setOperator(const LinearOperator<T>& A) = 0;
};