#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Operators/LinearOperator.h"
#include "Solver.cuh"
#include <cublas_v2.h>
#include <stdexcept>

class DoubleBiCGStab : public IterativeSolver<double>
{
public:
	DoubleBiCGStab();
	~DoubleBiCGStab();
	virtual int solve(const double* b, double* x, int maxIterations, double tolerance, cudaStream_t stream = cudaStreamPerThread) override;
	virtual void setOperator(const LinearOperator<double>& A) override;
private:
	/// <summary>
	/// Residual vector
	/// </summary>
	double* r;          
	/// <summary>
	/// Fixed reference vector
	/// </summary>
	double* r0;
	// temporary vectors
	double* p;
	double* v;
	double* s;
	double* t;

	cublasHandle_t handle;

	const LinearOperator<double>* A = nullptr;  // Non-owning pointer

	void allocateBuffers();
	void freeBuffers();
};

DoubleBiCGStab::DoubleBiCGStab()
{
	cublasCreate(&handle);
}

DoubleBiCGStab::~DoubleBiCGStab()
{
	freeBuffers();
	cublasDestroy(handle);
}

int DoubleBiCGStab::solve(const double* b, double* x, int maxIterations, double tolerance, cudaStream_t stream)
{
	if(!A)
	{
		throw std::runtime_error("Operator A is not set.");
	}

	// compute the initial residual r = b - Ax
	A->apply(x, r, stream); // r = Ax
	cublasDcopy(handle, A->size(), b, 1, r0, 1); // r0 = b
	const double neg1 = -1.0;
	cublasDaxpy(handle, A->size(), &neg1, r, 1, r0, 1); // r0 = b - Ax
	cudaMemcpyAsync(r, r0, A->size() * sizeof(double), cudaMemcpyDeviceToDevice, stream); // r <= r0

	double rho_old = 1.0, alpha = 1.0, omega = 1.0;
	double rho_new = 0.0, beta = 0.0;
	double resid = 0.0;

	for (int k = 0; k < maxIterations; ++k)
	{
		// Compute rho_new = r0 . r
		cublasDdot(handle, A->size(), r0, 1, r, 1, &rho_new);
		
		if (rho_new == 0.0) break; // Break if the residual is zero, since there's no progress.

		beta = (rho_new / rho_old) * (alpha / omega);
		
		// Update p vector
		// p = r + beta * (p - omega * v)
		// p = p - omega * v

		const double negOmega = -omega;
		cublasDaxpy(handle, A->size(), &negOmega, v, 1, p, 1); // p = p - omega * v
		// p = beta * p
		cublasDscal(handle, A->size(), &beta, p, 1);
		// p = p + alpha * r
		cublasDaxpy(handle, A->size(), &alpha, r, 1, p, 1);

	}
}

void DoubleBiCGStab::setOperator(const LinearOperator<double>& Aop)
{
	if ((A != nullptr && Aop.size() != this->A->size()) || A == nullptr)
	{
		if(A != nullptr)
			freeBuffers();  // If switching or resizing
		A = &Aop;
		allocateBuffers();
	}
	
}

void DoubleBiCGStab::allocateBuffers()
{
	if(A == nullptr) {
		throw std::runtime_error("Operator A is not set.");
	}
	int N = A->size();
	cudaMalloc(&r, N * sizeof(double));
	cudaMalloc(&r0, N * sizeof(double));
	cudaMalloc(&p, N * sizeof(double));
	cudaMalloc(&v, N * sizeof(double));
	cudaMalloc(&s, N * sizeof(double));
	cudaMalloc(&t, N * sizeof(double));

	// Initialize buffers to zero
	cudaMemset(p, 0, N * sizeof(double));
	cudaMemset(v, 0, N * sizeof(double));
}

void DoubleBiCGStab::freeBuffers()
{
	cudaFree(r);
	cudaFree(r0);
	cudaFree(p);
	cudaFree(v);
	cudaFree(s);
	cudaFree(t);
}
