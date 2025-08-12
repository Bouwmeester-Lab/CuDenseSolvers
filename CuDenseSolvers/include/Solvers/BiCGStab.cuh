#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Operators/LinearOperator.h"
#include "Solver.cuh"
#include <cublas_v2.h>
#include <stdexcept>
#include "BiCGStabKernel.cuh"
namespace CuDenseSolvers {

	class DoubleBiCGStab : public IterativeSolver<double>
	{
	public:
		DoubleBiCGStab();
		~DoubleBiCGStab();
		virtual int solve(const double* b, double* x, int maxIterations, double tolerance, cudaStream_t stream = cudaStreamPerThread) override;
		virtual void setOperator(const LinearOperator<double>& A) override;
		virtual int getNumIterations() const override;
		// Method to get the residual norm
		virtual double getResidualNorm() const override;
	private:
		// double* workspace = nullptr; // Pointer to workspace memory
		BiCGStabWorkspace* workspace; // Workspace object to manage memory
		size_t size_required = 0; // Size required for the workspace
		int numIterations = 0;


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
		workspace->zero(); // Initialize workspace to zero
		const size_t dynSmem = 0;

		int desiredBlocks = (A->size() + 255) / 256;
		void* args[] = { workspace, &A, &b, &x, &maxIterations, &tolerance, &numIterations };

		cudaLaunchCooperativeKernel((void*)BiCGStabKernel<LinearOperator<double>>, desiredBlocks, 256, args, dynSmem, stream);

		//if(!A)
		//{
		//	throw std::runtime_error("Operator A is not set.");
		//}
		//// Initialize buffers to zero
		//cudaMemset(p, 0, A->size() * sizeof(double));
		//cudaMemset(v, 0, A->size() * sizeof(double));

		//// compute the initial residual r = b - Ax
		//A->apply(x, r, stream); // r = Ax
		//cublasDcopy(handle, A->size(), b, 1, r0, 1); // r0 <= b
		//const double neg1 = -1.0;
		//cublasDaxpy(handle, A->size(), &neg1, r, 1, r0, 1); // r0 = r0 - Ax
		//cudaMemcpyAsync(r, r0, A->size() * sizeof(double), cudaMemcpyDeviceToDevice, stream); // r <= r0

		//double rho_old = 1.0, alpha = 1.0, omega = 1.0;
		//double rho_new = 0.0, beta = 0.0;
		//resid = 0.0;

		//for (int k = 0; k < maxIterations; ++k)
		//{
		//	// Compute rho_new = r0 . r
		//	cublasDdot(handle, A->size(), r0, 1, r, 1, &rho_new);
		//	
		//	if (rho_new == 0.0) break; // Break if the residual is zero, since there's no progress.

		//	beta = (rho_new / rho_old) * (alpha / omega);
		//	
		//	// Update p vector
		//	// p = alpha * r + beta * (p - omega * v)
		//	// p = p - omega * v

		//	const double negOmega = -omega;
		//	cublasDaxpy(handle, A->size(), &negOmega, v, 1, p, 1); // p = p - omega * v
		//	// p = beta * p
		//	cublasDscal(handle, A->size(), &beta, p, 1);
		//	// p = p + alpha * r
		//	cublasDaxpy(handle, A->size(), &alpha, r, 1, p, 1);

		//	// v = A * p
		//	A->apply(p, v, stream);

		//	double temp_dot;
		//	cublasDdot(handle, A->size(), r0, 1, v, 1, &temp_dot); // temp_dot = r0 . v
		//	alpha = rho_new / temp_dot;

		//	// s = r - alpha * v
		//	const double negAlpha = -alpha;
		//	cublasDcopy(handle, A->size(), r, 1, s, 1); // s <= r
		//	cublasDaxpy(handle, A->size(), &negAlpha, v, 1, s, 1); // s = s - alpha * v
		//	
		//	// Check if the norm of s is small enough
		//	cublasDnrm2(handle, A->size(), s, 1, &resid);

		//	if(resid < tolerance) {
		//		// If the residual is small enough, we can stop
		//		cublasDaxpy(handle, A->size(), &alpha, p, 1, x, 1); // x = x + alpha * p
		//		numIterations = k + 1; // Return number of iterations
		//		return numIterations;
		//	}

		//	// t = A * s
		//	A->apply(s, t, stream);
		//	double ts, tt;
		//	cublasDdot(handle, A->size(), s, 1, t, 1, &ts); // ts = s . t
		//	cublasDdot(handle, A->size(), t, 1, t, 1, &tt); // tt = t . t
		//	omega = ts / tt;

		//	// x = x + alpha * p + omega * s
		//	cublasDaxpy(handle, A->size(), &alpha, p, 1, x, 1);
		//	cublasDaxpy(handle, A->size(), &omega, s, 1, x, 1);

		//	// r = s - omega * t // this is the new residual. It's equivalent to doing b - A x with x been the updated solution above. r = b - A (x_0 + alpha * p + omega * s) = b - A * x_0 - alpha * A * p - omega * A * s = r_old - alpha * v - omega * t = s  - omega * t 
		//	cublasDcopy(handle, A->size(), s, 1, r, 1); // r <= s
		//	double neg_omega = -omega;
		//	cublasDaxpy(handle, A->size(), &neg_omega, t, 1, r, 1); // r = r - omega * t

		//	// Check convergence: ||r|| < tol
		//	cublasDnrm2(handle, A->size(), r, 1, &resid);
		//	if (resid < tolerance) break;


		//	rho_old = rho_new;
		//}

		//return (resid < tolerance) ? 0 : 1;
	}

	void DoubleBiCGStab::setOperator(const LinearOperator<double>& Aop)
	{
		if ((A != nullptr && Aop.size() != this->A->size()) || A == nullptr)
		{
			if (A != nullptr)
				freeBuffers();  // If switching or resizing
			A = &Aop;
			allocateBuffers();
		}

	}

	int DoubleBiCGStab::getNumIterations() const
	{
		return numIterations;
	}

	double DoubleBiCGStab::getResidualNorm() const
	{
		return *workspace->residual_norm;
	}

	void DoubleBiCGStab::allocateBuffers()
	{
		if (A == nullptr) {
			throw std::runtime_error("Operator A is not set.");
		}
		//size_t size_required = BiCGStabWorkspace::getRequiredSize(A->size());
		workspace = new BiCGStabWorkspace(A->size());


	}

	void DoubleBiCGStab::freeBuffers()
	{
		delete workspace;
	}

}