#pragma once
#include "../Operators/MatrixOperator.cuh"
#include "JacobiWorkspace.cuh"
#include "JacobiKernel.cuh"

template <size_t N>
class Jacobi
{
private:
	DoubleMatrixOperator<N>* op;
	JacobiWorkspace* workspace;
	int blockSize;
	int threads;
public:
	Jacobi();
	~Jacobi();
	
	void solve(double* x, double* b, size_t maxIterations, double tol) {

		double* xold;
		double* xnew;
		for (int k = 0; k < maxIterations; ++k) {
			
			if (k == 0) {
				xold = x;
			}
			else {
				xold = workspace->xprevious;
			}

			op->apply(xold, workspace->xnew);
			CuDenseSolvers::JacobiKernel << <blockSize, threads >> > (workspace->xnew, b, *workspace);


			xold = workspace->xnew;
			xnew = workspace->xprevious;

			workspace->xprevious = xold;
			workspace->xnew = xnew;
		}

		
	}
	void setOperator(DoubleMatrixOperator<N>* op) 
	{
		this->op = op;
		workspace = new JacobiWorkspace(op->size());
		threads = 256;
		blockSize = (N + threads - 1) / threads;
	}

};

template <size_t N>
Jacobi<N>::Jacobi()
{
}

template <size_t N>
Jacobi<N>::~Jacobi()
{
	delete workspace;
}