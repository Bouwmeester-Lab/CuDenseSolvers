#pragma once

#include "LinearOperator.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include "cublas_v2.h"

template <typename T, size_t N>
class MatrixOperator : public LinearOperator<T>
{
public:
	MatrixOperator();
	virtual ~MatrixOperator() override
	{
		cudaFree(devData);
		cublasDestroy(handle);
	}

	virtual int size() const override;
	
	void setData(const std::vector<T>& data)
	{
		if (data.size() != N * N) {
			throw std::runtime_error("Data size does not match matrix size.");
		}
		cudaMemcpy(devData, data.data(), N * N * sizeof(T), cudaMemcpyHostToDevice);
	}
	void setData(const T* data)
	{
		cudaMemcpy(devData, data, N * N * sizeof(T), cudaMemcpyHostToDevice);
	}

protected:
	T* devData; // Matrix data in GPU memory
	cublasHandle_t handle; // CUBLAS handle for matrix operations
};

template<typename T, size_t N>
MatrixOperator<T, N>::MatrixOperator()
{
	cudaMalloc(&devData, N * N * sizeof(T));
	cublasCreate(&handle);
}

template<typename T, size_t N>
int MatrixOperator<T, N>::size() const
{
	return N;
}

template<rsize_t N>
class DoubleMatrixOperator : public MatrixOperator<double, N>
{
public:
	using MatrixOperator<double, N>::MatrixOperator;
	virtual void apply(const double* x, double* y, cudaStream_t stream = cudaStreamPerThread) const override
	{
		cublasSetStream(this->handle, stream);
		// Implement the matrix-vector multiplication using CUBLAS
		double alpha = 1.0;
		double beta = 0.0;
		cublasDgemv(this->handle, CUBLAS_OP_N, N, N, &alpha, this->devData, N, x, 1, &beta, y, 1);
	}
private:

};

