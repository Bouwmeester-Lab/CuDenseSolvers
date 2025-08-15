#pragma once

#include "LinearOperator.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include "cublas_v2.h"
#include <stdexcept>

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

template<size_t N>
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
/// <summary>
/// It's important to note that this functor needs to be passed by value to the kernel, as it contains a pointer to device memory.
/// </summary>
/// <typeparam name="N"></typeparam>
template<size_t N>
struct MatrixOpFunctor 
{
	const double* devA; // Device pointer to matrix A

	__host__ __device__ MatrixOpFunctor(const double* devA) : devA(devA)
	{
		//printf("MatrixOpFunctor created with devA pointing to %p\n", devA);
	}

	__device__ double operator() (int row, const double* __restrict__ devX) const
	{
		double result = 0.0;
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		int indx = 0;
		for(size_t col = 0; col < N; col++) 
		{
			indx = row + N * col;
			result +=  devA[indx] * devX[col];
		}
#ifdef DEBUG_PRINT
		printf("%d : Result for row %d is %f\n", tid, row, result);
#endif // DEBUG_PRINT
		return result;
	}
};

