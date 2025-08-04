#pragma once

template<typename T>
class LinearOperator
{
public:
	LinearOperator();
	virtual ~LinearOperator();
	virtual void apply(const T* x, T* y, cudaStream_t stream = cudaStreamPerThread) const = 0;
	virtual int size() const = 0;
private:
	
};

template<typename T>
LinearOperator<T>::LinearOperator()
{
}

template<typename T>
LinearOperator<T>::~LinearOperator()
{
}