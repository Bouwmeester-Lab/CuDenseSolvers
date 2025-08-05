#pragma once
#include <gtest/gtest.h>
#include "Solvers/BiCGStab.cuh"
#include <Operators/MatrixOperator.cuh>



TEST(SolverTests, TestMatrixOperator) {
	DoubleMatrixOperator<2> A;
	// set the matrix data in column-major order (Fortran order)
	A.setData({
		4.0, 2.0, 1.0, 3.0
		});

	double x[2] = { 1.0, 0.0 }; // Input vector
	// copy the input vector to device memory
	double* devX;
	double* devY;
	cudaMalloc(&devX, 2 * sizeof(double));
	cudaMalloc(&devY, 2 * sizeof(double));

	cudaMemcpy(devX, x, 2 * sizeof(double), cudaMemcpyHostToDevice);

	A.apply(devX, devY);

	cudaDeviceSynchronize(); // Ensure the kernel has completed before copying back
	double y[2];
	cudaMemcpy(y, devY, 2 * sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(devX);
	cudaFree(devY);


	// check the result is what we expect
	EXPECT_DOUBLE_EQ(y[0], 4.0 * x[0] + 1.0 * x[1]);
	EXPECT_DOUBLE_EQ(y[1], 2.0 * x[0] + 3.0 * x[1]);
}

TEST(SolverTests, TestBiCGStabSolver) {
	DoubleMatrixOperator<2> A;
	double a[4] = { 4.0, 2.0, 1.0, 3.0 }; // Matrix data in column-major order

	A.setData(a);
	double b[2] = { 1.0, 0.0 }; // Right-hand side vector
	double x[2] = { 0.0, 0.0 }; // Initial guess

	double* devb;
	double* devx;

	cudaMalloc(&devb, 2 * sizeof(double));
	cudaMalloc(&devx, 2 * sizeof(double));
	cudaMemcpy(devb, b, 2 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devx, x, 2 * sizeof(double), cudaMemcpyHostToDevice);

	DoubleBiCGStab solver;
	solver.setOperator(A);

	int maxIterations = 100;
	double tolerance = 1e-6;

	int iterations = solver.solve(devb, devx, maxIterations, tolerance);

	cudaDeviceSynchronize(); // Ensure the kernel has completed before checking results
	cudaMemcpy(x, devx, 2 * sizeof(double), cudaMemcpyDeviceToHost);


	EXPECT_LE(iterations, maxIterations);
	EXPECT_LT(solver.getResidualNorm(), tolerance);

	double det = 1 / (a[0] * a[3] - a[1] * a[2]);
	double y1 = det * (a[3] * b[0] - a[2] * b[1]);
	double y2 = det * (a[0] * b[1] - a[1] * b[0]);

	EXPECT_NEAR(x[0], y1, tolerance);
	EXPECT_NEAR(x[1], y2, tolerance);
}