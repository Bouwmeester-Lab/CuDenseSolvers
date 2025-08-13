#pragma once
#include <gtest/gtest.h>
#include "Solvers/BiCGStab.cuh"
#include "Solvers/BiCGStabWorkspace.cuh"
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
	
	double a[4] = { 4.0, 2.0, 1.0, 3.0 }; // Matrix data in column-major order

	double* devA;

	cudaMalloc(&devA, 4 * sizeof(double));
	cudaMemcpy(devA, a, 4 * sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize(); // Ensure the kernel has completed before checking results
	MatrixOpFunctor<2> A(devA); // Create the operator with the matrix data
	

	double b[2] = { 1.0, 0.0 }; // Right-hand side vector
	double x[2] = { 0.0, 0.0 }; // Initial guess

	double* devb;
	double* devx;

	cudaMalloc(&devb, 2 * sizeof(double));
	cudaMalloc(&devx, 2 * sizeof(double));
	cudaMemcpy(devb, b, 2 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devx, x, 2 * sizeof(double), cudaMemcpyHostToDevice);

	CuDenseSolvers::DoubleBiCGStab<MatrixOpFunctor<2>, 2> solver;

	int maxIterations = 100;
	double tolerance = 1e-6;

	int iterations = solver.solve(A, devb, devx, maxIterations, tolerance);

	cudaDeviceSynchronize(); // Ensure the kernel has completed before checking results
	cudaMemcpy(x, devx, 2 * sizeof(double), cudaMemcpyDeviceToHost);


	EXPECT_LE(iterations, maxIterations);
	EXPECT_LT(solver.getResidualNorm(), tolerance);
	EXPECT_GT(solver.getNumIterations(), 0);

	double det = 1 / (a[0] * a[3] - a[1] * a[2]);
	double y1 = det * (a[3] * b[0] - a[2] * b[1]);
	double y2 = det * (a[0] * b[1] - a[1] * b[0]);

	EXPECT_NEAR(x[0], y1, tolerance);
	EXPECT_NEAR(x[1], y2, tolerance);
}
//
//TEST(SolverTests, TestBiCGStabWorkspace) {
//	const int n = 10; // Test with a system of size 10
//	
//	// Test memory size calculation
//	size_t requiredSize = BiCGStabWorkspace::getRequiredSize(n);
//	EXPECT_GT(requiredSize, 6 * n * sizeof(double)); // Should be at least 6 vectors worth
//	
//	// Test workspace allocation
//	BiCGStabWorkspace workspace(n);
//
//	EXPECT_NE(workspace.rawMemory, nullptr);
//	EXPECT_TRUE(workspace.isValid());
//	EXPECT_EQ(workspace.n, n);
//	
//	// Test that all pointers are non-null and properly ordered
//	EXPECT_NE(workspace.r, nullptr);
//	EXPECT_NE(workspace.r0, nullptr);
//	EXPECT_NE(workspace.p, nullptr);
//	EXPECT_NE(workspace.v, nullptr);
//	EXPECT_NE(workspace.s, nullptr);
//	EXPECT_NE(workspace.t, nullptr);
//	EXPECT_NE(workspace.scalars, nullptr);
//	EXPECT_NE(workspace.iterations, nullptr);
//	EXPECT_NE(workspace.residual_norm, nullptr);
//	
//	// Test that vectors are contiguous and properly spaced
//	EXPECT_EQ(workspace.r0 - workspace.r, n);
//	EXPECT_EQ(workspace.p - workspace.r0, n);
//	EXPECT_EQ(workspace.v - workspace.p, n);
//	EXPECT_EQ(workspace.s - workspace.v, n);
//	EXPECT_EQ(workspace.t - workspace.s, n);
//	
//	// Test vector access by index
//	EXPECT_EQ(workspace.getVector(0), workspace.r);
//	EXPECT_EQ(workspace.getVector(1), workspace.r0);
//	EXPECT_EQ(workspace.getVector(2), workspace.p);
//	EXPECT_EQ(workspace.getVector(3), workspace.v);
//	EXPECT_EQ(workspace.getVector(4), workspace.s);
//	EXPECT_EQ(workspace.getVector(5), workspace.t);
//	EXPECT_EQ(workspace.getVector(6), nullptr); // Invalid index
//	
//	// Test scalar constants
//	EXPECT_EQ(BiCGStabWorkspace::RHO_NEW, 0);
//	EXPECT_EQ(BiCGStabWorkspace::ALPHA, 1);
//	EXPECT_EQ(BiCGStabWorkspace::OMEGA, 2);
//	EXPECT_EQ(BiCGStabWorkspace::BETA, 3);
//	EXPECT_EQ(BiCGStabWorkspace::TEMP_DOT, 4);
//	EXPECT_EQ(BiCGStabWorkspace::TS, 5);
//	EXPECT_EQ(BiCGStabWorkspace::TT, 6);
//	EXPECT_EQ(BiCGStabWorkspace::NUM_SCALARS, 7);
//	
//	// Test zero initialization
//	workspace.zero();
//	cudaDeviceSynchronize();
//	
//	// Verify vectors are zeroed (test first element of each)
//	double testValues[6];
//	cudaMemcpy(&testValues[0], workspace.r, sizeof(double), cudaMemcpyDeviceToHost);
//	cudaMemcpy(&testValues[1], workspace.r0, sizeof(double), cudaMemcpyDeviceToHost);
//	cudaMemcpy(&testValues[2], workspace.p, sizeof(double), cudaMemcpyDeviceToHost);
//	cudaMemcpy(&testValues[3], workspace.v, sizeof(double), cudaMemcpyDeviceToHost);
//	cudaMemcpy(&testValues[4], workspace.s, sizeof(double), cudaMemcpyDeviceToHost);
//	cudaMemcpy(&testValues[5], workspace.t, sizeof(double), cudaMemcpyDeviceToHost);
//	
//	for (int i = 0; i < 6; i++) {
//		EXPECT_EQ(testValues[i], 0.0);
//	}
//	
//	// Verify scalars are zeroed
//	double scalarValues[BiCGStabWorkspace::NUM_SCALARS];
//	cudaMemcpy(scalarValues, workspace.scalars, BiCGStabWorkspace::NUM_SCALARS * sizeof(double), cudaMemcpyDeviceToHost);
//	for (int i = 0; i < BiCGStabWorkspace::NUM_SCALARS; i++) {
//		EXPECT_EQ(scalarValues[i], 0.0);
//	}
//	
//	// Verify output is zeroed
//	int iterCount;
//	double residNorm;
//	cudaMemcpy(&iterCount, workspace.iterations, sizeof(int), cudaMemcpyDeviceToHost);
//	cudaMemcpy(&residNorm, workspace.residual_norm, sizeof(double), cudaMemcpyDeviceToHost);
//	EXPECT_EQ(iterCount, 0);
//	EXPECT_EQ(residNorm, 0.0);
//	
//	// Test stride calculation
//	EXPECT_EQ(workspace.getVectorStride(), n);
//	
//	// Test additional helper methods
//	EXPECT_EQ(workspace.getVectorBase(), workspace.r);
//	EXPECT_EQ(workspace.getVectorStorageSize(), 6 * n * sizeof(double));
//	
//	//// Test compatibility check
//	//auto [rawMemory3, workspace3] = allocateBiCGStabWorkspace(n);
//	//EXPECT_TRUE(workspace.isCompatibleWith(workspace3));
//	//freeBiCGStabWorkspace(rawMemory3);
//	//
//	//auto [rawMemory4, workspace4] = allocateBiCGStabWorkspace(n + 1);
//	//EXPECT_FALSE(workspace.isCompatibleWith(workspace4));
//	//freeBiCGStabWorkspace(rawMemory4);
//	//
//	//// Free the workspace
//	//freeBiCGStabWorkspace(rawMemory);
//}
//
//TEST(SolverTests, TestBiCGStabWorkspaceEdgeCases) {
//	// Test invalid size
//	BiCGStabWorkspace workspace1(0);
//	EXPECT_EQ(workspace1.rawMemory, nullptr);
//	EXPECT_FALSE(workspace1.isValid());
//	
//	BiCGStabWorkspace workspace2(-1);
//	EXPECT_EQ(workspace2.rawMemory, nullptr);
//	EXPECT_FALSE(workspace2.isValid());
//}