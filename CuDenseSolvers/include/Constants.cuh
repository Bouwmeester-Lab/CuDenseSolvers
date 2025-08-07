#pragma once
namespace CuDenseSolvers
{
	// Constants used in the solver
	constexpr double EPSILON = 1e-10; // Small value to avoid division by zero
	constexpr int MAX_ITERATIONS = 1000; // Default maximum number of iterations for solvers
	constexpr int BLOCK_SIZE = 256;
}
