#!/usr/bin/env python3
"""
Benchmark script for the GMG solver
Demonstrates the Python implementation with various problem sizes
"""

from gmg_solver import GMGSolver
import time


def benchmark_problem(nx, ny, nz, num_levels, max_iterations):
    """Benchmark a single problem size"""
    print(f"\nBenchmarking {nx}x{ny}x{nz} grid with {num_levels} levels...")
    
    solver = GMGSolver(nx, ny, nz, num_levels)
    start_time = time.time()
    residual, iterations, timers = solver.solve(max_iterations)
    total_time = time.time() - start_time
    
    print(f"Results:")
    print(f"  Final residual: {residual:.6e}")
    print(f"  Iterations: {iterations}")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Time per iteration: {total_time/iterations:.4f}s")
    print(f"  Grid points: {nx*ny*nz:,}")
    print(f"  Grid points/sec: {nx*ny*nz*iterations/total_time:,.0f}")
    
    return residual, iterations, total_time


def main():
    """Run benchmark tests"""
    print("=" * 70)
    print("Geometric Multigrid Solver - Python Implementation Benchmark")
    print("=" * 70)
    
    # Test problems (nx, ny, nz, levels, max_iterations)
    problems = [
        (16, 16, 16, 10, 15),   # Small problem
    ]
    
    results = []
    
    for nx, ny, nz, levels, max_iter in problems:
        try:
            residual, iterations, total_time = benchmark_problem(nx, ny, nz, levels, max_iter)
            results.append((nx, ny, nz, residual, iterations, total_time))
        except Exception as e:
            print(f"Failed: {e}")
            results.append((nx, ny, nz, float('inf'), 0, 0))
    
    # Summary
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    print(f"{'Grid Size':<15} {'Residual':<12} {'Iterations':<10} {'Time (s)':<10} {'Converged':<10}")
    print("-" * 70)
    
    for nx, ny, nz, residual, iterations, total_time in results:
        grid_size = f"{nx}x{ny}x{nz}"
        converged = "Yes" if residual < 1e-10 else "No"
        print(f"{grid_size:<15} {residual:<12.2e} {iterations:<10} {total_time:<10.4f} {converged:<10}")
    
    print("\nNote: This Python implementation is intended for algorithm validation")
    print("and comparison with the optimized CUDA Bricks library implementation.")


if __name__ == "__main__":
    main()
