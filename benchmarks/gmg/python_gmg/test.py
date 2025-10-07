#!/usr/bin/env python3
"""
Benchmark script for the GMG solver
Demonstrates the Python implementation with various problem sizes
"""

import argparse
from gmg import SimpleGMG
from gmgoscar import SimpleGMG as SimpleGMGOSCAR
import time


def benchmark_problem(nx, ny, nz, num_levels, verbose, tolerance, pre_iter, post_iter, bottom_iter, max_iterations):
    """Benchmark a single problem size"""
    print(f"\nBenchmarking {nx}x{ny}x{nz} grid with {num_levels} levels...")
    # solver = SimpleGMG(nx, ny, nz, num_levels, verbose, tolerance, pre_iter, post_iter, bottom_iter)
    solver = SimpleGMGOSCAR(nx, ny, nz, num_levels, verbose, tolerance, pre_iter, post_iter, bottom_iter)
    start_time = time.time()
    # residual, iterations = solver.solve(max_iterations)
    residual, iterations = solver.solve_iterative(max_iterations)
    total_time = time.time() - start_time
    return residual, iterations, total_time


def main():
    """Run benchmark tests"""
    print("=" * 70)
    print("Simple Geometric Multigrid Solver - Python Implementation Benchmark")
    print("=" * 70)
    parser = argparse.ArgumentParser(description='Geometric Multigrid Solver')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print detailed level information')
    args = parser.parse_args()
    
    # Test problems (nx, ny, nz, levels, max_iterations, tolerance, pre_iter, post_iter, bottom_iter)
    problems = [
        (16, 16, 16, 3, 10, 1e-3, 6, 6, 10),   # Small problem
        # (24, 24, 8, 3, 12, 1e-4, 6, 6, 100),    # Rectangular grid, moderate z
        # (32, 32, 16, 4, 20, 1e-5, 8, 8, 150),   # Medium problem, more levels
        # (32, 32, 32, 4, 20, 1e-6, 6, 6, 100),   # Medium cube
        # (48, 24, 12, 4, 25, 1e-6, 8, 8, 200),   # Non-cube, more levels
        # (64, 32, 8, 4, 30, 1e-7, 10, 10, 200),  # Large, flat in z
        # (64, 64, 16, 5, 40, 1e-8, 10, 10, 250), # Large, more levels
        # (128, 64, 8, 5, 50, 1e-8, 12, 12, 300), # Very large, flat in z
        # (128, 128, 32, 6, 60, 1e-9, 12, 12, 400), # Huge, may be slow
    ]
    
    results = []
    
    for nx, ny, nz, levels, max_iter, tolerance, pre_iter, post_iter, bottom_iter in problems:
        try:
            residual, iterations, total_time = benchmark_problem(nx, ny, nz, levels, args.verbose, tolerance, pre_iter, post_iter, bottom_iter, max_iter)
            results.append((nx, ny, nz, residual, iterations, total_time))
        except Exception as e:
            print(f"Failed: {e}")
            results.append((nx, ny, nz, float('inf'), 0, 0))
    
    # Summary
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    print(f"{'Grid Size':<15} {'Residual':<12} {'Tolerance':<12} {'Iterations':<10} {'Time (s)':<10} {'Converged':<10}")
    print("-" * 70)
    
    for problem_idx, (nx, ny, nz, residual, iterations, total_time) in enumerate(results):
        grid_size = f"{nx}x{ny}x{nz}"
        tolerance = problems[problem_idx][5]
        converged = "Yes" if residual < tolerance else "No"
        print(f"{grid_size:<15} {residual:<12.2e} {tolerance:<12.2e} {iterations:<10} {total_time:<10.4f} {converged:<10}")

if __name__ == "__main__":
    main()
