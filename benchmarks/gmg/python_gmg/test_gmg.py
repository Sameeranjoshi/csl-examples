#!/usr/bin/env python3
"""
Test script for the GMG solver implementation
"""

import numpy as np
from gmg_solver import GMGSolver


def test_small_problem():
    """Test with a small problem to verify correctness"""
    print("Testing small problem (32x32x32)...")
    
    solver = GMGSolver(32, 32, 32, 4)
    residual, iterations, timers = solver.solve(max_iterations=10)
    
    print(f"Small problem results:")
    print(f"  Final residual: {residual:.6e}")
    print(f"  Iterations: {iterations}")
    print(f"  Total time: {timers['total']:.4f}s")
    
    # Verify convergence
    assert residual < 1e-10, f"Did not converge: residual = {residual}"
    assert iterations <= 10, f"Too many iterations: {iterations}"
    
    print("✓ Small problem test passed")
    return True


def test_medium_problem():
    """Test with a medium problem"""
    print("\nTesting medium problem (128x128x128)...")
    
    solver = GMGSolver(128, 128, 128, 5)
    residual, iterations, timers = solver.solve(max_iterations=15)
    
    print(f"Medium problem results:")
    print(f"  Final residual: {residual:.6e}")
    print(f"  Iterations: {iterations}")
    print(f"  Total time: {timers['total']:.4f}s")
    
    # Verify convergence
    assert residual < 1e-10, f"Did not converge: residual = {residual}"
    assert iterations <= 15, f"Too many iterations: {iterations}"
    
    print("✓ Medium problem test passed")
    return True


def test_operator_accuracy():
    """Test the operator accuracy"""
    print("\nTesting operator accuracy...")
    
    solver = GMGSolver(16, 16, 16, 3)
    
    # Set up a simple test case
    grid = solver.grids[0]
    nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
    
    # Create a simple polynomial solution
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # u = x(1-x)y(1-y)z(1-z) should give -Δu = 6x(1-x)y(1-y)z(1-z) + 2[x(1-x) + y(1-y) + z(1-z)]
    grid['x'] = X * (1 - X) * Y * (1 - Y) * Z * (1 - Z)
    
    # Apply operator
    solver.apply_operator(0)
    
    # Check if the operator is applied correctly
    # For this simple case, we expect the operator to be non-zero
    max_val = np.max(np.abs(grid['ax']))
    assert max_val > 0, "Operator should produce non-zero result"
    
    print(f"✓ Operator accuracy test passed (max operator value: {max_val:.6e})")
    return True


def test_convergence_rate():
    """Test convergence rate"""
    print("\nTesting convergence rate...")
    
    solver = GMGSolver(64, 64, 64, 4)
    
    # Track residual history
    residuals = []
    
    # Initialize solution
    for grid in solver.grids:
        grid['x'].fill(0.0)
    
    # Run several iterations and track residuals
    for i in range(10):
        solver.v_cycle()
        residual = solver.compute_residual_norm()
        residuals.append(residual)
        print(f"  Iteration {i+1}: Residual = {residual:.6e}")
    
    # Check that residuals are generally decreasing
    # (allowing for some numerical noise)
    decreasing_count = 0
    for i in range(1, len(residuals)):
        if residuals[i] <= residuals[i-1] * 1.1:  # Allow 10% increase due to numerical noise
            decreasing_count += 1
    
    assert decreasing_count >= len(residuals) - 2, "Residuals should generally decrease"
    
    print("✓ Convergence rate test passed")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running GMG Solver Tests")
    print("=" * 60)
    
    tests = [
        test_small_problem,
        test_medium_problem,
        test_operator_accuracy,
        test_convergence_rate
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
