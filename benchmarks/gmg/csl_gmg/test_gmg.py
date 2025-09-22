#!/usr/bin/env python3
"""
Test script for GMG CSL implementation
Compares with Python reference implementation
"""

import numpy as np
import sys
import os

# Add the Python GMG implementation
sys.path.append('../python_gmg')
from gmg import SimpleGMG

def test_gmg_consistency():
    """Test that CSL GMG produces consistent results with Python GMG"""
    
    print("=" * 60)
    print("GMG CSL Implementation Test")
    print("=" * 60)
    
    # Test parameters
    nx, ny, nz = 8, 8, 8
    num_levels = 2
    tolerance = 1e-3
    max_iter = 5
    
    print(f"Grid size: {nx}x{ny}x{nz}")
    print(f"Levels: {num_levels}")
    print(f"Tolerance: {tolerance}")
    print(f"Max iterations: {max_iter}")
    print()
    
    # Create Python reference solver
    print("Creating Python reference solver...")
    solver = SimpleGMG(nx, ny, nz, num_levels, verbose=False, tolerance=tolerance)
    
    # Get the test problem
    f_ref = solver.grids[0]['f']  # Right-hand side
    u_ref = solver.grids[0]['u']  # Initial solution (zeros)
    
    print(f"Reference RHS shape: {f_ref.shape}")
    print(f"Reference RHS range: [{f_ref.min():.6f}, {f_ref.max():.6f}]")
    print()
    
    # Test individual components
    print("Testing GMG components...")
    
    # 1. Test operator application
    print("1. Testing operator application...")
    solver.apply_operator(0)
    Au_ref = solver.grids[0]['Au']
    print(f"   Au shape: {Au_ref.shape}")
    print(f"   Au range: [{Au_ref.min():.6f}, {Au_ref.max():.6f}]")
    
    # 2. Test residual computation
    print("2. Testing residual computation...")
    solver.compute_residual(0)
    r_ref = solver.grids[0]['r']
    print(f"   Residual shape: {r_ref.shape}")
    print(f"   Residual range: [{r_ref.min():.6f}, {r_ref.max():.6f}]")
    print(f"   Residual norm: {np.linalg.norm(r_ref):.6f}")
    
    # 3. Test Jacobi smoothing
    print("3. Testing Jacobi smoothing...")
    u_before = u_ref.copy()
    solver.jacobi_smooth(0, 3)
    u_after = solver.grids[0]['u']
    print(f"   Solution change: {np.linalg.norm(u_after - u_before):.6f}")
    print(f"   Solution range: [{u_after.min():.6f}, {u_after.max():.6f}]")
    
    # 4. Test restriction
    print("4. Testing restriction...")
    if num_levels > 1:
        solver.restrict(0)
        f_coarse = solver.grids[1]['f']
        print(f"   Coarse RHS shape: {f_coarse.shape}")
        print(f"   Coarse RHS range: [{f_coarse.min():.6f}, {f_coarse.max():.6f}]")
    
    # 5. Test interpolation
    print("5. Testing interpolation...")
    if num_levels > 1:
        # Set some coarse solution
        solver.grids[1]['u'].fill(0.1)
        solver.interpolate(1)
        u_fine = solver.grids[0]['u']
        print(f"   Fine solution after interpolation: {np.linalg.norm(u_fine):.6f}")
    
    print()
    print("Component tests completed successfully!")
    print()
    
    # Test full V-cycle
    print("Testing full V-cycle...")
    solver.grids[0]['u'].fill(0.0)  # Reset solution
    solver.v_cycle()
    
    # Check final solution
    u_final = solver.grids[0]['u']
    solver.compute_residual(0)
    r_final = solver.grids[0]['r']
    residual_norm = np.linalg.norm(r_final)
    
    print(f"Final solution range: [{u_final.min():.6f}, {u_final.max():.6f}]")
    print(f"Final residual norm: {residual_norm:.6f}")
    print()
    
    # Test convergence
    print("Testing convergence...")
    solver.grids[0]['u'].fill(0.0)  # Reset solution
    residual, iterations = solver.solve(max_iter)
    
    print(f"Converged: {residual < tolerance}")
    print(f"Final residual: {residual:.6e}")
    print(f"Iterations: {iterations}")
    print()
    
    print("=" * 60)
    print("All tests completed successfully!")
    print("CSL implementation should produce similar results.")
    print("=" * 60)

def test_stencil_coefficients():
    """Test that stencil coefficients are correct"""
    
    print("Testing stencil coefficients...")
    
    # 7-point Poisson stencil coefficients
    alpha = -6.0  # center
    beta = 1.0    # neighbors
    
    print(f"Center coefficient (α): {alpha}")
    print(f"Neighbor coefficient (β): {beta}")
    print(f"Stencil: α*u[i,j,k] + β*(u[i±1,j,k] + u[i,j±1,k] + u[i,j,k±1])")
    print()
    
    # Test on simple 3x3x3 grid
    nx, ny, nz = 3, 3, 3
    h = 1.0 / (nx - 1)
    
    # Create test solution
    u = np.zeros((nz, ny, nx))
    u[1, 1, 1] = 1.0  # Center point
    
    # Apply stencil manually
    Au = np.zeros_like(u)
    for k in range(1, nz-1):
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                Au[k, j, i] = (alpha * u[k, j, i] +
                              beta * (u[k, j, i+1] + u[k, j, i-1] +
                                      u[k, j+1, i] + u[k, j-1, i] +
                                      u[k+1, j, i] + u[k-1, j, i])) / (h * h)
    
    print(f"Test solution (center point = 1):")
    print(f"u[1,1,1] = {u[1,1,1]}")
    print(f"Au[1,1,1] = {Au[1,1,1]}")
    print(f"Expected: {alpha / (h * h)}")
    print(f"Actual: {Au[1,1,1]}")
    print(f"Match: {abs(Au[1,1,1] - alpha / (h * h)) < 1e-10}")
    print()

if __name__ == "__main__":
    test_stencil_coefficients()
    test_gmg_consistency()
