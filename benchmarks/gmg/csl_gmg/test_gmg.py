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

def test_down_cycle_only(nx, ny, nz, num_levels, tolerance, pre_iter, post_iter, bottom_iter):
    """Test only the down cycle using the existing only_down_cycle function"""
    
    print("=" * 60)
    print("GMG Down Cycle Test")
    print("=" * 60)
    
    print(f"Grid size: {nx}x{ny}x{nz}")
    print(f"Levels: {num_levels}")
    print()
    
    # Create Python reference solver
    solver = SimpleGMG(nx, ny, nz, num_levels, verbose=False, tolerance=tolerance, pre_iter=pre_iter, post_iter=post_iter, bottom_iter=bottom_iter)
    
    print("Performing down cycle using only_down_cycle()...")
    print()
    
    # Use the existing only_down_cycle function
    solver.only_down_cycle()
    
    # Print residual norms at all levels after down cycle
    print("Residual norms at all levels after down cycle:")
    for level in range(num_levels):
        # Compute residual at each level
        r_level = solver.grids[level]['r']
        residual_norm = np.dot(r_level.flatten(), r_level.flatten())
        grid_shape = solver.grids[level]['r'].shape
        total_points = np.prod(grid_shape)
        print(f"  Level {level}: residual norm: {residual_norm:.6e} (grid: {grid_shape}, points: {total_points})")
    
    print("=" * 60)
    print("Down cycle test completed!")
    print("=" * 60)


if __name__ == "__main__":
    # test_stencil_coefficients()
    # Test parameters
    nx, ny, nz = 16, 16, 16
    num_levels = 5
    tolerance = 1e-6
    pre_iter = 6
    post_iter = 6
    bottom_iter = 100
    test_down_cycle_only(nx, ny, nz, num_levels, tolerance, pre_iter, post_iter, bottom_iter)


