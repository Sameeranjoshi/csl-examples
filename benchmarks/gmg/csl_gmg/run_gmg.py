#!/usr/bin/env python3
"""
Geometric Multigrid (GMG) solver using CSL
Host-side implementation that coordinates with device kernels
Demonstrates the Python implementation with various problem sizes
---Algorithm---
loop (pre smooth iter)
    x_new = x_old + alpha * (b- Ax)
end
residual = b - A*x_new
b_next = Restrict(residual)
---

The input data sizes are 3D cube shaped.
"""
import os
from typing import Optional
from pathlib import Path
import shutil
import subprocess
import random
import math
import numpy as np
import sys
import copy
from scipy.sparse import linalg as sparse_LA
sys.path.append(os.path.join(os.path.dirname(__file__), '..', "python_gmg"))
# from gmg import SimpleGMG
from gmgoscar import SimpleGMG as SimpleGMGOSCAR
from cmd_parser import parse_args, print_arguments
from util import (
    hwl_2_oned_colmajor,
    oned_to_hwl_colmajor,
    laplacian_modified,
    csr_7_pt_stencil,
)
# Import Cerebras SDK
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyOrder, MemcpyDataType
DTYPE = np.float32


def csl_compile_core(
    cslc: str, width: int, height: int, pe_length: int, blockSize: int, file_config: str, elf_dir: str, levels: int,
    fabric_width: int, fabric_height: int, core_fabric_offset_x: int, core_fabric_offset_y: int, use_precompile: bool, arch: Optional[str], 
    C0: int, C1: int, C2: int, C3: int, C4: int, C5: int, C6: int, C7: int, C8: int, 
    channels: int, width_west_buf: int, width_east_buf: int):
  if not use_precompile:
    args = []
    args.append(cslc) # command
    args.append(file_config)
    args.append(f"--fabric-dims={fabric_width},{fabric_height}")
    args.append(f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}")
    args.append(f"--params=width:{width},height:{height},MAX_ZDIM:{pe_length}, levels:{levels}")
    args.append(f"--params=BLOCK_SIZE:{blockSize}")
    args.append(f"--params=C0_ID:{C0}")
    args.append(f"--params=C1_ID:{C1}")
    args.append(f"--params=C2_ID:{C2}")
    args.append(f"--params=C3_ID:{C3}")
    args.append(f"--params=C4_ID:{C4}")
    args.append(f"--params=C5_ID:{C5}")
    args.append(f"--params=C6_ID:{C6}")
    args.append(f"--params=C7_ID:{C7}")
    args.append(f"--params=C8_ID:{C8}")

    args.append(f"-o={elf_dir}")
    if arch is not None:
      args.append(f"--arch={arch}")
    args.append("--memcpy")
    args.append(f"--channels={channels}")
    args.append(f"--width-west-buf={width_west_buf}")
    args.append(f"--width-east-buf={width_east_buf}")

    print(f"subprocess.check_call(args = {args}")
    subprocess.check_call(args)
  else:
    print("\tuse pre-compile ELFs")

def subsample_activenodes_only(input_array_2d, level=0):
    """
    Subsamples a 2D array (a single z-layer) based on a factor derived from the level.

    An element at index (x, y) is kept if:
    (x % factor == 0) and (y % factor == 0)

    Args:
        input_array_2d (np.ndarray): The 2D layer (nx, ny) to subsample.
        level (int): The current level of resolution, used to calculate the factor.

    Returns:
        np.ndarray: The subsampled 2D array.
    """
    # 1. Calculate the factor
    factor = 2**(level)

    # 2. Perform subsampling using NumPy slicing.
    # [::factor] selects every 'factor'-th element starting from index 0.
    # This precisely implements the (x % factor == 0) and (y % factor == 0) condition.
    subsampled_array = input_array_2d[::factor, ::factor]

    return subsampled_array

def calculate_fabric_dimensions(args, width, height, width_west_buf, width_east_buf):
  """Calculate fabric dimensions based on core size and buffers"""

  # fabric-offsets = 1,1
  fabric_offset_x = 1
  fabric_offset_y = 1
  # starting point of the core rectangle = (core_fabric_offset_x, core_fabric_offset_y)
  # memcpy framework requires 3 columns at the west of the core rectangle
  # memcpy framework requires 2 columns at the east of the core rectangle
  core_fabric_offset_x = fabric_offset_x + 3 + width_west_buf
  core_fabric_offset_y = fabric_offset_y
  # (min_fabric_width, min_fabric_height) is the minimal dimension to run the app
  min_fabric_width = (core_fabric_offset_x + width + 2 + 1 + width_east_buf)
  min_fabric_height = (core_fabric_offset_y + height + 1)

  fabric_width = 0
  fabric_height = 0
  if args.fabric_dims:
    w_str, h_str = args.fabric_dims.split(",")
    fabric_width = int(w_str)
    fabric_height = int(h_str)

  if fabric_width == 0 or fabric_height == 0:
    fabric_width = min_fabric_width
    fabric_height = min_fabric_height

  assert fabric_width >= min_fabric_width
  assert fabric_height >= min_fabric_height

  return fabric_width, fabric_height, core_fabric_offset_x, core_fabric_offset_y

def copy_data_h2d(height, width, zDim, memcpy_dtype, memcpy_order, simulator, 
                  symbol_u, symbol_f, symbol_stencil_coeff, device_solver):                  
    """Copy data from host to device"""

    LEVEL_ZERO = 0

    u_hwl = device_solver.grids[LEVEL_ZERO]['u']  # (nx, ny, nz) -> (height, width, zDim) - no transpose needed
    f_hwl = device_solver.grids[LEVEL_ZERO]['f']  # (nx, ny, nz) -> (height, width, zDim) - no transpose needed


    # Flatten the data    
    u_1d_level_0 = hwl_2_oned_colmajor(height, width, zDim, u_hwl, DTYPE)
    f_1d_level_0 = hwl_2_oned_colmajor(height, width, zDim, f_hwl, DTYPE)    

    # # Copy solution vector u
    simulator.memcpy_h2d(symbol_u, u_1d_level_0, 0, 0, width, height, zDim,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    # Copy right-hand side f
    simulator.memcpy_h2d(symbol_f, f_1d_level_0, 0, 0, width, height, zDim,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    # # Copy stencil coefficients
    # simulator.memcpy_h2d(symbol_stencil_coeff, stencil_coeff_1d, 0, 0, width, height, 7,
    #                      streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)

def copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_device):
    """Copy single data array from device to host with automatic reshaping to 3D format"""
    # Copy data from device to host
    data_1d = np.zeros(height * width * zDim, dtype=DTYPE)
    simulator.memcpy_d2h(data_1d, symbol_device, 0, 0, width, height, zDim,
                        streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)    
    # Reshape to 3D format (height, width, zDim) in column-major order
    data_3d = oned_to_hwl_colmajor(height, width, zDim, data_1d, DTYPE)
    
    return data_3d

def copy_scalar_d2h(memcpy_dtype, memcpy_order, simulator, symbol_device):
    """Copy a single scalar value from device to host
    
    Args:
        memcpy_dtype: Memory copy data type (e.g., MemcpyDataType.MEMCPY_32BIT)
        memcpy_order: Memory copy order (e.g., MemcpyOrder.COL_MAJOR)
        simulator: Device simulator
        symbol_device: Symbol for the scalar on device
        
    Returns:
        float: The scalar value
        
    Example:
        xi_value = copy_scalar_d2h(memcpy_dtype, memcpy_order, simulator, symbol_xi)
    """
    # Create a 1-element array to hold the scalar
    scalar_array = np.zeros(1, dtype=np.float32)
    
    # Copy from device to host (only from PE at position 0,0)
    simulator.memcpy_d2h(scalar_array, symbol_device, 0, 0, 1, 1, 1,
                        streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    
    # Return the scalar value
    return scalar_array[0]
    

def copy_data_d2h_3d(height, width, zDim, memcpy_dtype, memcpy_order, simulator, *symbols):
    """Copy multiple data arrays from device to host with automatic reshaping to 3D format
    
    Args:
        height, width, zDim: Grid dimensions
        memcpy_dtype, memcpy_order: Memory copy parameters
        simulator: Device simulator
        *symbols: Variable number of symbols to copy
        
    Returns:
        List of 3D arrays corresponding to each symbol
        
    Example:
        data1, data2 = copy_data_d2h_3d(h, w, z, dtype, order, sim, symbol1, symbol2)
    """
    results = []
    
    # Process each symbol
    for symbol in symbols:
        data_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol)
        results.append(data_3d)
    
    return results

def init_operator(device_solver, zDim, simulator, LEVEL_ID, down=True):
    hx = device_solver.grids[LEVEL_ID]['hx']
    hy = device_solver.grids[LEVEL_ID]['hy']
    hz = device_solver.grids[LEVEL_ID]['hz']
    simulator.launch("f_gmg_init", np.int16(zDim), np.uint16(LEVEL_ID), np.float32(hx), np.float32(hy), np.float32(hz), np.int16(1 if down else 0), nonblock=False)

def residual_operator(simulator, height, width, zDim, memcpy_dtype, memcpy_order, symbol_Au, symbol_r):

    print("\tStep 1: Apply operator(A*u) only on active PEs")
    # Au = A*u(laplacian)
    simulator.launch("f_apply_operator", nonblock=False)
    # get Au from device and print.
    Au_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_Au)
    print(f"Step 1: Au result(not-subsampled): \n {Au_3d[:, :, 0]}")
    # # Compute residual
    print("\tStep 2: Compute residual(b- Au) only on active PEs")
    # r = b- au
    simulator.launch("f_residual", nonblock=False)

# x_new = x_old + JACOBI_COEFF_PER_LEVEL * (b- Ax)
def jacobi_smoothing_operator(device_solver, simulator, iterations, current_level):

  for i in range(iterations):
    # Jacobi smoothing
    # print(f"Iteration {i+1}: Jacobi smoothing")
    # Based on gmgoscar.py formulation with omega=2/3 and diagonal=-2/(hx^2)-2/(hy^2)-2/(hz^2)
    # JACOBI_COEFF = omega / diagonal = (2/3) / (-2/(hx^2) - 2/(hy^2) - 2/(hz^2))
    # Simplified: -1.0 / (3.0 * (1/(hx^2) + 1/(hy^2) + 1/(hz^2)))
    hx = device_solver.grids[current_level]['hx']
    hy = device_solver.grids[current_level]['hy']
    hz = device_solver.grids[current_level]['hz']
    JACOBI_COEFF_PER_LEVEL = -1.0 / (3.0 * (1.0/(hx*hx) + 1.0/(hy*hy) + 1.0/(hz*hz)))

    simulator.launch("f_apply_operator", nonblock=False) # applyOp = A*u
    # x_new = x_old + JACOBI_COEFF_PER_LEVEL * (b- applyOp)
    # 1 here means loop only once, we perform the loop as of now on the host, need to move to device later.
    simulator.launch("f_jacobi_smooth", np.int16(1), np.float32(JACOBI_COEFF_PER_LEVEL), nonblock=False)

def restrict_operator(device_solver, simulator, zDim):
    print("\tStep *.1: Reduction top left pattern")
    simulator.launch("f_reduction_top_left_pattern", np.int16(zDim), nonblock=False)

    print("\tStep *.2: Divide restrict to get result")
    simulator.launch("f_restriction_division", np.int16(zDim), nonblock=False)

def interpolation_operator(device_solver, simulator, zDim, symbol_u, symbol_r, height, width, memcpy_dtype, memcpy_order, args, current_level):
   
    # before bcast value of 'u'  values and after
    # u_3d_before = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u)
    # print(f"\tDEBUG: x_coarse (subsampled): \n {subsample_activenodes_only(u_3d_before, current_level)[:, :, 0]}")

    print("\tStep *.1: Interpolation bcast from top left")
    simulator.launch("f_bcast_from_top_left", np.int16(zDim), nonblock=False)

    # print u after bcast
    # u_3d_after_bcast = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u)
    # print(f"\tDEBUG: P(x_coarse) (subsampled): \n {subsample_activenodes_only(u_3d_after_bcast, current_level)[:, :, 0]}")
    # print the smooth values
    # symbol_u_smooth = simulator.get_id("u_smooth")
    # smooth_u_3d = copy_data_d2h_single(height, width, zDim*args.levels, memcpy_dtype, memcpy_order, simulator, symbol_u_smooth)
    # print(f"\tDEBUG: x_smooth (subsampled): \n {subsample_activenodes_only(smooth_u_3d, current_level)[:, :, 0]}")
                

    print("\tStep *.2: Interpolation add")
    simulator.launch("f_interpolation_add", nonblock=False)
    


def process_single_level_down(device_solver, height, width, zDim, memcpy_dtype, memcpy_order, simulator, 
                        symbol_r, symbol_f, symbol_u, current_level, symbol_Au):
    """Process a single level of the GMG algorithm"""
    print(f"Step 0: Initialize GMG LEVEL_ID = {current_level}")
    init_operator(device_solver, zDim, simulator, current_level, down=True)
    
    print(f"Step 1: Jacobi smoothing operator LEVEL_ID = {current_level}")
    jacobi_smoothing_operator(device_solver, simulator, device_solver.PRE_SMOOTH_ITER, current_level)
    
    # # DEBUG: Print u after pre-smoothing
    # u_after_presmooth = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u)
    # print(f"\tDEBUG: U AFTER pre-smooth LEVEL_ID={current_level}: \n {subsample_activenodes_only(u_after_presmooth, current_level)[:, :, 0]}")
    
    print(f"Step 2: Compute residual LEVEL_ID = {current_level}")
    residual_operator(simulator, height, width, zDim, memcpy_dtype, memcpy_order, symbol_Au, symbol_r)
    residual_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_r)
    
    print(f"Step 3: Restriction LEVEL_ID = {current_level}")
    restrict_operator(device_solver, simulator, zDim)
    
    # Copy restriction result from device to host
    restrict_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_f)
    
    rho_level = device_solver.calculate_rho(residual_3d)
    
    return residual_3d, restrict_3d, rho_level

def process_coarse_level(device_solver, height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_f, symbol_r, symbol_u, current_level, symbol_Au):
    """Process the coarse level"""
    print(f"Step 0: Initialize GMG LEVEL_ID = {current_level}")
    init_operator(device_solver, zDim, simulator, current_level, down=True)


    print(f"Step 1: Jacobi smoothing operator LEVEL_ID = {current_level}")
    jacobi_smoothing_operator(device_solver, simulator, device_solver.BOTTOM_SOLVER_ITER, current_level)

    # Copy coarse level solution back from device
    print(f"Step 1.5: Copy coarse level solution from device")
    u_coarse_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u)

    # residual extra step don't count in time.
    print(f"Step 2: Compute residual LEVEL_ID = {current_level}")
    residual_operator(simulator, height, width, zDim, memcpy_dtype, memcpy_order, symbol_Au, symbol_r)
    residual_3d_coarse = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_r)
    rho_level_coarse = device_solver.calculate_rho(residual_3d_coarse)
    
    return u_coarse_3d, residual_3d_coarse, rho_level_coarse

def process_single_level_up(device_solver, height, width, zDim, memcpy_dtype, memcpy_order, simulator, 
                           symbol_u, symbol_r, current_level, args, symbol_f, symbol_Au):
    """Process a single level of the GMG algorithm during up cycle (interpolation)"""
    print(f"Step 0: Initialize GMG LEVEL_ID = {current_level}")
    init_operator(device_solver, zDim, simulator, current_level, down=False)
    
    print(f"Step 1: Interpolation LEVEL_ID = {current_level}")
    interpolation_operator(device_solver, simulator, zDim, symbol_u, symbol_r, height, width, memcpy_dtype, memcpy_order, args, current_level)
    
    # # Copy interpolation result from device to host
    correction_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u)
    
    # print(f"Step 2: Jacobi smoothing operator LEVEL_ID = {current_level}")
    # jacobi_smoothing_operator(device_solver, simulator, device_solver.POST_SMOOTH_ITER, current_level)

    # print u after post-smooth
    # u_3d_after_postsmooth = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u)
    # print(f"\tDEBUG: U AFTER JACOBI LEVEL_ID={current_level}: \n {subsample_activenodes_only(u_3d_after_postsmooth, current_level)[:, :, 0]}")
    # print correction_3d
    # don't subsample
    print(f"Step 2.1: Correction result(not-subsampled): \n {correction_3d[:, :, 0]}")
    print(f"Step 2: Correction result(subsampled): \n {subsample_activenodes_only(correction_3d, current_level)[:, :, 0]}")
########################################################################################
    # print A, U , Au
    symbol_stencil_coeff = simulator.get_id("stencil_coeff")
    symbol_u = simulator.get_id("u")
    symbol_Au = simulator.get_id("Au")
    symbol_r = simulator.get_id("r")
    # stencil_coeff_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_stencil_coeff)
    # u_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u)
    # Au_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_Au)
    # f_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_f)
    # r_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_r)
    # print(f"BEFORE residual A, U, Au result(not-subsampled): \n {stencil_coeff_3d[:, :, 0]} \n {u_3d[:, :, 0]} \n {Au_3d[:, :, 0]}, \n {f_3d[:, :, 0]}, \n {r_3d[:, :, 0]}")    
########################################################################################

    print(f"Step 3: Compute residual LEVEL_ID = {current_level}")
    residual_operator(simulator, height, width, zDim, memcpy_dtype, memcpy_order, symbol_Au, symbol_r)
########################################################################################
    # print A, U , Au
    
    stencil_coeff_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_stencil_coeff)
    u_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u)
    Au_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_Au)
    f_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_f)
    r_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_r)
    print(f"AFTER residual stencil, U, Au, f, r:  \n {stencil_coeff_3d[:, :, 0]} \n {u_3d[:, :, 0]} \n {Au_3d[:, :, 0]}, \n {f_3d[:, :, 0]}, \n {r_3d[:, :, 0]}")    
########################################################################################

    residual_3d = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_r)
    # print(f"Step 3: Residual result(subsampled): \n {subsample_activenodes_only(residual_3d, current_level)[:, :, 0]}")
    rho_level = device_solver.calculate_rho(residual_3d)
    print(f"Step 3: Residual result(rho): \n {rho_level}")
    
    return correction_3d, residual_3d, rho_level

def gmg_algorithm(device_solver, height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_f, symbol_r, symbol_u, args, symbol_Au):
    """Main GMG algorithm"""
    print("=" * 50)
    print(f"\nHardware : grid size {height}x{width}")
    print("=" * 50)    
    device_solver._print_level_info()
    device_solver._print_initialization_info()

    # DOWN CYCLE: Process each level using a loop
    for i in range(args.levels - 1):
        current_level = i
        print(f"\nDOWN CYCLE - Processing Level {i} (LEVEL_ID = {current_level})")
        
        # Process the current level
        residual_3d, restrict_3d, rho_level = process_single_level_down(
            device_solver, height, width, zDim, memcpy_dtype, memcpy_order, simulator,
            symbol_r, symbol_f, symbol_u, current_level, symbol_Au
        )
        device_solver.grids[current_level]['r'] = subsample_activenodes_only(residual_3d, current_level)
        device_solver.grids[current_level]['rho'] = rho_level
        device_solver.grids[current_level + 1]['f'] = subsample_activenodes_only(restrict_3d, current_level + 1)
    
    # solve coarse level
    print(f"\nDOWN CYCLE - Processing Level {args.levels - 1}(COARSE) (LEVEL_ID = {args.levels - 1})")
    u_coarse_3d, residual_3d_coarse, rho_level_coarse = process_coarse_level(device_solver, height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_f, symbol_r, symbol_u, args.levels - 1, symbol_Au)
    device_solver.grids[args.levels - 1]['u'] = subsample_activenodes_only(u_coarse_3d, args.levels - 1)
    device_solver.grids[args.levels - 1]['r'] = subsample_activenodes_only(residual_3d_coarse, args.levels - 1)
    device_solver.grids[args.levels - 1]['rho'] = rho_level_coarse
    device_solver.grids[args.levels - 1]['rho_up'] = rho_level_coarse

    # UP CYCLE: Process each level in reverse order (interpolation)
    for i in reversed(range(args.levels - 1)):
        current_level = i
        print(f"\nUP CYCLE - Processing Level {i} (LEVEL_ID = {current_level})")

        # Process the current level (interpolation + post-smoothing)
        correction_3d, residual_3d, rho_level = process_single_level_up(
            device_solver, height, width, zDim, memcpy_dtype, memcpy_order, simulator,
            symbol_u, symbol_r, current_level, args, symbol_f, symbol_Au
        )
        # print(f"Step 1: x = x_smooth + P(u_coarse) (3D): \n {correction_3d[:, :, 0]}")
        device_solver.grids[current_level]['u'] = subsample_activenodes_only(correction_3d, current_level)
        device_solver.grids[current_level]['r'] = subsample_activenodes_only(residual_3d, current_level)
        device_solver.grids[current_level]['rho_up'] = rho_level


def get_exponent(A: int) -> int:
    return int(math.log2(A))

def main():
    """Main function"""
    np.random.seed(2)

    # parse arguments
    args, logs_dir = parse_args()

    # Hardware-specific parameters
    cslc = "cslc"
    compile_only = args.compile_only
    fabric_dims = args.fabric_dims

    # Input problem-specific parameters
    height = args.m
    width = args.n
    pe_length = args.k  # This seems like the max size the 3rd dimension can be
    zDim = args.zDim
    max_possible_levels = get_exponent(height)  # pick any randome dimension
    if (args.levels > max_possible_levels):
        raise ValueError(f"ERROR: args.levels ({args.levels}) is greater than max_possible_levels ({max_possible_levels}) levels for this problem size. We stop at 2x2 coarse grid size.")

    # perform argument checks
    assert pe_length >= 2, "the maximum size of z must be greater than 1"
    assert zDim >= 2, "the minimum size of zDim must be greater than 1"
    assert zDim <= pe_length, "[0, zDim) cannot exceed the storage"

    # Host GMG for validation
    #########################################################
    # Initialize solver data.
    host_solver = SimpleGMGOSCAR(width, height, zDim, args.levels, args.verbose, args.tolerance, args.pre_iter, args.post_iter, args.bottom_iter)
    device_solver = copy.deepcopy(host_solver)    # Before solving make sure to make a deep copy as python might modify the data in the original object.
    host_solver.solve_iterative(args.max_ite)
    # host_solver.only_down_cycle()
    # host_solver.solve_coarse()
    # host_solver.only_up_cycle(args.levels - 2)

    #########################################################

    # Device side
    # Calculate fabric dimensions
    # TODO: Add inside csl_compile_core
    # fabric_width, fabric_height, core_fabric_offset_x, core_fabric_offset_y = calculate_fabric_dimensions(args, width, height, args.width_west_buf, args.width_east_buf)
    
    # # Compile the kernel
    # layout_file = "./src/layout_gmg.csl"
    # # This is used when user doesn't use a compile command first.
    # csl_compile_core(=cslc, width=width, height=height, pe_length=pe_length, blockSize=args.blockSize, file_config=layout_file,
    #     elf_dir=logs_dir, fabric_width=fabric_width, fabric_height=fabric_height, core_fabric_offset_x=core_fabric_offset_x, core_fabric_offset_y=core_fabric_offset_y, use_precompile=args.run_only,
    #     arch=args.arch if args.arch else "wse2", C0=0, C1=1, C2=2, C3=3, C4=4, C5=5, C6=6, C7=7, C8=8, channels=args.channels, 
    #     width_west_buf=args.width_west_buf, width_east_buf=args.width_east_buf, levels=args.levels)
    
    ### Initialize data
    print("Initializing data...")
    # Create simulator
    memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
    memcpy_order = MemcpyOrder.COL_MAJOR
    simulator = SdkRuntime(logs_dir, cmaddr=args.cmaddr)
    symbol_u = simulator.get_id("u")  # solution vector
    symbol_f = simulator.get_id("f")  # right-hand side
    symbol_r = simulator.get_id("r")  # residual (reuse x for now)
    symbol_stencil_coeff = simulator.get_id("stencil_coeff")
    symbol_Au = simulator.get_id("Au")
    symbol_b_next = simulator.get_id("b_next")
    symbol_xi = simulator.get_id("xi")
    simulator.load()
    simulator.run()

    ### Run GMG algorithm on device
    # Copy data to device
    print("Copying data to device...")
    copy_data_h2d(height, width, zDim, memcpy_dtype, memcpy_order, simulator,
                  symbol_u, symbol_f, symbol_stencil_coeff, device_solver)
    # Run GMG algorithm
    print("Running GMG algorithm...")
    gmg_algorithm(device_solver, height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_f, symbol_r, symbol_u, args, symbol_Au)
    print("Copying results back...")

    # clean up
    simulator.stop()

    ### Verification
    # Extract host data for all levels using a loop
    print("Host rho")
    for level_index in range(args.levels):  # levels 0, 1, 2, 3
            print(f"Level {level_index}: {host_solver.grids[level_index]['rho']:.6e}")
    for level_index in reversed(range(args.levels)):  # Print from bottom (coarse) to top (fine)
            print(f"Level {level_index}: {host_solver.grids[level_index]['rho_up']:.6e}")


    print("Device rho")
    for level_index in range(args.levels):  # levels 0, 1, 2, 3
            print(f"Level {level_index}: {device_solver.grids[level_index]['rho']:.6e}")
    for level_index in reversed(range(args.levels)):  # Print from bottom (coarse) to top (fine)
            print(f"Level {level_index}: {device_solver.grids[level_index]['rho_up']:.6e}")

    # Down check b_next
    for level_index in range(args.levels):  # levels 0, 1, 2, 3
        # np.testing.assert_allclose(device_solver.grids[level_index]['r'], host_solver.grids[level_index]['r'], atol=1e-5, rtol=1e-5)
        if level_index != args.levels - 1:  # There is no restrict at coarse level
            np.testing.assert_allclose(device_solver.grids[level_index + 1]['f'], host_solver.grids[level_index + 1]['f'], atol=1e-5, rtol=1e-5)

    # up check interpolated values
    for level_index in reversed(range(args.levels)):  # Print from bottom (coarse) to top (fine)
        if level_index >= 0:  #
            print(f"Level = {level_index}")
            np.testing.assert_allclose(device_solver.grids[level_index]['u'], host_solver.grids[level_index]['u'], atol=1e-5, rtol=1e-5)

    print("SUCCESS!")

if __name__ == "__main__":
    main()
