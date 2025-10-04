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
from gmg import SimpleGMG
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

def copy_data_h2d(u_1d, f_1d, stencil_coeff, height, width, zDim, memcpy_dtype, memcpy_order, simulator, 
                  symbol_u, symbol_f, symbol_stencil_coeff):
    """Copy data from host to device"""
    # # Copy solution vector u
    simulator.memcpy_h2d(symbol_u, u_1d, 0, 0, width, height, zDim,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    # Copy right-hand side f
    simulator.memcpy_h2d(symbol_f, f_1d, 0, 0, width, height, zDim,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    # Copy stencil coefficients
    simulator.memcpy_h2d(symbol_stencil_coeff, stencil_coeff, 0, 0, width, height, 7,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)

def copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_device):
    """Copy single data array from device to host with automatic reshaping to 3D format"""
    # Copy data from device to host
    data_1d = np.zeros(height * width * zDim, dtype=DTYPE)
    simulator.memcpy_d2h(data_1d, symbol_device, 0, 0, width, height, zDim,
                        streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)    
    # Reshape to 3D format (height, width, zDim) in column-major order
    data_3d = oned_to_hwl_colmajor(height, width, zDim, data_1d, DTYPE)
    
    return data_3d

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

def init_operator(device_solver, zDim, simulator, LEVEL_ID):
    h_level = device_solver.grids[LEVEL_ID]['h']
    simulator.launch("f_gmg_init", np.int16(zDim), np.uint16(LEVEL_ID), np.float32(h_level), nonblock=False)

def residual_operator(simulator):

    print("Step 1: Apply operator")
    # Au = A*u(laplacian)
    simulator.launch("f_apply_operator", nonblock=False)
    # # Compute residual
    print("Step 2: Compute residual")
    # r = b- au
    simulator.launch("f_residual", nonblock=False)

# x_new = x_old + JACOBI_COEFF_PER_LEVEL * (b- Ax)
def jacobi_smoothing_operator(device_solver, simulator):

  for i in range(device_solver.PRE_SMOOTH_ITER):
    # Jacobi smoothing
    print(f"Iteration {i+1}: Jacobi smoothing")
    JACOBI_COEFF_PER_LEVEL = -1.0 * (device_solver.grids[0]['h'] * device_solver.grids[0]['h']) / 12.0;

    simulator.launch("f_apply_operator", nonblock=False) # applyOp = A*u
    # x_new = x_old + JACOBI_COEFF_PER_LEVEL * (b- applyOp)
    simulator.launch("f_jacobi_smooth", np.int16(1), np.float32(JACOBI_COEFF_PER_LEVEL), nonblock=False)

def restrict_operator(device_solver, simulator, zDim):
    print("Step 3: Reduction top left pattern")
    simulator.launch("f_reduction_top_left_pattern", np.int16(zDim), nonblock=False)

    print("Step 4: Divide restrict to get result")
    simulator.launch("f_restriction_division", np.int16(zDim), nonblock=False)


def gmg_algorithm(device_solver, height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u, symbol_f, symbol_r, symbol_b_next, symbol_stencil_coeff, LEVEL_ID):
    """Main GMG algorithm"""
    print("=" * 50)
    print(f"\nHardware : grid size {height}x{width}")
    print("=" * 50)    
    device_solver._print_level_info()
    device_solver._print_initialization_info()

#########################################################
    # Initialize GMG
    print(f"Step 0: Initialize GMG LEVEL_ID = {LEVEL_ID}")
    init_operator(device_solver, zDim, simulator, LEVEL_ID)
    # Jacobi smoothing
    print(f"Step 1: Jacobi smoothing operator LEVEL_ID = {LEVEL_ID}")
    jacobi_smoothing_operator(device_solver, simulator)
    # Compute residual
    print(f"Step 2: Compute residual LEVEL_ID = {LEVEL_ID}")
    residual_operator(simulator)

    # copy residual at level 0
    residual_3d_first = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_r)
    # print(f"residual_3d_level0 shape: {residual_3d_first[:, :, 0]}")

    # # Restriction (for now, just copy)
    print(f"Step 5: Restriction LEVEL_ID = {LEVEL_ID}")
    restrict_operator(device_solver, simulator, zDim)

    # copy residual at level 1
    restrict_3d_first = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_f)
    # print(f"restrict_3d_first shape: {restrict_3d_first[:, :, 0]}")
    
  #########################################################
    # Initialize GMG
    print(f"Step 0: Initialize GMG LEVEL_ID = {LEVEL_ID +1 }")
    init_operator(device_solver, zDim, simulator, LEVEL_ID + 1)
    # Jacobi smoothing
    print(f"Step 1: Jacobi smoothing operator LEVEL_ID = {LEVEL_ID + 1}")
    jacobi_smoothing_operator(device_solver, simulator)
    # Compute residual
    print(f"Step 2: Compute residual LEVEL_ID = {LEVEL_ID + 1}")
    residual_operator(simulator)

    # get the residual by memcpy_d2h before changed by restrict operator
    residual_3d_second = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_r)
    # print(f"residual_3d_level1 shape: {residual_3d_second[:, :, 0]}")

    # # Restriction (for now, just copy)
    print(f"Step 5: Restriction LEVEL_ID = {LEVEL_ID + 1}")
    restrict_operator(device_solver, simulator, zDim)
    restrict_3d_second = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_f)

#########################################################

    # # level 2
    print(f"Step 0: Initialize GMG LEVEL_ID = {LEVEL_ID + 2}")
    init_operator(device_solver, zDim, simulator, LEVEL_ID + 2)
    # Jacobi smoothing
    print(f"Step 1: Jacobi smoothing operator LEVEL_ID = {LEVEL_ID + 2}")
    jacobi_smoothing_operator(device_solver, simulator)
    # Compute residual
    print(f"Step 2: Compute residual LEVEL_ID = {LEVEL_ID + 2}")
    residual_operator(simulator)
    # get the residual by memcpy_d2h before changed by restrict operator
    residual_3d_third = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_r)
    # print(f"residual_3d_level2 shape: {residual_3d_third[:, :, 0]}")
    # # Restriction (for now, just copy)
    print(f"Step 5: Restriction LEVEL_ID = {LEVEL_ID + 2}")
    restrict_operator(device_solver, simulator, zDim)
    restrict_3d_third = copy_data_d2h_single(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_f)

#########################################################
    # # Interpolation (for now, just copy)
    # print("Step 6: Interpolation")
    # simulator.launch("f_interpolate", nonblock=False)
    
    # # Add correction
    # print("Step 7: Add correction")
    # simulator.launch("f_add_correction", nonblock=False)
    
    # return residual_norm[0]
    return residual_3d_first, restrict_3d_first, residual_3d_second, restrict_3d_second, residual_3d_third, restrict_3d_third

def main():
    """Main function"""
    np.random.seed(2)
    LEVEL_ID = 0

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

    # perform argument checks
    assert pe_length >= 2, "the maximum size of z must be greater than 1"
    assert zDim >= 2, "the minimum size of zDim must be greater than 1"
    assert zDim <= pe_length, "[0, zDim) cannot exceed the storage"

    # Initialize solver data.
    host_solver = SimpleGMG(width, height, zDim, args.levels, args.verbose, args.tolerance, args.pre_iter, args.post_iter, args.bottom_iter)
    device_solver = copy.deepcopy(host_solver)    # Before solving make sure to make a deep copy as python might modify the data in the original object.


    # Host GMG for validation
    #########################################################
    # host_residual, host_iterations = host_solver.solve(args.max_ite)
    # level 0
    host_solver.jacobi_smooth(LEVEL_ID, args.pre_iter)
    host_solver.compute_residual(LEVEL_ID)
    host_solver.restrict(LEVEL_ID)
    host_solver.grids[LEVEL_ID + 1]['u'].fill(0.0)

    # # level 1
    host_solver.jacobi_smooth(LEVEL_ID + 1, args.pre_iter)
    host_solver.compute_residual(LEVEL_ID + 1)
    host_solver.restrict(LEVEL_ID + 1)
    host_solver.grids[LEVEL_ID + 2]['u'].fill(0.0)

    # # level 2
    host_solver.jacobi_smooth(LEVEL_ID + 2, args.pre_iter)
    host_solver.compute_residual(LEVEL_ID + 2)
    host_solver.restrict(LEVEL_ID + 2)
    host_solver.grids[LEVEL_ID + 3]['u'].fill(0.0)

    first_smooth_u = host_solver.grids[LEVEL_ID]['u']
    first_residual = host_solver.grids[LEVEL_ID]['r']
    first_b_next = host_solver.grids[LEVEL_ID + 1]['f']

    second_smooth_u = host_solver.grids[LEVEL_ID + 1]['u']
    second_residual = host_solver.grids[LEVEL_ID + 1]['r']
    second_b_next = host_solver.grids[LEVEL_ID + 2]['f']

    # third
    third_smooth_u = host_solver.grids[LEVEL_ID + 2]['u']
    third_residual = host_solver.grids[LEVEL_ID + 2]['r']
    third_b_next = host_solver.grids[LEVEL_ID + 3]['f']

    # print values at level 0 and level 1 for residual
    # print(f"first_residual shape: {first_residual[:, :, 0]}")
    

    # # Use hop-based laplacian to match WSE implementation
    # # Active PEs are determined by factor, neighbors are immediate (hops=1)
    # factor = 2**LEVEL_ID  # factor = 2 (determines which PEs are active)
    # hops = factor  # immediate neighbors for 7-point stencil
    # laplacian_modified(stencil_coeff, zDim, u_hwl, f_hwl, hops=hops, factor=factor)   

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
    
    # Initialize data
    print("Initializing data...")
    # device_grid = device_solver.get_grids() # Already has data filled and shapes initialized.
    u_hwl = device_solver.grids[LEVEL_ID]['u']  # (nx, ny, nz) -> (height, width, zDim) - no transpose needed
    f_hwl = device_solver.grids[LEVEL_ID]['f']  # (nx, ny, nz) -> (height, width, zDim) - no transpose needed
    # order: {c_west, c_east, c_south, c_north, c_bottom, c_top, c_center}
    stencil_coeff = np.zeros((height, width, 7), dtype=DTYPE)  # 3D-7pt
    h = device_solver.grids[LEVEL_ID]['h']
    stencil_coeff[:, :, 0] = device_solver.BETA/h**2 # west(-1)
    stencil_coeff[:, :, 1] = device_solver.BETA/h**2 # east(-1)
    stencil_coeff[:, :, 2] = device_solver.BETA/h**2 # south(-1)
    stencil_coeff[:, :, 3] = device_solver.BETA/h**2 # north(-1)
    stencil_coeff[:, :, 4] = device_solver.BETA/(h**2) # bottom(-1)
    stencil_coeff[:, :, 5] = device_solver.BETA/(h**2) # top(-1)
    stencil_coeff[:, :, 6] = device_solver.ALPHA/(h**2)  # center (-6)

    # Flatten the data    
    u_1d_level_0 = hwl_2_oned_colmajor(height, width, zDim, u_hwl, DTYPE)
    f_1d_level_0 = hwl_2_oned_colmajor(height, width, zDim, f_hwl, DTYPE)    
    stencil_coeff_1d = hwl_2_oned_colmajor(height, width, 7, stencil_coeff, DTYPE)

    # Create simulator
    memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
    memcpy_order = MemcpyOrder.COL_MAJOR
    simulator = SdkRuntime(logs_dir, cmaddr=args.cmaddr, msg_level="INFO")

    # Get symbols
    symbol_u = simulator.get_id("u")  # solution vector
    symbol_f = simulator.get_id("f")  # right-hand side
    symbol_r = simulator.get_id("r")  # residual (reuse x for now)
    symbol_stencil_coeff = simulator.get_id("stencil_coeff")
    symbol_Au = simulator.get_id("Au")
    symbol_b_next = simulator.get_id("b_next")
    # host  = device
    #TODO: Leo
     # Load and run
    simulator.load()
    simulator.run()

    # Copy data to device
    print("Copying data to device...")
    copy_data_h2d(u_1d_level_0, f_1d_level_0, stencil_coeff_1d, height, width, zDim, memcpy_dtype, memcpy_order, simulator,
                  symbol_u, symbol_f, symbol_stencil_coeff)
    
    # Run GMG algorithm
    print("Running GMG algorithm...")
    residual_3d_first, restrict_3d_first, residual_3d_second, restrict_3d_second, residual_3d_third, restrict_3d_third = gmg_algorithm(device_solver, height, width, zDim, 
                                 memcpy_dtype, memcpy_order, simulator, symbol_u, symbol_f, symbol_r, symbol_b_next, symbol_stencil_coeff, LEVEL_ID
                                 )
   

    # Copy results back
    print("Copying results back...")
    # u_result_3d, b_next_3d = copy_data_d2h_3d(width, height, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u, symbol_b_next)

    # b needs special treatment as shapes reduce.
    residual_3d_first = subsample_activenodes_only(residual_3d_first, LEVEL_ID)
    residual_3d_second = subsample_activenodes_only(residual_3d_second, LEVEL_ID + 1)
    residual_3d_third = subsample_activenodes_only(residual_3d_third, LEVEL_ID + 2)
    restrict_3d_first = subsample_activenodes_only(restrict_3d_first, LEVEL_ID + 1) # Why +1 because we want active nodes in the next level 
    restrict_3d_second = subsample_activenodes_only(restrict_3d_second, LEVEL_ID + 2) # Why +2 because we want active nodes in the next level 
    restrict_3d_third = subsample_activenodes_only(restrict_3d_third, LEVEL_ID + 3)

 # print shapes of all arrays
    print(f"residual_3d_first shape: {residual_3d_first.shape}")
    print(f"restrict_3d_first shape: {restrict_3d_first.shape}")
    print(f"residual_3d_second shape: {residual_3d_second.shape}")
    print(f"restrict_3d_second shape: {restrict_3d_second.shape}")
    print(f"restrict_3d_third shape: {restrict_3d_third.shape}")
    print(f"residual_3d_third shape: {residual_3d_third.shape}")
    # print shapes of first_residual, 
    print(f"first_residual shape: {first_residual.shape}")
    print(f"first_b_next shape: {first_b_next.shape}")
    print(f"second_residual shape: {second_residual.shape}")
    print(f"second_b_next shape: {second_b_next.shape}")
    print(f"third_residual shape: {third_residual.shape}")
    print(f"third_b_next shape: {third_b_next.shape}")



    # clean up
    simulator.stop()

    # print residual and restrict at level 0 and level 1
    print(f"residual_3d_first shape: {residual_3d_first[:, :, 0]}")
    print(f"residual_3d_second shape: {residual_3d_second[:, :, 0]}")
    print(f"residual_3d_third shape: {residual_3d_third[:, :, 0]}")
    print(f"restrict_3d_first shape: {restrict_3d_first[:, :, 0]}")
    print(f"restrict_3d_second shape: {restrict_3d_second[:, :, 0]}")
    print(f"restrict_3d_third shape: {restrict_3d_third[:, :, 0]}")

    # Verify - no transposes needed since we're using consistent (nx, ny, nz) ordering
    # np.testing.assert_allclose(u_result_3d, first_smooth_u, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(residual_3d_first, first_residual, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(restrict_3d_first, first_b_next, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(residual_3d_second, second_residual, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(restrict_3d_second, second_b_next, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(residual_3d_third, third_residual, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(restrict_3d_third, third_b_next, atol=1e-5, rtol=1e-5)
    
    # print("u_result_3d (first z-layer, k=0):")
    # print(u_result_3d[:, :, 0])
    # print("\n")


    print("SUCCESS!")

if __name__ == "__main__":
    main()
