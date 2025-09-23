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
    laplacian,
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

# def copy_data_h2d(u_1d, f_1d, stencil_coeff, height, width, zDim, memcpy_dtype, simulator, 
#                   symbol_u, symbol_f, symbol_stencil_coeff):
#     """Copy data from host to device"""
#     # Copy solution vector u
#     simulator.memcpy_h2d(u_1d, symbol_u, 0, 0, height, width, zDim,
#                          streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)
    
#     # Copy right-hand side f
#     simulator.memcpy_h2d(f_1d, symbol_f, 0, 0, height, width, zDim,
#                          streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)
    
#     # Copy stencil coefficients
#     simulator.memcpy_h2d(stencil_coeff, symbol_stencil_coeff, 0, 0, height, width, 7,
#                          streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

# def copy_data_d2h(height, width, zDim, memcpy_dtype, simulator, symbol_u):
#     """Copy data from device to host"""
#     # Copy solution vector u
#     u_result = np.zeros((height, width, zDim), dtype=memcpy_dtype)
#     simulator.memcpy_d2h(u_result, symbol_u, 0, 0, height, width, zDim,
#                         streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
#     return u_result

# def gmg_algorithm(zDim, nx, ny, nz, omega, h, num_smooth_iter, memcpy_dtype, simulator, 
#                   symbol_u, symbol_f, symbol_r, symbol_Au, symbol_residual_norm):
#     """Main GMG algorithm"""
#     print(f"GMG Algorithm: grid size {nx}x{ny}x{nz}, zDim={zDim}")
#     print(f"Parameters: omega={omega}, h={h}, smooth_iter={num_smooth_iter}")
    
#     # Initialize GMG
#     print("Step 1: Initialize GMG")
#     simulator.launch("f_gmg_init", np.int16(zDim), np.int16(nx), np.int16(ny), np.int16(nz), 
#                      np.float32(omega), np.float32(h), nonblock=False)
    
#     # Apply operator to get Au
#     print("Step 2: Apply operator")
#     simulator.launch("f_apply_operator", nonblock=False)
    
#     # Compute residual
#     print("Step 3: Compute residual")
#     simulator.launch("f_compute_residual", nonblock=False)
    
#     # Get residual norm
#     residual_norm = np.zeros(1, dtype=np.float32)
#     simulator.memcpy_d2h(residual_norm, symbol_residual_norm, 0, 0, 1, 1, 1,
#                        streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
#     print(f"Residual norm: {residual_norm[0]}")
    
#     # Jacobi smoothing
#     print("Step 4: Jacobi smoothing")
#     simulator.launch("f_jacobi_smooth", np.int16(num_smooth_iter), nonblock=False)
    
#     # Restriction (for now, just copy)
#     print("Step 5: Restriction")
#     simulator.launch("f_restrict", nonblock=False)
    
#     # Interpolation (for now, just copy)
#     print("Step 6: Interpolation")
#     simulator.launch("f_interpolate", nonblock=False)
    
#     # Add correction
#     print("Step 7: Add correction")
#     simulator.launch("f_add_correction", nonblock=False)
    
#     return residual_norm[0]

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

    # perform argument checks
    assert pe_length >= 2, "the maximum size of z must be greater than 1"
    assert zDim >= 2, "the minimum size of zDim must be greater than 1"
    assert zDim <= pe_length, "[0, zDim) cannot exceed the storage"

    # Initialize solver data.
    host_solver = SimpleGMG(width, height, zDim, args.levels, args.verbose, args.tolerance, args.pre_iter, args.post_iter, args.bottom_iter)
    device_solver = copy.deepcopy(host_solver)    # Before solving make sure to make a deep copy as python might modify the data in the original object.


    # Host GMG for validation
    # host_residual, host_iterations = host_solver.solve(args.max_ite)
    # print(f"Host residual: {host_residual}, host iterations: {host_iterations}")

    # Device side
    # Calculate fabric dimensions
    # fabric_width, fabric_height, core_fabric_offset_x, core_fabric_offset_y = calculate_fabric_dimensions(args, width, height, args.width_west_buf, args.width_east_buf)
    
    # # Compile the kernel
    # layout_file = "./src/layout_gmg.csl"
    # # This is used when user doesn't use a compile command first.
    # csl_compile_core(cslc=cslc, width=width, height=height, pe_length=pe_length, blockSize=args.blockSize, file_config=layout_file,
    #     elf_dir=logs_dir, fabric_width=fabric_width, fabric_height=fabric_height, core_fabric_offset_x=core_fabric_offset_x, core_fabric_offset_y=core_fabric_offset_y, use_precompile=args.run_only,
    #     arch=args.arch if args.arch else "wse2", C0=0, C1=1, C2=2, C3=3, C4=4, C5=5, C6=6, C7=7, C8=8, channels=args.channels, 
    #     width_west_buf=args.width_west_buf, width_east_buf=args.width_east_buf, levels=args.levels)
    

    # device_grid = device_solver.get_grids() # Already has data filled and shapes initialized.
    u_hwl = np.transpose(device_solver.grids[0]['u'], (1, 2, 0))  # (ny, nx, nz) -> (height, width, zDim)
    f_hwl = np.transpose(device_solver.grids[0]['f'], (1, 2, 0))  # Change order because CSL expects (height, width, zDim), numpy is (zDim, height, width)

    x_1d_level_0 = hwl_2_oned_colmajor(height, width, zDim, u_hwl, DTYPE)
    f_1d_level_0 = hwl_2_oned_colmajor(height, width, zDim, f_hwl, DTYPE)    
    # x_1d_level_0 = hwl_2_oned_colmajor(height, width, zDim, device_solver.grids[0]['u'], DTYPE)
    # f_1d_level_0 = hwl_2_oned_colmajor(height, width, zDim, device_solver.grids[0]['f'], DTYPE)
    print(f"f_1d_level_0.shape: {f_1d_level_0}")
    # order: {c_west, c_east, c_south, c_north, c_bottom, c_top, c_center}
    stencil_coeff = np.zeros((height, width, 7), dtype=DTYPE)  # 3D-7pt
    stencil_coeff[:, :, 0] = device_solver.BETA # west(-1)
    stencil_coeff[:, :, 1] = device_solver.BETA # east(-1)
    stencil_coeff[:, :, 2] = device_solver.BETA # south(-1)
    stencil_coeff[:, :, 3] = device_solver.BETA # north(-1)
    stencil_coeff[:, :, 4] = device_solver.BETA # bottom(-1)
    stencil_coeff[:, :, 5] = device_solver.BETA # top(-1)
    stencil_coeff[:, :, 6] = device_solver.ALPHA  # center (-6)



    # Create simulator
    memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
    memcpy_order = MemcpyOrder.COL_MAJOR
    simulator = SdkRuntime(logs_dir, cmaddr=args.cmaddr)
    
    # Load and run
    simulator.load()
    simulator.run()
    
    # Get symbols
    symbol_u = simulator.get_id("u")
    symbol_f = simulator.get_id("f")
    symbol_r = simulator.get_id("r")
    symbol_Au = simulator.get_id("Au")
    symbol_residual_norm = simulator.get_id("residual_norm")
    symbol_stencil_coeff = simulator.get_id("stencil_coeff")
 
    
#     # Initialize data
#     print("Initializing data...")
    
#     # Create test problem: 3D Poisson with sin source
#     x = np.linspace(0, 1, nx)
#     y = np.linspace(0, 1, ny)
#     z = np.linspace(0, 1, nz)
#     X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
#     # Right-hand side: f = sin(πx) * sin(πy) * sin(πz)
#     f_3d = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    
#     # Initial solution
#     u_3d = np.zeros_like(f_3d)
    
#     # Convert to 1D for device
#     u_1d = hwl_2_oned_colmajor(height, width, pe_length, u_3d, np.float32)
#     f_1d = hwl_2_oned_colmajor(height, width, pe_length, f_3d, np.float32)
    
#     # Stencil coefficients for 7-point Poisson
#     stencil_coeff = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -6.0], dtype=np.float32)
    
#     # Copy data to device
#     print("Copying data to device...")
#     copy_data_h2d(u_1d, f_1d, stencil_coeff, height, width, pe_length, memcpy_dtype, simulator,
#                   symbol_u, symbol_f, symbol_stencil_coeff)
    
#     # Run GMG algorithm
#     print("Running GMG algorithm...")
#     residual_norm = gmg_algorithm(pe_length, nx, ny, nz, omega, h, num_smooth_iter, 
#                                  memcpy_dtype, simulator, symbol_u, symbol_f, symbol_r, 
#                                  symbol_Au, symbol_residual_norm)
    
#     # Copy results back
#     print("Copying results back...")
#     u_result = copy_data_d2h(height, width, pe_length, memcpy_dtype, simulator, symbol_u)
    
#     # Convert back to 3D
#     u_result_3d = oned_to_hwl_colmajor(height, width, pe_length, u_result, np.float32)
    
#     # Print results
#     print("=" * 60)
#     print("GMG Results")
#     print("=" * 60)
#     print(f"Final residual norm: {residual_norm}")
#     print(f"Solution shape: {u_result_3d.shape}")
#     print(f"Solution range: [{u_result_3d.min():.6f}, {u_result_3d.max():.6f}]")
    
#     # Cleanup
#     simulator.stop()

if __name__ == "__main__":
    main()
