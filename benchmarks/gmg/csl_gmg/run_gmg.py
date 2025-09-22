#!/usr/bin/env python3
"""
Geometric Multigrid (GMG) solver using CSL
Host-side implementation that coordinates with device kernels
"""

import numpy as np
import argparse
import os
import sys
from typing import Tuple, List
import time

# Add the conjugate gradient utilities
sys.path.append('../../conjugate-gradient')
from util import COL_MAJOR, hwl_2_oned_colmajor, oned_to_hwl_colmajor, laplacian, csr_7_pt_stencil
from cmd_parser import parse_args

# Import Cerebras SDK
try:
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyOrder, MemcpyDataType
except ImportError:
    print("Warning: Cerebras SDK not available. This script requires the Cerebras SDK to run.")
    SdkRuntime = None
    MemcpyOrder = None
    MemcpyDataType = None

def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)

def csl_compile_core(
    cslc: str,
    width: int,
    height: int,
    pe_length: int,
    blockSize: int,
    file_config: str,
    elf_dir: str,
    fabric_width: int,
    fabric_height: int,
    core_fabric_offset_x: int,
    core_fabric_offset_y: int,
    use_precompile: bool,
    arch: str,
    C0: int, C1: int, C2: int, C3: int, C4: int, C5: int, C6: int, C7: int, C8: int,
    channels: int,
    width_west_buf: int,
    width_east_buf: int
):
    """Compile the CSL kernel"""
    if use_precompile:
        print(f"use pre-compile ELFs")
        return
    
    # Build the cslc command
    cmd = [
        cslc, file_config,
        "--arch", arch,
        "--fabric-dims", f"{fabric_width},{fabric_height}",
        "--fabric-offsets", f"{core_fabric_offset_x},{core_fabric_offset_y}",
        "--params", f"width:{width},height:{height},MAX_ZDIM:{pe_length}",
        "--params", f"BLOCK_SIZE:{blockSize}",
        "--params", f"C0_ID:{C0},C1_ID:{C1},C2_ID:{C2},C3_ID:{C3},C4_ID:{C4}",
        "--params", f"C5_ID:{C5},C6_ID:{C6},C7_ID:{C7},C8_ID:{C8}",
        "-o", elf_dir,
        "--memcpy",
        "--channels", str(channels),
        "--width-west-buf", str(width_west_buf),
        "--width-east-buf", str(width_east_buf)
    ]
    
    print(" ".join(cmd))
    result = os.system(" ".join(cmd))
    if result != 0:
        raise RuntimeError(f"cslc failed with exit code {result}")

def calculate_fabric_dimensions(args, width, height, width_west_buf, width_east_buf):
    """Calculate fabric dimensions based on core size and buffers"""
    fabric_width = width + width_west_buf + width_east_buf
    fabric_height = height
    
    # Add some padding
    fabric_width = max(fabric_width + 2, 8)
    fabric_height = max(fabric_height + 2, 8)
    
    return fabric_width, fabric_height

def timing_analysis(height, width, zDim, time_memcpy_hwl, time_ref_hwl):
    """Analyze timing data"""
    # Convert timing data to cycles
    time_start = make_u48(time_memcpy_hwl[0:3])
    time_end = make_u48(time_memcpy_hwl[3:6])
    time_ref = make_u48(time_ref_hwl[0:3])
    
    # Adjust for reference time
    time_start_adj = time_start - time_ref
    time_end_adj = time_end - time_ref
    
    # Calculate elapsed time in cycles
    cycles_elapsed = time_end_adj - time_start_adj
    
    # Convert to microseconds (assuming 850 MHz clock)
    time_us = (cycles_elapsed / 0.85) * 1e-3
    
    return time_us

def copy_data_h2d(u_1d, f_1d, stencil_coeff, height, width, zDim, memcpy_dtype, simulator, 
                  symbol_u, symbol_f, symbol_stencil_coeff):
    """Copy data from host to device"""
    # Copy solution vector u
    simulator.memcpy_h2d(u_1d, symbol_u, 0, 0, height, width, zDim,
                         streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)
    
    # Copy right-hand side f
    simulator.memcpy_h2d(f_1d, symbol_f, 0, 0, height, width, zDim,
                         streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)
    
    # Copy stencil coefficients
    simulator.memcpy_h2d(stencil_coeff, symbol_stencil_coeff, 0, 0, height, width, 7,
                         streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

def copy_data_d2h(height, width, zDim, memcpy_dtype, simulator, symbol_u, symbol_time_buf_u16, symbol_time_ref):
    """Copy data from device to host"""
    # Copy solution vector u
    u_result = np.zeros((height, width, zDim), dtype=memcpy_dtype)
    simulator.memcpy_d2h(u_result, symbol_u, 0, 0, height, width, zDim,
                        streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
    
    # Copy timing data
    time_buf = np.zeros((timestamp.tsc_size_words*2,), dtype=np.uint16)
    simulator.memcpy_d2h(time_buf, symbol_time_buf_u16, 0, 0, 1, 1, timestamp.tsc_size_words*2,
                        streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
    
    time_ref = np.zeros((timestamp.tsc_size_words,), dtype=np.uint16)
    simulator.memcpy_d2h(time_ref, symbol_time_ref, 0, 0, 1, 1, timestamp.tsc_size_words,
                        streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
    
    return u_result, time_buf, time_ref

def gmg_algorithm(zDim, nx, ny, nz, omega, h, num_smooth_iter, memcpy_dtype, simulator, 
                  symbol_u, symbol_f, symbol_r, symbol_Au, symbol_residual_norm):
    """Main GMG algorithm"""
    print(f"GMG Algorithm: grid size {nx}x{ny}x{nz}, zDim={zDim}")
    print(f"Parameters: omega={omega}, h={h}, smooth_iter={num_smooth_iter}")
    
    # Initialize GMG
    print("Step 1: Initialize GMG")
    simulator.launch("f_gmg_init", np.int16(zDim), np.int16(nx), np.int16(ny), np.int16(nz), 
                     np.float32(omega), np.float32(h), nonblock=False)
    
    # Apply operator to get Au
    print("Step 2: Apply operator")
    simulator.launch("f_apply_operator", nonblock=False)
    
    # Compute residual
    print("Step 3: Compute residual")
    simulator.launch("f_compute_residual", nonblock=False)
    
    # Get residual norm
    residual_norm = np.zeros(1, dtype=np.float32)
    simulator.memcpy_d2h(residual_norm, symbol_residual_norm, 0, 0, 1, 1, 1,
                       streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
    print(f"Residual norm: {residual_norm[0]}")
    
    # Jacobi smoothing
    print("Step 4: Jacobi smoothing")
    simulator.launch("f_jacobi_smooth", np.int16(num_smooth_iter), nonblock=False)
    
    # Restriction (for now, just copy)
    print("Step 5: Restriction")
    simulator.launch("f_restrict", nonblock=False)
    
    # Interpolation (for now, just copy)
    print("Step 6: Interpolation")
    simulator.launch("f_interpolate", nonblock=False)
    
    # Add correction
    print("Step 7: Add correction")
    simulator.launch("f_add_correction", nonblock=False)
    
    return residual_norm[0]

def main():
    """Main function"""
    args, logs_dir = parse_args()
    
    # GMG parameters
    nx, ny, nz = 8, 8, 8  # Grid dimensions
    omega = 0.5  # Jacobi relaxation parameter
    h = 1.0 / (nx - 1)  # Grid spacing
    num_smooth_iter = 3  # Number of smoothing iterations
    
    # Calculate dimensions
    width = args.n
    height = args.m
    zDim = args.zDim
    pe_length = zDim
    
    # Calculate fabric dimensions
    fabric_width, fabric_height = calculate_fabric_dimensions(args, width, height, args.width_west_buf, args.width_east_buf)
    
    print("=" * 60)
    print("Geometric Multigrid (GMG) Solver - CSL Implementation")
    print("=" * 60)
    print(f"Grid size: {nx}x{ny}x{nz}")
    print(f"Core size: {width}x{height}")
    print(f"Fabric size: {fabric_width}x{fabric_height}")
    print(f"zDim: {zDim}")
    print(f"Omega: {omega}")
    print(f"Grid spacing h: {h}")
    print("=" * 60)
    
    # Compile the kernel
    csl_compile_core(
        cslc="cslc",
        width=width,
        height=height,
        pe_length=pe_length,
        blockSize=args.blockSize,
        file_config="./src/layout_gmg.csl",
        elf_dir=logs_dir,
        fabric_width=fabric_width,
        fabric_height=fabric_height,
        core_fabric_offset_x=1,
        core_fabric_offset_y=1,
        use_precompile=args.run_only,
        arch=args.arch if args.arch else "wse2",
        C0=0, C1=1, C2=2, C3=3, C4=4, C5=5, C6=6, C7=7, C8=8,
        channels=args.channels,
        width_west_buf=args.width_west_buf,
        width_east_buf=args.width_east_buf
    )
    
    # Initialize simulator
    if SdkRuntime is None:
        print("Error: Cerebras SDK not available. Cannot run GMG simulation.")
        return
    
    memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
    
    # Create simulator
    simulator = SdkRuntime(
        name="gmg",
        cmaddr=args.cmaddr,
        elf_dir=logs_dir,
        fabric_dims=(fabric_width, fabric_height),
        arch=args.arch if args.arch else "wse2"
    )
    
    # Load and run
    simulator.load()
    simulator.run()
    
    # Get symbols
    symbol_u = simulator.get_id("u")
    symbol_f = simulator.get_id("f")
    symbol_r = simulator.get_id("r")
    symbol_Au = simulator.get_id("Au")
    symbol_residual_norm = simulator.get_id("residual_norm")
    symbol_time_buf_u16 = simulator.get_id("time_buf_u16")
    symbol_time_ref = simulator.get_id("time_ref")
    
    # Initialize data
    print("Initializing data...")
    
    # Create test problem: 3D Poisson with sin source
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Right-hand side: f = sin(πx) * sin(πy) * sin(πz)
    f_3d = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    
    # Initial solution
    u_3d = np.zeros_like(f_3d)
    
    # Convert to 1D for device
    u_1d = hwl_2_oned_colmajor(height, width, pe_length, u_3d, np.float32)
    f_1d = hwl_2_oned_colmajor(height, width, pe_length, f_3d, np.float32)
    
    # Stencil coefficients for 7-point Poisson
    stencil_coeff = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -6.0], dtype=np.float32)
    
    # Copy data to device
    print("Copying data to device...")
    copy_data_h2d(u_1d, f_1d, stencil_coeff, height, width, pe_length, memcpy_dtype, simulator,
                  symbol_u, symbol_f, symbol_stencil_coeff)
    
    # Run GMG algorithm
    print("Running GMG algorithm...")
    start_time = time.time()
    
    residual_norm = gmg_algorithm(pe_length, nx, ny, nz, omega, h, num_smooth_iter, 
                                 memcpy_dtype, simulator, symbol_u, symbol_f, symbol_r, 
                                 symbol_Au, symbol_residual_norm)
    
    total_time = time.time() - start_time
    
    # Copy results back
    print("Copying results back...")
    u_result, time_buf, time_ref = copy_data_d2h(height, width, pe_length, memcpy_dtype, simulator,
                                                symbol_u, symbol_time_buf_u16, symbol_time_ref)
    
    # Convert back to 3D
    u_result_3d = oned_to_hwl_colmajor(height, width, pe_length, u_result, np.float32)
    
    # Print results
    print("=" * 60)
    print("GMG Results")
    print("=" * 60)
    print(f"Final residual norm: {residual_norm}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Solution shape: {u_result_3d.shape}")
    print(f"Solution range: [{u_result_3d.min():.6f}, {u_result_3d.max():.6f}]")
    
    # Cleanup
    simulator.stop()

if __name__ == "__main__":
    main()
