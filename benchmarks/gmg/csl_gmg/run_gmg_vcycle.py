#!/usr/bin/env python3
"""
Geometric Multigrid (GMG) V-cycle solver using CSL with state machine
Runs complete V-cycle on device
"""

import math
import os
import sys
from typing import Optional
from pathlib import Path
import shutil
import subprocess
import numpy as np
import copy

# sys.path.append(os.path.join(os.path.dirname(__file__), '..', "python_gmg"))
from python_gmg.gmgoscar import SimpleGMG as SimpleGMGOSCAR
from cmd_parser import parse_args, print_arguments
from util import hwl_2_oned_colmajor, oned_to_hwl_colmajor

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyOrder, MemcpyDataType

DTYPE = np.float32

def csl_compile_core(
    cslc: str, width: int, height: int, pe_length: int, blockSize: int, 
    file_config: str, elf_dir: str, levels: int,
    fabric_width: int, fabric_height: int, 
    core_fabric_offset_x: int, core_fabric_offset_y: int, 
    use_precompile: bool, arch: Optional[str], 
    C0: int, C1: int, C2: int, C3: int, C4: int, C5: int, C6: int, C7: int, C8: int, 
    channels: int, width_west_buf: int, width_east_buf: int):
    
    if not use_precompile:
        args = []
        args.append(cslc)
        args.append(file_config)
        args.append(f"--fabric-dims={fabric_width},{fabric_height}")
        args.append(f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}")
        args.append(f"--params=width:{width},height:{height},MAX_ZDIM:{pe_length},LEVELS:{levels}")
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
    fabric_offset_x = 1
    fabric_offset_y = 1
    core_fabric_offset_x = fabric_offset_x + 3 + width_west_buf
    core_fabric_offset_y = fabric_offset_y
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

def subsample_activenodes_only(input_array_3d, level=0):
    """Subsample array based on active nodes at given level"""
    factor = 2**(level)
    subsampled_array = input_array_3d[::factor, ::factor, :]
    return subsampled_array

def get_exponent(A: int) -> int:
    return int(math.log2(A))

def copy_data_h2d(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u, symbol_f, symbol_hx_array, symbol_hy_array, symbol_hz_array, symbol_jacobi_coeff_array, device_solver, args):
    """
    Copies problem and grid spacing data to device, but now
    for hx/hy/hz/jacobi_coeff arrays, repeats values so each PE (height x width)
    gets a full levels-vector (across the 3rd dimension). Arranges those
    arrays into (height, width, levels) layout for device copy.
    """
    # Prepare grid spacing arrays for one PE (vector length = args.levels)
    hx_base = np.array([device_solver.grids[i]['hx'] for i in range(args.levels)], dtype=DTYPE)
    hy_base = np.array([device_solver.grids[i]['hy'] for i in range(args.levels)], dtype=DTYPE)
    hz_base = np.array([device_solver.grids[i]['hz'] for i in range(args.levels)], dtype=DTYPE)
    
    # Prepare Jacobi coefficient array for one PE
    jacobi_coeff_base = np.zeros(args.levels, dtype=DTYPE)
    for level in range(args.levels):
        hx = device_solver.grids[level]['hx']
        hy = device_solver.grids[level]['hy']
        hz = device_solver.grids[level]['hz']
        jacobi_coeff_base[level] = -1.0 / (3.0 * (1.0/(hx*hx) + 1.0/(hy*hy) + 1.0/(hz*hz)))
    
    # Repeat these arrays so shape is (height, width, levels)
    hx_array = np.tile(hx_base, (height, width, 1))
    hy_array = np.tile(hy_base, (height, width, 1))
    hz_array = np.tile(hz_base, (height, width, 1))
    jacobi_coeff_array = np.tile(jacobi_coeff_base, (height, width, 1))

    # Prepare u/f arrays
    u_hwl = device_solver.grids[0]['u']
    f_hwl = device_solver.grids[0]['f']
    u_1d = hwl_2_oned_colmajor(height, width, zDim, u_hwl, DTYPE)
    f_1d = hwl_2_oned_colmajor(height, width, zDim, f_hwl, DTYPE)
    # Copy u and f arrays
    simulator.memcpy_h2d(symbol_u, u_1d, 0, 0, width, height, zDim,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    simulator.memcpy_h2d(symbol_f, f_1d, 0, 0, width, height, zDim,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    
    # Debug: Print what we're sending
    print(f"\nGrid spacing and Jacobi coefficients:")
    for level in range(args.levels):
        print(f"  Level {level}: hx={hx_base[level]:.6f}, hy={hy_base[level]:.6f}, hz={hz_base[level]:.6f}, jacobi={jacobi_coeff_base[level]:.6e}")
    
    # Copy spacing/jacobi arrays: flatten in COLUMN-MAJOR order to match memcpy
    hx_flat = hwl_2_oned_colmajor(height, width, args.levels, hx_array, DTYPE)
    hy_flat = hwl_2_oned_colmajor(height, width, args.levels, hy_array, DTYPE)
    hz_flat = hwl_2_oned_colmajor(height, width, args.levels, hz_array, DTYPE)
    jacobi_flat = hwl_2_oned_colmajor(height, width, args.levels, jacobi_coeff_array, DTYPE)
    # The target symbol expected shape is (height, width, levels)
    simulator.memcpy_h2d(symbol_hx_array, hx_flat, 0, 0, width, height, args.levels,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    simulator.memcpy_h2d(symbol_hy_array, hy_flat, 0, 0, width, height, args.levels,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    simulator.memcpy_h2d(symbol_hz_array, hz_flat, 0, 0, width, height, args.levels,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    simulator.memcpy_h2d(symbol_jacobi_coeff_array, jacobi_flat, 0, 0, width, height, args.levels,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)

def make_u48(words):
    """Convert three u16 words to 48-bit timestamp"""
    return words[0] + (words[1] << 16) + (words[2] << 32)

def copy_timing(height, width, levels, simulator, symbol_timing):
    """Copy timing data from device and return as numpy array"""
    # Each level has 6 u16 values (start[3] + end[3])
    timing_size = levels * 6
    timing_1d = np.zeros(height * width * timing_size, np.uint32)
    simulator.memcpy_d2h(timing_1d, symbol_timing, 0, 0, width, height, timing_size,
                        streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT, 
                        order=MemcpyOrder.COL_MAJOR, nonblock=False)

    # Convert to hwl format: (height, width, timing_size)
    # With column-major, data for each PE is contiguous (all levels together)
    timing_hwl = oned_to_hwl_colmajor(height, width, timing_size, timing_1d, np.uint16)
    
    # Now extract each level's 6 values from the third dimension
    time_memcpy_hwl_levels = []
    for level in range(levels):
        offset = level * 6
        # Extract the 6 timing values for this level from each PE
        time_memcpy_hwl_level = timing_hwl[:, :, offset:offset+6]
        time_memcpy_hwl_levels.append(time_memcpy_hwl_level)

    return time_memcpy_hwl_levels
  

def process_reference_data(height, width, time_ref_hwl):
    """Adjust reference clock by propagation delay and convert to full timestamps
    
    Returns: time_ref array of shape (height, width) with adjusted 48-bit timestamps
    """
    time_ref = np.zeros((height, width)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(width):
        for h in range(height):
            word[0] = time_ref_hwl[h, w, 0]
            word[1] = time_ref_hwl[h, w, 1]
            word[2] = time_ref_hwl[h, w, 2]
            time_ref[h, w] = make_u48(word)
    
    # Adjust reference clock by propagation delay
    # The right-bottom PE signals other PEs, the propagation delay is:
    #     (height-1) - py + (width-1) - px
    for py in range(height):
        for px in range(width):
            time_ref[py, px] = time_ref[py, px] - ((width + height - 2) - (px + py))
    
    return time_ref

def process_timing_data(height, width, levels, timing_hwl_levels, time_ref_hwl, operation_name, counters):
    """Process timing data for one operation across all levels
    
    Adjusts timestamps relative to reference clock before computing cycles
    """
    
    word = np.zeros(3).astype(np.uint16)
    time_start_levels = []
    time_end_levels = []
    for level in range(levels):
        timing_hwl = timing_hwl_levels[level]
        # Create NEW arrays for each level (don't reuse!)
        time_start = np.zeros((height, width)).astype(int)
        time_end = np.zeros((height, width)).astype(int)
        for w in range(width):
            for h in range(height):
                word[0] = timing_hwl[h, w, 0]
                word[1] = timing_hwl[h, w, 1]
                word[2] = timing_hwl[h, w, 2]
                time_start[h, w] = make_u48(word)
                word[0] = timing_hwl[h, w, 3]
                word[1] = timing_hwl[h, w, 4]
                word[2] = timing_hwl[h, w, 5]
                time_end[h, w] = make_u48(word)
        # store this start and end for this level
        time_start_levels.append(time_start)
        time_end_levels.append(time_end)


    # do for reference clock as well.
    # only 1 reference unlike levels
    time_ref = np.zeros((height, width)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(width):
        for h in range(height):
            word[0] = time_ref_hwl[h, w, 0]
            word[1] = time_ref_hwl[h, w, 1]
            word[2] = time_ref_hwl[h, w, 2]
            time_ref[h, w] = make_u48(word)

    # Adjust reference clock by propagation delay
    # The right-bottom PE signals other PEs, the propagation delay is:
    #     (height-1) - py + (width-1) - px
    for py in range(height):
        for px in range(width):
            time_ref[py, px] = time_ref[py, px] - ((width + height - 2) - (px + py))
    
    # now perform a shift in time for each level by the reference time
    time_start_levels_shifted = []
    time_end_levels_shifted = []
    for level in range(levels):
        shift_start = time_start_levels[level] - time_ref
        shift_end = time_end_levels[level] - time_ref
        time_start_levels_shifted.append(shift_start)
        time_end_levels_shifted.append(shift_end)
    
    # min, max, cycles, and time for each level
    timing_per_level = []
    for level in range(levels):
        min_cycles = time_start_levels_shifted[level].min()
        max_cycles = time_end_levels_shifted[level].max()
        cycles_send = max_cycles - min_cycles
        time_send = (cycles_send / 0.875) * 1.e-3
        timing_per_level.append({
            'level': level,
            'operation': operation_name,
            'cycles_send': cycles_send if counters[level] > 0 else 0,
            'time_send': time_send if counters[level] > 0 else 0
        })
    return timing_per_level

def copy_data_d2h(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u, symbol_r, device_solver, args):
    
    u_wse_1d = np.zeros(height*width*zDim, DTYPE)
    simulator.memcpy_d2h(u_wse_1d, symbol_u, 0, 0, width, height, zDim,
                        streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    
    r_wse_1d = np.zeros(height*width*zDim, DTYPE)
    simulator.memcpy_d2h(r_wse_1d, symbol_r, 0, 0, width, height, zDim,
                        streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)

    return u_wse_1d, r_wse_1d

def copy_timing_data(height, width, levels, simulator, symbol_timing_smooth, symbol_timing_apply_op, symbol_timing_residual, symbol_timing_restrict, symbol_timing_interp, args):
    timing_smooth_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_smooth)
    timing_residual_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_residual)
    timing_apply_op_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_apply_op)
    timing_restrict_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_restrict)
    timing_interp_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_interp)
    return timing_smooth_hwl, timing_residual_hwl, timing_apply_op_hwl, timing_restrict_hwl, timing_interp_hwl

def copy_counters(height, width, levels, simulator, symbol_counter):
    """Copy operation counter data from device"""
    counter_1d = np.zeros(height * width * levels, np.uint32)
    simulator.memcpy_d2h(counter_1d, symbol_counter, 0, 0, width, height, levels,
                        streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT, 
                        order=MemcpyOrder.COL_MAJOR, nonblock=False)
    
    # Convert to hwl format: (height, width, levels)
    counter_hwl = oned_to_hwl_colmajor(height, width, levels, counter_1d, np.uint16)
    
    # Average across all PEs for each level (or take from PE(0,0))
    # For now, take from PE(0,0) as representative
    counter_per_level = counter_hwl[0, 0, :]
    
    return counter_per_level

def copy_counter_data(height, width, levels, simulator, symbol_counter_smooth_down, symbol_counter_smooth_up, 
                     symbol_counter_apply_op, symbol_counter_restrict, symbol_counter_interp, args):
    """Copy all operation counters from device"""
    counter_smooth_down = copy_counters(height, width, args.levels, simulator, symbol_counter_smooth_down)
    counter_smooth_up = copy_counters(height, width, args.levels, simulator, symbol_counter_smooth_up)
    counter_apply_op = copy_counters(height, width, args.levels, simulator, symbol_counter_apply_op)
    counter_restrict = copy_counters(height, width, args.levels, simulator, symbol_counter_restrict)
    counter_interp = copy_counters(height, width, args.levels, simulator, symbol_counter_interp)
    
    return counter_smooth_down, counter_smooth_up, counter_apply_op, counter_restrict, counter_interp

def main():
    """Main function"""
    np.random.seed(2)

############################################################
# Parameters
############################################################
    args, logs_dir = parse_args()
    
    # Problem parameters
    height = args.m
    width = args.n
    pe_length = args.k
    zDim = args.zDim
    
    print(f"width = {width}, height = {height}, pe_length = {pe_length}, zDim = {zDim}")
    print(f"levels = {args.levels}, max_ite = {args.max_ite}")
    
    # Validation
    assert pe_length >= 2, "pe_length must be >= 2"
    assert zDim >= 2, "zDim must be >= 2"
    assert zDim <= pe_length, "zDim must be <= pe_length"
    
    max_possible_levels = get_exponent(height)
    if args.levels > max_possible_levels:
        raise ValueError(f"levels ({args.levels}) > max_possible_levels ({max_possible_levels})")

############################################################
# Host
############################################################
    # Create reference solver on host
    print("\n" + "="*60)
    print("Creating reference solver on host...")
    print("="*60)
    host_solver = SimpleGMGOSCAR(width, height, zDim, args.levels, args.verbose, 
                                 args.tolerance, args.pre_iter, args.post_iter, args.bottom_iter)
    device_solver = copy.deepcopy(host_solver)
    
    # Run reference on host
    host_solver.solve_iterative(args.max_ite)
    
############################################################
# Device
############################################################
    # Initialize device
    print("\n" + "="*60)
    print("Initializing device...")
    print("="*60)
    
    memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
    memcpy_order = MemcpyOrder.COL_MAJOR
    simulator = SdkRuntime(logs_dir, cmaddr=args.cmaddr, simfab_numthreads=64)
    
    symbol_u = simulator.get_id("u")
    symbol_f = simulator.get_id("f")
    symbol_r = simulator.get_id("r")
    symbol_hx_array = simulator.get_id("hx_array")
    symbol_hy_array = simulator.get_id("hy_array")
    symbol_hz_array = simulator.get_id("hz_array")
    symbol_jacobi_coeff_array = simulator.get_id("jacobi_coeff_array")
    # timing
    symbol_timing_smooth = simulator.get_id("timing_smooth")
    symbol_timing_apply_op = simulator.get_id("timing_apply_op")
    symbol_timing_residual = simulator.get_id("timing_residual")
    symbol_timing_restrict = simulator.get_id("timing_restrict")
    symbol_timing_interp = simulator.get_id("timing_interp")
    symbol_time_ref = simulator.get_id("time_ref")
    # counters
    symbol_counter_smooth = simulator.get_id("counter_smooth")
    symbol_counter_apply_op = simulator.get_id("counter_apply_op")
    symbol_counter_residual = simulator.get_id("counter_residual")
    symbol_counter_restrict = simulator.get_id("counter_restrict")
    symbol_counter_interp = simulator.get_id("counter_interp")
    
    simulator.load()
    simulator.run()

    # Copy initial data
    print("\n" + "="*60)
    print("Copying data to device...")
    print("="*60)
    device_solver.compute_residual(0)
    print(f"Initial residual at level 0: {device_solver.calculate_rho(device_solver.grids[0]['r']):.6e}")
   
    copy_data_h2d(height, width, zDim, memcpy_dtype, memcpy_order, simulator, 
                 symbol_u, symbol_f, symbol_hx_array, symbol_hy_array, symbol_hz_array, symbol_jacobi_coeff_array, device_solver, args)
    
    # Enable timing and synchronize PEs
    print("\nEnabling timer...")
    simulator.launch("f_enable_timer", nonblock=False)
    
    print("Synchronizing PEs...")
    simulator.launch("f_sync", nonblock=False)
    
    print("Copying reference clock...")
    simulator.launch("f_reference_timestamps", nonblock=False)
    # 
    # Run GMG V-cycle on device
    print("\n" + "="*60)
    print(f"Running GMG V-cycle on device (levels={args.levels})...")
    print("="*60)
    
    simulator.launch("f_gmg_vcycle", 
                    # np.int16(zDim), 
                    np.int16(args.levels),
                    np.int16(args.pre_iter),
                    np.int16(args.post_iter),
                    np.int16(args.bottom_iter),
                    nonblock=False)
    
    # Copy results back
    print("\n" + "="*60)
    print("Copying results from device...")
    print("="*60)
    
    u_wse_1d, r_wse_1d = copy_data_d2h(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u, symbol_r, device_solver, args)
    
    # Copy timing data from device
    print("\nCopying timing data...")
    timing_smooth_hwl_levels, timing_residual_hwl_levels, timing_apply_op_hwl_levels, timing_restrict_hwl_levels, timing_interp_hwl_levels = copy_timing_data(height, width, args.levels, simulator, symbol_timing_smooth, symbol_timing_apply_op, symbol_timing_residual, symbol_timing_restrict, symbol_timing_interp, args)
    
    # Copy operation counters from device
    print("Copying operation counters...")
    counter_smooth, counter_residual, counter_apply_op, counter_restrict, counter_interp = copy_counter_data(height, width, args.levels, simulator, symbol_counter_smooth, symbol_counter_residual, symbol_counter_apply_op, symbol_counter_restrict, symbol_counter_interp, args)
    
    # Copy reference clock
    print("Copying reference clock...")
    time_ref_1d = np.zeros(height * width * 3, np.uint32)
    simulator.memcpy_d2h(time_ref_1d, symbol_time_ref, 0, 0, width, height, 3,
                         streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT, 
                         order=MemcpyOrder.COL_MAJOR, nonblock=False)
    time_ref_hwl = oned_to_hwl_colmajor(height, width, 3, time_ref_1d, np.uint16)
    simulator.stop()
    
    # Reshape results
    u_wse_3d = oned_to_hwl_colmajor(height, width, zDim, u_wse_1d, DTYPE)
    # r_wse_3d = oned_to_hwl_colmajor(height, width, zDim, r_wse_1d, DTYPE)
    # Store device results
    # print(f"u_wse_3d shape = {u_wse_3d.shape}")
    # print(f"u_wse_3d[:, :, 0] = {u_wse_3d[:, :, 0]}")
    device_solver.grids[0]['u'] = u_wse_3d
    device_solver.compute_residual(0)
    device_solver.grids[0]['rho_up'] = device_solver.calculate_rho(device_solver.grids[0]['r'])


############################################################
# Verification
############################################################
    # Verification
    print("\n" + "="*60)
    print("Verification")
    print("Checking u, f, r, Au at level 0 for host_solver and device_solver")
    print("="*60)


    # Check/verify u, f, r, Au at level 0 for host_solver and device_solver
    fields = ['u', 'f', 'r', 'Au']
    for field in fields:
        host_field = host_solver.grids[0][field]
        device_field = device_solver.grids[0][field]
        print(f"Host {field}: shape={host_field.shape}, dtype={host_field.dtype}")
        print(f"Device {field}: shape={device_field.shape}, dtype={device_field.dtype}")
        np.testing.assert_allclose(host_field, device_field, atol=1e-5, rtol=1e-5)

        
    device_u = device_solver.grids[0]['u']
    host_u = host_solver.grids[0]['u']
    device_rh = device_solver.grids[0]['rho_up']
    host_rh = host_solver.grids[0]['rho_up']

    # Compare u values at a few points
    print(f"\n All u values at level 0:")
    print(f"  Host   u[:,:,0] = {host_u[:,:,0]}")
    print(f"  Device u[:,:,0] = {device_u[:,:,0]}")
    
    print(f"\nLevel 0 comparison:")
    print(f"  Host   rho_up: {host_rh:.6e}")
    print(f"  Device rho_up: {device_rh:.6e}")
    print(f"  Ratio (device/host): {device_rh / host_rh:.4f}")
    
    # Relaxed tolerance
    try:
        # Compare using the original 1D array from device (u_wse_1d) 
        # not the reshaped 3D array, to avoid memory layout issues
        nrm2_u = np.linalg.norm(u_wse_1d, 2)
        print(f"  |u|_2 = {nrm2_u:.6e}")
        # Host is row-major by default, so ravel with order='F' to match device column-major
        z = host_u.ravel(order='F') - u_wse_1d.ravel()
        nrm2_z = np.linalg.norm(z, np.inf)
        print(f"  |u_host - u_wse| = {nrm2_z:.6e}")
        np.testing.assert_allclose(host_u.ravel(order='F'), u_wse_1d.ravel(), atol=1e-5, rtol=1e-5)
        print("\n SUCCESS! Device and host results match.")
    
    except AssertionError as e:
        print(f"\n Results differ significantly")
        print(f"   This suggests a bug in the state machine implementation.")
        # Print more details
        print(f"\nDetailed comparison (first few points):")
        for i in range(min(3, height)):
            for j in range(min(3, width)):
                h_val = host_u[i,j,0]
                d_val = device_u[i,j,0]
                ratio = d_val / h_val if h_val != 0 else 0
                print(f"   ({i},{j},0): host={h_val:.6e}, device={d_val:.6e}, ratio={ratio:.4f}")

############################################################
# Timing
############################################################
    # Process and display timing information
    print("\n" + "="*60)
    print("Performance Timing")
    print("="*60)

    timing_smooth_data = process_timing_data(height, width, args.levels, timing_smooth_hwl_levels, time_ref_hwl, "smooth", counter_smooth)
    timing_residual_data = process_timing_data(height, width, args.levels, timing_residual_hwl_levels, time_ref_hwl, "residual", counter_residual)
    # timing_apply_data = process_timing_data(height, width, args.levels, timing_apply_op_hwl_levels, time_ref_hwl, "apply_op", counter_apply_op)
    timing_restrict_data = process_timing_data(height, width, args.levels, timing_restrict_hwl_levels, time_ref_hwl, "restriction", counter_restrict)
    timing_interp_data = process_timing_data(height, width, args.levels, timing_interp_hwl_levels, time_ref_hwl, "interpolation", counter_interp)
    

    print("\nTime per operation and level (cycles, time, operation count):")
    print("-" * 100)
    
    for level in range(args.levels):
        smooth = timing_smooth_data[level]
        residual = timing_residual_data[level]
        # apply = timing_apply_data[level]
        restrict = timing_restrict_data[level]
        interp = timing_interp_data[level]
        
        # Get operation counts for this level
        smooth_count = counter_smooth[level]
        residual_count = counter_residual[level]
        apply_count = counter_apply_op[level]
        restrict_count = counter_restrict[level]
        interp_count = counter_interp[level]
        
        print(f"Level {level}:")
        print(f"  smooth :        [{smooth['cycles_send']:8.0f} cycles, {smooth['time_send']:8.3f} us] - {smooth_count} smoothOps")
        print(f"  residual:       [{residual['cycles_send']:8.0f} cycles, {residual['time_send']:8.3f} us] - {residual_count} residualOps")
        # print(f"  apply_op:       [{apply['cycles_send']:8.0f} cycles, {apply['time_send']:8.3f} us] - {apply_count} applyOps")
        print(f"  restriction:    [{restrict['cycles_send']:8.0f} cycles, {restrict['time_send']:8.3f} us] - {restrict_count} restrictions")
        print(f"  interpolation:  [{interp['cycles_send']:8.0f} cycles, {interp['time_send']:8.3f} us] - {interp_count} interpolations")
        # total cycles and time
        total_time = smooth['time_send'] + residual['time_send'] + restrict['time_send'] + interp['time_send']
        total_cycles = smooth['cycles_send'] + residual['cycles_send'] + restrict['cycles_send'] + interp['cycles_send']
        print(f"  total:          [{total_cycles:8.0f} cycles, {total_time:8.3f} us] - {smooth_count + residual_count + restrict_count + interp_count} totalOps")
        print()
    
    # Calculate totals
    total_smooth_cycles = sum([t['cycles_send'] for t in timing_smooth_data])
    total_residual_cycles = sum([t['cycles_send'] for t in timing_residual_data])
    # total_apply_cycles = sum([t['cycles_send'] for t in timing_apply_data])
    total_restrict_cycles = sum([t['cycles_send'] for t in timing_restrict_data])
    total_interp_cycles = sum([t['cycles_send'] for t in timing_interp_data])

    total_smooth_us = sum([t['time_send'] for t in timing_smooth_data])
    total_residual_us = sum([t['time_send'] for t in timing_residual_data])
    # total_apply_us = sum([t['time_send'] for t in timing_apply_data])
    total_restrict_us = sum([t['time_send'] for t in timing_restrict_data])
    total_interp_us = sum([t['time_send'] for t in timing_interp_data])
    total_time_us = total_smooth_us + total_residual_us + total_restrict_us + total_interp_us
    total_time_cycles = total_smooth_cycles + total_residual_cycles + total_restrict_cycles + total_interp_cycles
    
    # Calculate total operation counts
    total_smooth_ops = int(np.sum(counter_smooth))
    total_residual_ops = int(np.sum(counter_residual))
    # total_apply_ops = int(np.sum(counter_apply_op))
    total_restrict_ops = int(np.sum(counter_restrict))
    total_interp_ops = int(np.sum(counter_interp))
    total_ops = total_smooth_ops + total_residual_ops + total_restrict_ops + total_interp_ops
    
    print("=" * 100)
    print("Total Time and Operation Count Summary:")
    print(f"  Total smooth:        {total_smooth_cycles:10.0f} cycles ({total_smooth_us:10.3f} us) - {total_smooth_ops} iterations")
    print(f"  Total residual:      {total_residual_cycles:10.0f} cycles ({total_residual_us:10.3f} us) - {total_residual_ops} residualOps")
    # print(f"  Total apply_op:      {total_apply_cycles:10.0f} cycles ({total_apply_us:10.3f} us) - {total_apply_ops} applyOps")
    print(f"  Total restriction:   {total_restrict_cycles:10.0f} cycles ({total_restrict_us:10.3f} us) - {total_restrict_ops} restrictions")
    print(f"  Total interpolation: {total_interp_cycles:10.0f} cycles ({total_interp_us:10.3f} us) - {total_interp_ops} interpolations")
    print(f"  Total V-cycle time:  {total_time_cycles:10.0f} cycles ({total_time_us:10.3f} us) - {total_ops} totalOps")
    print("=" * 100)
    
    if args.cmaddr is None:
        # Move simulation logs
        dst_log = Path(f"{logs_dir}/sim.log")
        src_log = Path("sim.log")
        if src_log.exists():
            shutil.move(src_log, dst_log)
        
        dst_trace = Path(f"{logs_dir}/simfab_traces")
        src_trace = Path("simfab_traces")
        if dst_trace.exists():
            shutil.rmtree(dst_trace)
        if src_trace.exists():
            shutil.move(src_trace, dst_trace)

if __name__ == "__main__":
    main()

