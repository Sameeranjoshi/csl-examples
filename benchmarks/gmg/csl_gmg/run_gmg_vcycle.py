#!/usr/bin/env python3
"""
Geometric Multigrid (GMG) V-cycle solver using CSL with state machine
Runs complete V-cycle on device
"""

import math
import os
import sys
import time
from typing import Optional
from pathlib import Path
import shutil
import subprocess
import numpy as np
import copy

# sys.path.append(os.path.join(os.path.dirname(__file__), '..', "python_gmg"))
from python_gmg.gmgoscar import SimpleGMG as SimpleGMGOSCAR
# from gmgoscar import SimpleGMG as SimpleGMGOSCAR
from cmd_parser import parse_args, print_arguments
from util import hwl_2_oned_colmajor, oned_to_hwl_colmajor

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyOrder, MemcpyDataType

DTYPE = np.float32    
WORDS_PER_TIMESTAMP = 3
WORDS_PER_START_END = WORDS_PER_TIMESTAMP * 2


def l2(v): 
    return float(np.sqrt(np.dot(v.ravel(), v.ravel())))

def compare_u(u_host, u_dev):
    diff = u_host - u_dev
    return {
        "L_inf_abs": float(np.max(np.abs(diff))),
        "L2_abs": l2(diff),
        "L2_rel": l2(diff) / max(l2(u_host), 1e-30),
        "mean_abs": float(np.mean(np.abs(diff))),
        "median_abs": float(np.median(np.abs(diff))),
    }

def top_k_indices_absdiff(u_host, u_dev, k=10):
    diff = np.abs(u_host - u_dev).ravel()
    if diff.size == 0:
        return []
    idx = np.argpartition(diff, -k)[-k:]
    idx = idx[np.argsort(-diff[idx])]
    shape = u_host.shape
    triples = [np.unravel_index(int(i), shape) for i in idx]
    return [(tuple(t), float(diff[np.ravel_multi_index(t, shape)])) for t in triples]

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
    
    simulator.launch("f_tic_h2d", nonblock=True)
    # Copy u and f arrays
    simulator.memcpy_h2d(symbol_u, u_1d, 0, 0, width, height, zDim,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    simulator.memcpy_h2d(symbol_f, f_1d, 0, 0, width, height, zDim,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    simulator.launch("f_toc_h2d", nonblock=False)

    # Debug: Print what we're sending
    print(f"Grid spacing and Jacobi coefficients:")
    for level in range(args.levels):
        print(f"  Level {level}: hx={hx_base[level]:.6f}, hy={hy_base[level]:.6f}, hz={hz_base[level]:.6f}, jacobi={jacobi_coeff_base[level]:.6e}")
    
    # Copy spacing/jacobi arrays: flatten in COLUMN-MAJOR order to match memcpy
    hx_flat = hwl_2_oned_colmajor(height, width, args.levels, hx_array, DTYPE)
    hy_flat = hwl_2_oned_colmajor(height, width, args.levels, hy_array, DTYPE)
    hz_flat = hwl_2_oned_colmajor(height, width, args.levels, hz_array, DTYPE)
    jacobi_flat = hwl_2_oned_colmajor(height, width, args.levels, jacobi_coeff_array, DTYPE)
    # The target symbol expected shape is (height, width, levels)
    simulator.launch("f_tic_h2d", nonblock=True)
    simulator.memcpy_h2d(symbol_hx_array, hx_flat, 0, 0, width, height, args.levels,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    simulator.memcpy_h2d(symbol_hy_array, hy_flat, 0, 0, width, height, args.levels,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    simulator.memcpy_h2d(symbol_hz_array, hz_flat, 0, 0, width, height, args.levels,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    simulator.memcpy_h2d(symbol_jacobi_coeff_array, jacobi_flat, 0, 0, width, height, args.levels,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    simulator.launch("f_toc_h2d", nonblock=False)
    total_bytes = (
        u_1d.nbytes
        + f_1d.nbytes
        + hx_flat.nbytes
        + hy_flat.nbytes
        + hz_flat.nbytes
        + jacobi_flat.nbytes
    )

    return total_bytes


def make_u48(words):
    """Convert three u16 words to 48-bit timestamp"""
    return words[0] + (words[1] << 16) + (words[2] << 32)

def copy_timing(height, width, levels, simulator, symbol_timing, words_per_entry):
    """Copy timing data from device and return as numpy array"""
    timing_size = levels * words_per_entry
    timing_1d = np.zeros(height * width * timing_size, np.uint32)   # this is just placeholder, and not really 32 bit mismatch.
    simulator.memcpy_d2h(timing_1d, symbol_timing, 0, 0, width, height, timing_size,
                        streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT, 
                        order=MemcpyOrder.COL_MAJOR, nonblock=False)

    # Convert to hwl format: (height, width, timing_size)
    # With column-major, data for each PE is contiguous (all levels together)
    timing_hwl = oned_to_hwl_colmajor(height, width, timing_size, timing_1d, np.uint16)
    
    # Now extract each level's 6 values from the third dimension
    time_memcpy_hwl_levels = []
    for level in range(levels):
        offset = level * words_per_entry
        time_memcpy_hwl_level = timing_hwl[:, :, offset:offset+words_per_entry]
        time_memcpy_hwl_levels.append(time_memcpy_hwl_level)

    return time_memcpy_hwl_levels
  

def process_reference_data(height, width, time_ref_hwl):
    """Adjust reference clock by propagation delay and convert to full timestamps
    
    Returns: time_ref array of shape (height, width) with adjusted 48-bit timestamps
    """
    if isinstance(time_ref_hwl, list):
        assert len(time_ref_hwl) == 1
        time_ref_hwl = time_ref_hwl[0]

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
def print_2d(matrix):
    for row in matrix:
        print(" | ".join(f"{v:8.2f}" for v in row))

def process_timing_data(height, width, levels, timing_hwl_levels, time_ref_hwl, operation_name, counters, is_ref_used, words_per_entry):
    """Process timing data for one operation across all levels
    
    Adjusts timestamps relative to reference clock before computing cycles
    """
    
    word = np.zeros(3).astype(np.uint16)
    timing_per_level = []

    if not is_ref_used: # crude kernel times, no reference clock used
        assert words_per_entry == WORDS_PER_TIMESTAMP
        elapsed_levels = []
        for level in range(levels):
            timing_hwl = timing_hwl_levels[level]
            elapsed = np.zeros((height, width)).astype(int) # 64 bit because of astype(int)
            for w in range(width):
                for h in range(height):
                    word[0] = timing_hwl[h, w, 0]
                    word[1] = timing_hwl[h, w, 1]
                    word[2] = timing_hwl[h, w, 2]
                    elapsed[h, w] = make_u48(word)
            elapsed_levels.append(elapsed)

        for level in range(levels):
            elapsed = elapsed_levels[level]
            cycles_send = elapsed.max()
            time_send = (cycles_send / 0.875) * 1.e-3
            timing_per_level.append({
                'level': level,
                'operation': operation_name,
                'cycles_send': cycles_send if counters[level] > 0 else 0,
                'time_send': time_send if counters[level] > 0 else 0
            })
        return timing_per_level

    else: # ref is used, so we need to process the reference clock

        time_start_levels = []
        time_end_levels = []
        assert words_per_entry == WORDS_PER_START_END
        half_words = words_per_entry // 2
        for level in range(levels):
            timing_hwl = timing_hwl_levels[level]
            time_start = np.zeros((height, width)).astype(int)
            time_end = np.zeros((height, width)).astype(int)
            for w in range(width):
                for h in range(height):
                    for idx in range(half_words):
                        word[idx] = timing_hwl[h, w, idx]
                    time_start[h, w] = make_u48(word)
                    for idx in range(half_words):
                        word[idx] = timing_hwl[h, w, idx + half_words]
                    time_end[h, w] = make_u48(word)
            time_start_levels.append(time_start)
            time_end_levels.append(time_end)

        time_ref = process_reference_data(height, width, time_ref_hwl)
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

    simulator.launch("f_tic_d2h", nonblock=True)
    u_wse_1d = np.zeros(height*width*zDim, DTYPE)
    simulator.memcpy_d2h(u_wse_1d, symbol_u, 0, 0, width, height, zDim,
                        streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    
    r_wse_1d = np.zeros(height*width*zDim, DTYPE)
    simulator.memcpy_d2h(r_wse_1d, symbol_r, 0, 0, width, height, zDim,
                        streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    simulator.launch("f_toc_d2h", nonblock=False)

    total_bytes = u_wse_1d.nbytes + r_wse_1d.nbytes

    return u_wse_1d, r_wse_1d, total_bytes

def copy_timing_data(height, width, levels, simulator, symbol_timing_smooth, symbol_timing_apply_op, 
                    symbol_timing_residual, symbol_timing_restrict, symbol_timing_interp, symbol_timing_setup_init, 
                    symbol_timing_rho_check, symbol_time_total_start_end, symbol_time_h2d, symbol_time_d2h, symbol_time_ref, 
                    symbol_timing_communication, symbol_timing_compute, symbol_timing_spmv_total, symbol_timing_spmv_communication, 
                    symbol_timing_spmv_compute, args):
    """Copy timing data from device"""
    timing_smooth_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_smooth, WORDS_PER_TIMESTAMP)
    timing_residual_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_residual, WORDS_PER_TIMESTAMP)
    timing_apply_op_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_apply_op, WORDS_PER_TIMESTAMP)
    timing_restrict_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_restrict, WORDS_PER_TIMESTAMP)
    timing_interp_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_interp, WORDS_PER_TIMESTAMP)
    timing_setup_init_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_setup_init, WORDS_PER_TIMESTAMP)
    timing_rho_check_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_rho_check, WORDS_PER_TIMESTAMP)
    timing_communication_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_communication, WORDS_PER_TIMESTAMP)
    timing_compute_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_compute, WORDS_PER_TIMESTAMP)
    timing_spmv_total_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_spmv_total, WORDS_PER_TIMESTAMP)
    timing_spmv_communication_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_spmv_communication, WORDS_PER_TIMESTAMP)
    timing_spmv_compute_hwl = copy_timing(height, width, args.levels, simulator, symbol_timing_spmv_compute, WORDS_PER_TIMESTAMP)
    # EXTRA TIMES
    time_total_start_end_hwl = copy_timing(height, width, 1, simulator, symbol_time_total_start_end, WORDS_PER_TIMESTAMP)    # Not level based
    time_h2d_hwl = copy_timing(height, width, 1, simulator, symbol_time_h2d, WORDS_PER_TIMESTAMP)    # Not level based
    time_d2h_hwl = copy_timing(height, width, 1, simulator, symbol_time_d2h, WORDS_PER_TIMESTAMP)    # Not level based
    time_ref_hwl = copy_timing(height, width, 1, simulator, symbol_time_ref, WORDS_PER_TIMESTAMP)    # Not level based

    return timing_smooth_hwl, timing_residual_hwl, timing_apply_op_hwl, \
        timing_restrict_hwl, timing_interp_hwl, timing_setup_init_hwl, timing_rho_check_hwl, \
        time_total_start_end_hwl, time_h2d_hwl, time_d2h_hwl, time_ref_hwl, \
        timing_communication_hwl, timing_compute_hwl, timing_spmv_total_hwl, \
        timing_spmv_communication_hwl, timing_spmv_compute_hwl

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
                     symbol_counter_apply_op, symbol_counter_restrict, symbol_counter_interp, symbol_counter_setup_init, symbol_counter_rho_check, args):
    """Copy all operation counters from device"""
    counter_smooth_down = copy_counters(height, width, args.levels, simulator, symbol_counter_smooth_down)
    counter_smooth_up = copy_counters(height, width, args.levels, simulator, symbol_counter_smooth_up)
    counter_apply_op = copy_counters(height, width, args.levels, simulator, symbol_counter_apply_op)
    counter_restrict = copy_counters(height, width, args.levels, simulator, symbol_counter_restrict)
    counter_interp = copy_counters(height, width, args.levels, simulator, symbol_counter_interp)
    counter_setup_init = copy_counters(height, width, args.levels, simulator, symbol_counter_setup_init)
    counter_rho_check = copy_counters(height, width, args.levels, simulator, symbol_counter_rho_check)
    return counter_smooth_down, counter_smooth_up, counter_apply_op, counter_restrict, counter_interp, counter_setup_init, counter_rho_check

def print_configuration_summary(
    args,
    device_solver,
    # host_iterations,
    # host_residual,
    device_rho,
    counter_rho_check,
):
    dtype_name = DTYPE.__name__ if hasattr(DTYPE, "__name__") else str(DTYPE)
    device_iterations = int(counter_rho_check[0]) if np.ndim(counter_rho_check) > 0 and counter_rho_check.size > 0 else 0

    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)

    config_items = [
        # problem/PE size.
        ("HeightxWidthxZDim", f"{args.m}x{args.n}x{args.zDim}"),
        ("Levels", args.levels),
        ("Max iterations", args.max_ite),
        ("Tolerance (abs)", f"{device_solver.abs_tolerance:.2e}"),
        ("Tolerance (rel)", f"{device_solver.rel_tolerance:.2e}"),
        ("Tolerance (rel)^2", f"{device_solver.rel_tolerance * device_solver.rel_tolerance:.2e}"),
        ("Pre/Post/Bottom iter", f"{args.pre_iter}/{args.post_iter}/{args.bottom_iter}"),
        ("Datatype", dtype_name),
        ("Jacobi omega", f"{device_solver.omega:.6f}"),
        ("Stencil alpha/beta", f"{device_solver.ALPHA}/{device_solver.BETA}"),
        ("Channels", args.channels),
        ("Block size", args.blockSize),
        ("West/East buffer width", f"{args.width_west_buf}/{args.width_east_buf}"),
        # ("Host iterations", host_iterations),
        # ("Host final rho", f"{host_residual:.3e}"),
        ("Device iterations", device_iterations),
        ("Device final |rho|_2", f"{device_rho:.3e}")
    ]

    key_width = 32
    for label, value in config_items:
        print(f"{label:<{key_width}}: {value}")

def profiling(
    args, height, width,
    timing_smooth_hwl_levels, timing_residual_hwl_levels,
    timing_restrict_hwl_levels, timing_interp_hwl_levels,
    timing_setup_init_hwl_levels, timing_rho_check_hwl_levels,
    time_total_start_end_hwl, time_h2d_hwl, time_d2h_hwl, time_ref_hwl,
    timing_communication_hwl_levels, timing_compute_hwl_levels,
    timing_spmv_total_hwl_levels, timing_spmv_communication_hwl_levels,
    timing_spmv_compute_hwl_levels,
    counter_smooth, counter_residual, counter_restrict, counter_interp, counter_setup_init, counter_rho_check, counter_apply_op,
    WORDS_PER_TIMESTAMP, WORDS_PER_START_END, total_bytes_h2d, total_bytes_d2h
    ):
    import numpy as np

    print("\n" + "="*60)
    print("Performance Timing")
    print("="*60)

    timing_smooth_data = process_timing_data(height, width, args.levels, timing_smooth_hwl_levels, time_ref_hwl, "smooth", counter_smooth, False, WORDS_PER_TIMESTAMP)
    timing_residual_data = process_timing_data(height, width, args.levels, timing_residual_hwl_levels, time_ref_hwl, "residual", counter_residual, False, WORDS_PER_TIMESTAMP)
    timing_restrict_data = process_timing_data(height, width, args.levels, timing_restrict_hwl_levels, time_ref_hwl, "restriction", counter_restrict, False, WORDS_PER_TIMESTAMP)
    timing_interp_data = process_timing_data(height, width, args.levels, timing_interp_hwl_levels, time_ref_hwl, "interpolation", counter_interp, False, WORDS_PER_TIMESTAMP)
    timing_setup_init_data = process_timing_data(height, width, args.levels, timing_setup_init_hwl_levels, time_ref_hwl, "setup_init", counter_setup_init, False, WORDS_PER_TIMESTAMP)
    timing_rho_check_data = process_timing_data(height, width, args.levels, timing_rho_check_hwl_levels, time_ref_hwl, "rho_check", counter_rho_check, False, WORDS_PER_TIMESTAMP)
    timing_spmv_total_data = process_timing_data(height, width, args.levels, timing_spmv_total_hwl_levels, time_ref_hwl, "spmv_total", counter_apply_op, False, WORDS_PER_TIMESTAMP)
    timing_spmv_communication_data = process_timing_data(height, width, args.levels, timing_spmv_communication_hwl_levels, time_ref_hwl, "spmv_communication", counter_apply_op, False, WORDS_PER_TIMESTAMP)
    timing_spmv_compute_data = process_timing_data(height, width, args.levels, timing_spmv_compute_hwl_levels, time_ref_hwl, "spmv_compute", counter_apply_op, False, WORDS_PER_TIMESTAMP)

    counter_one = np.array([1]) # Because we have only 1 level
    ones = np.ones(args.levels, dtype=int)
    timing_total_start_end_data = process_timing_data(height, width, 1, time_total_start_end_hwl, time_ref_hwl, "total", counter_one, False, WORDS_PER_TIMESTAMP)
    timing_h2d_data = process_timing_data(height, width, 1, time_h2d_hwl, time_ref_hwl, "h2d", counter_one, False, WORDS_PER_TIMESTAMP)
    timing_d2h_data = process_timing_data(height, width, 1, time_d2h_hwl, time_ref_hwl, "d2h", counter_one, False, WORDS_PER_TIMESTAMP)
    timing_communication_data = process_timing_data(height, width, args.levels, timing_communication_hwl_levels, time_ref_hwl, "communication", ones, False, WORDS_PER_TIMESTAMP)
    timing_compute_data = process_timing_data(height, width, args.levels, timing_compute_hwl_levels, time_ref_hwl, "compute", ones, False, WORDS_PER_TIMESTAMP)

    
    print("\nTime per operation and level (us[cycles]):")
    operators = [
        ("smooth", timing_smooth_data, counter_smooth),
        ("residual", timing_residual_data, counter_residual),
        ("restriction", timing_restrict_data, counter_restrict),
        ("interpolation", timing_interp_data, counter_interp),
        ("setup_init", timing_setup_init_data, counter_setup_init),
        ("rho_check", timing_rho_check_data, counter_rho_check)
    ]

    header_cols = ["level"] + [op[0] for op in operators] + ["total"]
    col_width = 20

    def format_entry(timing_entry):
        return f"{timing_entry['time_send']:6.3f}us({timing_entry['cycles_send']:7.0f})"

    def build_divider(columns, char="-"):
        return "+" + "+".join(char * col_width for _ in columns) + "+"

    print(build_divider(header_cols, "="))
    print("|" + "|".join(f"{name:^{col_width}}" for name in header_cols) + "|")
    print(build_divider(header_cols))

    level_totals_cycles = []
    level_totals_time = []

    for level in range(args.levels):
        row_entries = [f"{level:^{col_width}}"]
        total_cycles = 0
        total_time = 0.0

        for _, timing_list, _ in operators:
            entry = timing_list[level]
            row_entries.append(f"{format_entry(entry):^{col_width}}")
            total_cycles += entry["cycles_send"]
            total_time += entry["time_send"]

        level_totals_cycles.append(total_cycles)
        level_totals_time.append(total_time)

        total_str = f"{total_time:6.3f}us({total_cycles:7.0f})"
        row_entries.append(f"{total_str:^{col_width}}")
        print("|" + "|".join(row_entries) + "|")
        print(build_divider(header_cols))

    total_row_entries = [f"{'total':^{col_width}}"]
    for _, timing_list, _ in operators:
        cycles_sum = sum(entry["cycles_send"] for entry in timing_list)
        time_sum = sum(entry["time_send"] for entry in timing_list)
        total_row_entries.append(f"{time_sum:6.3f}us({cycles_sum:7.0f})".center(col_width))

    grand_total_cycles = sum(level_totals_cycles)
    grand_total_time = sum(level_totals_time)
    total_row_entries.append(f"{grand_total_time:6.3f}us({grand_total_cycles:7.0f})".center(col_width))
    print("|" + "|".join(total_row_entries) + "|")
    print(build_divider(header_cols, "="))

    # ------------------------------------------------------------
    # Compute vs Communication Timing Summary
    # ------------------------------------------------------------
    print("\nSPMV Compute vs Communication Time per Level (us[cycles]):")
    compute_comm_header = ["level", "Total SPMV Time", "Communication Time", "Compute Time"]
    print(build_divider(compute_comm_header, "="))
    print("|" + "|".join(f"{name:^{col_width}}" for name in compute_comm_header) + "|")
    print(build_divider(compute_comm_header))


    sum_spmv_total_time = 0.0
    sum_spmv_communication_time = 0.0
    sum_spmv_compute_time = 0.0
    sum_spmv_total_cycles = 0
    sum_spmv_communication_cycles = 0
    sum_spmv_compute_cycles = 0

    for level in range(args.levels):
        spmv_total_entry = timing_spmv_total_data[level]
        spmv_communication_entry = timing_spmv_communication_data[level]
        spmv_compute_entry = timing_spmv_compute_data[level]

        spmv_total_time_level = spmv_total_entry["time_send"]
        spmv_communication_time_level = spmv_communication_entry["time_send"]
        spmv_compute_time_level = spmv_compute_entry["time_send"]

        spmv_total_cycles_level = spmv_total_entry["cycles_send"]
        spmv_communication_cycles_level = spmv_communication_entry["cycles_send"]
        spmv_compute_cycles_level = spmv_compute_entry["cycles_send"]


        row = [
            f"{level:^{col_width}}",
            f"{spmv_total_time_level:6.3f}us({spmv_total_cycles_level:7.0f})".center(col_width),
            f"{spmv_communication_time_level:6.3f}us({spmv_communication_cycles_level:7.0f})".center(col_width),
            f"{spmv_compute_time_level:6.3f}us({spmv_compute_cycles_level:7.0f})".center(col_width),
        ]
        print("|" + "|".join(row) + "|")
        print(build_divider(compute_comm_header))

        sum_spmv_total_time += spmv_total_time_level
        sum_spmv_communication_time += spmv_communication_time_level
        sum_spmv_compute_time += spmv_compute_time_level
        sum_spmv_total_cycles += spmv_total_cycles_level
        sum_spmv_communication_cycles += spmv_communication_cycles_level
        sum_spmv_compute_cycles += spmv_compute_cycles_level

    total_row = [
        f"{'total':^{col_width}}",
        f"{sum_spmv_total_time:6.3f}us({sum_spmv_total_cycles:7.0f})".center(col_width),
        f"{sum_spmv_communication_time:6.3f}us({sum_spmv_communication_cycles:7.0f})".center(col_width),
        f"{sum_spmv_compute_time:6.3f}us({sum_spmv_compute_cycles:7.0f})".center(col_width)
    ]
    print("|" + "|".join(total_row) + "|")
    print(build_divider(compute_comm_header, "="))

    
    print("\nOperation counts per level:")
    count_header_cols = ["level"] + [op[0] for op in operators]
    print(build_divider(count_header_cols, "="))
    print("|" + "|".join(f"{name:^{col_width}}" for name in count_header_cols) + "|")
    print(build_divider(count_header_cols))

    for level in range(args.levels):
        row_entries = [f"{level:^{col_width}}"]
        for _, _, counter_list in operators:
            if np.ndim(counter_list) > 0:
                count_val = int(counter_list[level])
            else:
                count_val = int(counter_list)
            row_entries.append(f"{count_val:^{col_width}}")
        print("|" + "|".join(row_entries) + "|")
        print(build_divider(count_header_cols))

    total_count_row = [f"{'total':^{col_width}}"]
    for _, _, counter_list in operators:
        if np.ndim(counter_list) > 0:
            total_count_row.append(f"{int(np.sum(counter_list)):^{col_width}}")
        else:
            total_count_row.append(f"{int(counter_list * args.levels):^{col_width}}")
    print("|" + "|".join(total_count_row) + "|")
    print(build_divider(count_header_cols, "="))

    total_time_us = grand_total_time
    total_time_cycles = grand_total_cycles

    print("=" * 100)
    print(f"Total H2D data: {total_bytes_h2d} bytes")
    print(f"Total H2D time: {timing_h2d_data[0]['time_send']:10.3f} us ({timing_h2d_data[0]['cycles_send']:10.0f} cycles)")
    print(f"Total H2D bandwidth: {(total_bytes_h2d / timing_h2d_data[0]['time_send']):.3f} MB/s")   # 1 Bytes = 1/10^6 MB and 1 us = 1/10^6 s. both cancel out
    print(f"Total D2H data: {total_bytes_d2h} bytes")
    print(f"Total D2H time: {timing_d2h_data[0]['time_send']:10.3f} us ({timing_d2h_data[0]['cycles_send']:10.0f} cycles)")
    print(f"Total D2H bandwidth: {(total_bytes_d2h / timing_d2h_data[0]['time_send']):.3f} MB/s")   # 1 Bytes = 1/10^6 MB and 1 us = 1/10^6 s. both cancel out
    print(f"Total V-cycle time (sum of operations): {total_time_us:10.3f} us ({total_time_cycles:10.0f} cycles)")
    print(f"Total V-cycle time ((No kernel launch)V-cycle time): {timing_total_start_end_data[0]['time_send']:10.3f} us ({timing_total_start_end_data[0]['cycles_send']:10.0f} cycles)")
    print("=" * 100)

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
    # host_residual, host_iterations = host_solver.solve_iterative(args.max_ite)

############################################################
# Device
############################################################
    # Initialize device
    print("\n" + "="*60)
    print("Device calculations...")
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
    symbol_timing_setup_init = simulator.get_id("timing_setup_init")
    symbol_timing_rho_check = simulator.get_id("timing_rho_check")
    symbol_timing_communication = simulator.get_id("timing_communication")
    symbol_timing_compute = simulator.get_id("timing_compute")
    symbol_time_ref = simulator.get_id("time_ref")
    symbol_time_total_start_end = simulator.get_id("time_total_start_end")
    symbol_time_h2d = simulator.get_id("time_h2d")
    symbol_time_d2h = simulator.get_id("time_d2h")
    symbol_timing_spmv_total = simulator.get_id("timing_spmv_total")
    symbol_timing_spmv_communication = simulator.get_id("timing_spmv_communication")
    symbol_timing_spmv_compute = simulator.get_id("timing_spmv_compute")
    # counters
    symbol_counter_smooth = simulator.get_id("counter_smooth")
    symbol_counter_apply_op = simulator.get_id("counter_apply_op")
    symbol_counter_residual = simulator.get_id("counter_residual")
    symbol_counter_restrict = simulator.get_id("counter_restrict")
    symbol_counter_interp = simulator.get_id("counter_interp")
    symbol_counter_setup_init = simulator.get_id("counter_setup_init")
    symbol_counter_rho_check = simulator.get_id("counter_rho_check")
    # convergence
    symbol_rho = simulator.get_id("rho")
    
    simulator.load()
    simulator.run()

############################################################
# Copy data to device
############################################################
    # Enable timing and synchronize PEs
    print("1. Enabling timer...")
    simulator.launch("f_enable_timer", nonblock=False)
    
    # Copy initial data
    # print("2. Copying initial data to device...")
    # device_solver.compute_residual(0)
    # initial_residual = device_solver.calculate_rho(device_solver.grids[0]['r'])
   # Timing measures inside the function
    total_bytes_h2d = copy_data_h2d(height, width, zDim, memcpy_dtype, memcpy_order, simulator, 
                 symbol_u, symbol_f, symbol_hx_array, symbol_hy_array, symbol_hz_array, symbol_jacobi_coeff_array, device_solver, args)
############################################################
# Kernel launch
############################################################
    print("3. Synchronizing PEs for timing...")
    simulator.launch("f_sync", nonblock=False)
    
    print("4. Copying reference clock...")
    simulator.launch("f_reference_timestamps", nonblock=False)

    # Run GMG V-cycle with convergence checking on device
    print(f"5. Running GMG V-cycle(max_iter={args.max_ite}, levels={args.levels}) on device...")
    simulator.launch("f_gmg_vcycle", 
                    # np.int16(zDim), 
                    np.int16(args.levels),
                    np.int16(args.pre_iter),
                    np.int16(args.post_iter),
                    np.int16(args.bottom_iter),
                    np.int16(args.max_ite),  # max_iter parameter
                    np.float32(device_solver.rel_tolerance),  # tolerance parameter (relative tolerance, will be squared in kernel)
                    nonblock=False)
############################################################
# Copy results and timing data back
############################################################
    # Copy results back
    print("6. Copying results from device...")
    
    print("  6.1. Copying u from device...")
    # timing measures inside the function
    u_wse_1d, r_wse_1d, total_bytes_d2h = copy_data_d2h(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u, symbol_r, device_solver, args)
    u_wse_3d = oned_to_hwl_colmajor(height, width, zDim, u_wse_1d, DTYPE)

    # Copy timing data from device
    print("  6.2. Copying timing data...")
    timing_smooth_hwl_levels, timing_residual_hwl_levels, \
        timing_apply_op_hwl_levels, timing_restrict_hwl_levels, \
            timing_interp_hwl_levels, timing_setup_init_hwl_levels, \
                timing_rho_check_hwl_levels, time_total_start_end_hwl, \
                    time_h2d_hwl, time_d2h_hwl, time_ref_hwl, timing_communication_hwl_levels, \
                        timing_compute_hwl_levels, timing_spmv_total_hwl_levels, \
                            timing_spmv_communication_hwl_levels, timing_spmv_compute_hwl_levels = \
            copy_timing_data(height, width, args.levels, simulator, symbol_timing_smooth, symbol_timing_apply_op, 
                             symbol_timing_residual, symbol_timing_restrict, symbol_timing_interp, 
                             symbol_timing_setup_init, symbol_timing_rho_check, symbol_time_total_start_end, symbol_time_h2d, symbol_time_d2h, symbol_time_ref, 
                             symbol_timing_communication, symbol_timing_compute, symbol_timing_spmv_total, symbol_timing_spmv_communication, 
                             symbol_timing_spmv_compute, args)
    print("  6.3. Copying operation counters...")
    counter_smooth, counter_residual, counter_apply_op, counter_restrict, counter_interp, counter_setup_init, counter_rho_check = \
        copy_counter_data(height, width, args.levels, simulator, 
                         symbol_counter_smooth, symbol_counter_residual, 
                         symbol_counter_apply_op, symbol_counter_restrict, 
                         symbol_counter_interp, symbol_counter_setup_init, symbol_counter_rho_check, args)
    # Copy rho (convergence metric) from device
    print("  6.4. Copying final rho from device...")
    rho_wse = np.zeros(1, np.float32)
    simulator.memcpy_d2h(rho_wse, symbol_rho, 0, 0, 1, 1, 1, streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
    rho_device = rho_wse[0]

############################################################
# Stop simulator
############################################################
    print("7. Stopping simulator...")
    simulator.stop()
    device_solver.grids[0]['u'] = u_wse_3d
    device_solver.grids[0]['rho_up'] = rho_device

############################################################
# Verification
############################################################
    # # # Verification
    # print("\n" + "="*60)
    # print("Verification")
    # print("Checking u at level 0 for host_solver and device_solver")
    # print("="*60)
  
    # # # Check/verify u, f, r, Au at level 0 for host_solver and device_solver
    # fields = ['u']
    # for field in fields:
    #     host_field = host_solver.grids[0][field]
    #     device_field = device_solver.grids[0][field]
    #     np.testing.assert_allclose(host_field.ravel(), device_field.ravel(), atol=1e-5, rtol=1e-5)
    #     stats = compare_u(host_field, device_field)
    #     print(stats)
    #     # print(f"Top-10 largest |Δ{field}| indices: {top_k_indices_absdiff(host_field, device_field, k=10)}")


    #     nrm2_u = np.linalg.norm(device_field.ravel(), 2)
    #     print(f"|{field}|_2 = {nrm2_u}")
    #     z = host_field.ravel() - device_field.ravel()
    #     nrm_z = np.linalg.norm(z, np.inf)
    #     print(f"|{field}_host - {field}_device| = {nrm_z}")
    #     print(f"\nSUCCESSFULLY VERIFIED {field} VALUES BETWEEN HOST AND DEVICE!")
############################################################
# Convergence
############################################################
    print("\n" + "="*60)
    print("Convergence")
    print("="*60)
    device_rh = device_solver.grids[0]['rho_up']

    print(f"[GMG] rho = |b-A*x|^2 = {device_rh:.6e}")
    # Use rel_tolerance^2 for convergence check (matching solve_iterative pattern)
    tolerance_squared = device_solver.rel_tolerance * device_solver.rel_tolerance
    print(f"  Tolerance^2 = {tolerance_squared:.6e}")
    converged = device_rh <= tolerance_squared
    print(f"  Converged: {'Yes' if converged else 'No'}")

############################################################
# Timing
############################################################

    profiling(args, height, width,
        timing_smooth_hwl_levels, timing_residual_hwl_levels,
        timing_restrict_hwl_levels, timing_interp_hwl_levels,
        timing_setup_init_hwl_levels, timing_rho_check_hwl_levels,
        time_total_start_end_hwl, time_h2d_hwl, time_d2h_hwl, time_ref_hwl,
        timing_communication_hwl_levels, timing_compute_hwl_levels, 
        timing_spmv_total_hwl_levels, timing_spmv_communication_hwl_levels, timing_spmv_compute_hwl_levels,
        counter_smooth, counter_residual, counter_restrict, counter_interp, counter_setup_init, counter_rho_check, counter_apply_op,
        WORDS_PER_TIMESTAMP, WORDS_PER_START_END, total_bytes_h2d, total_bytes_d2h)
    print_configuration_summary(
        args,
        device_solver,
        # host_iterations,
        # host_residual,
        device_rh,
        counter_rho_check,
    )

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

