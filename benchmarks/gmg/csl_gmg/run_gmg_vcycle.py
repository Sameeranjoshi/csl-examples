#!/usr/bin/env python3
"""
Geometric Multigrid (GMG) V-cycle solver using CSL with state machine
Runs complete V-cycle on device
"""

import math
from pathlib import Path
import shutil
import copy
import numpy as np
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', "python_gmg"))
# from python_gmg.gmgoscar import SimpleGMG as SimpleGMGOSCAR
from gmgoscar import SimpleGMG as SimpleGMGOSCAR
from cmd_parser import parse_args, print_arguments
from util import hwl_2_oned_colmajor, oned_to_hwl_colmajor
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyOrder, MemcpyDataType

DTYPE = np.float32    
WORDS_PER_TIMESTAMP = 3
WORDS_PER_START_END = WORDS_PER_TIMESTAMP * 2

def get_exponent(A: int) -> int:
    return int(math.log2(A))

def make_u48(words):
    """Convert three u16 words to 48-bit timestamp"""
    return words[0] + (words[1] << 16) + (words[2] << 32)

def print_2d(matrix):
    for row in matrix:
        print(" | ".join(f"{v:8.2f}" for v in row))

def copy_data_d2h(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u, symbol_r, symbol_rho, symbol_rho_history, args):
    pass
    # u_wse_1d = np.zeros(height*width*zDim, DTYPE)
    # r_wse_1d = np.zeros(height*width*zDim, DTYPE)
    # rho_wse = np.zeros(1, np.float32)
    # simulator.memcpy_d2h(u_wse_1d, symbol_u, 0, 0, width, height, zDim,
    #                     streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    # simulator.memcpy_d2h(r_wse_1d, symbol_r, 0, 0, width, height, zDim,
    #                     streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    # simulator.memcpy_d2h(rho_wse, symbol_rho, 0, 0, 1, 1, 1, 
    #                     streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    # # rho_history will be copied later after counter is available
    # rho_history_array = np.array([], dtype=np.float32)
    # total_bytes = u_wse_1d.nbytes + r_wse_1d.nbytes + rho_wse.nbytes
    # return u_wse_1d, r_wse_1d, rho_wse[0], rho_history_array, total_bytes

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
    # Match host formula: diagonal = -2*(1/hx² + 1/hy² + 1/hz²)
    # diag_inv = 1/diagonal = -1/(2*(1/hx² + 1/hy² + 1/hz²))
    # update = omega * diag_inv * (f - Au) = omega * (-1/(2*(1/hx² + 1/hy² + 1/hz²))) * (f - Au)
    # So jacobi_coeff = omega * (-1/(2*(1/hx² + 1/hy² + 1/hz²))) = -omega/(2*(1/hx² + 1/hy² + 1/hz²))
    jacobi_coeff_base = np.zeros(args.levels, dtype=DTYPE)
    omega = device_solver.omega  # 0.75
    for level in range(args.levels):
        hx = device_solver.grids[level]['hx']
        hy = device_solver.grids[level]['hy']
        hz = device_solver.grids[level]['hz']
        jacobi_coeff_base[level] = -omega / (2.0 * (1.0/(hx*hx) + 1.0/(hy*hy) + 1.0/(hz*hz)))
    
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
    # Debug: Print what we're sending
    print(f"Grid spacing and Jacobi coefficients:")
    for level in range(args.levels):
        print(f"  Level {level}: hx={hx_base[level]:.6f}, hy={hy_base[level]:.6f}, hz={hz_base[level]:.6f}, jacobi={jacobi_coeff_base[level]:.6e}")
    
    # Copy spacing/jacobi arrays: flatten in COLUMN-MAJOR order to match memcpy
    hx_flat = hwl_2_oned_colmajor(height, width, args.levels, hx_array, DTYPE)
    hy_flat = hwl_2_oned_colmajor(height, width, args.levels, hy_array, DTYPE)
    hz_flat = hwl_2_oned_colmajor(height, width, args.levels, hz_array, DTYPE)
    jacobi_flat = hwl_2_oned_colmajor(height, width, args.levels, jacobi_coeff_array, DTYPE)

    # simulator.launch("f_tic_h2d", nonblock=True)
    # Copy u and f arrays
    simulator.memcpy_h2d(symbol_u, u_1d, 0, 0, width, height, zDim,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=True)
    simulator.memcpy_h2d(symbol_f, f_1d, 0, 0, width, height, zDim,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=True)
    simulator.memcpy_h2d(symbol_hx_array, hx_flat, 0, 0, width, height, args.levels,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=True)
    simulator.memcpy_h2d(symbol_hy_array, hy_flat, 0, 0, width, height, args.levels,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=True)
    simulator.memcpy_h2d(symbol_hz_array, hz_flat, 0, 0, width, height, args.levels,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=True)
    simulator.memcpy_h2d(symbol_jacobi_coeff_array, jacobi_flat, 0, 0, width, height, args.levels,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=True)
    # simulator.launch("f_toc_h2d", nonblock=False)
    total_bytes = (
        u_1d.nbytes
        + f_1d.nbytes
        + hx_flat.nbytes
        + hy_flat.nbytes
        + hz_flat.nbytes
        + jacobi_flat.nbytes
    )

    return total_bytes

def copy_timing_make_48bit(height, width, levels, simulator, symbol_timing, words_per_entry, operation_name):
    """
    Copy timing data from device and extract per-level 48-bit timestamps.
    All simulator.memcpy_d2h operations must use data_type=MEMCPY_32BIT, then process as needed.
    """
    timing_size = words_per_entry * levels
    timing_1d = np.zeros(height*width*timing_size, np.uint32)
    simulator.memcpy_d2h(
        timing_1d, symbol_timing, 0, 0, width, height, timing_size,
        streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.COL_MAJOR, nonblock=False
    )
    # Repack to work with oned_to_hwl_colmajor as before, using uint16 version
    timing_hwl = oned_to_hwl_colmajor(width, height, timing_size, timing_1d, np.uint16)
    per_pe_time_levels = timing_hwl[0, 0, :]  # All levels' words packed for PE(0,0)
    # if operation_name == "restriction":
    #     print("restriction PE0,0 = per_pe_time_levels", per_pe_time_levels)
    timing_per_level = []
    for level in range(levels):
        base = level * words_per_entry
        level_time = per_pe_time_levels[base:base+3]
        if len(level_time) < 3:
            continue
        cycles_send = make_u48(level_time)
        time_send = (cycles_send / 0.875) * 1.e-3
        timing_per_level.append({
            'level': level,
            'operation': operation_name,
            'cycles_send': cycles_send,
            'time_send': time_send
        })
    return timing_per_level

def copy_counters(height, width, levels, simulator, symbol_counter):
    """Copy operation counter data from device - optimized to copy only minimal region (2x1) since only PE(0,0) is used"""

    counter_1d = np.zeros(1*1*levels, dtype=np.uint32)
    simulator.memcpy_d2h(counter_1d, symbol_counter, 0, 0, 1, 1, levels,
                        streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, 
                        order=MemcpyOrder.COL_MAJOR, nonblock=False)
    counter_hwl = oned_to_hwl_colmajor(1, 1, levels, counter_1d, np.uint16)
    per_pe_time_levels = counter_hwl[0, 0, :]  # All levels' words packed for PE(0,0)                        

    return per_pe_time_levels

def copy_rho_history(height, width, memcpy_dtype, memcpy_order, simulator, symbol_rho_history, counter_rho_check, args):
    """Copy rho_history from device using counter to determine how many elements to copy"""
    
    # actual_iterations = int(counter_rho_check[0, 0]) if np.ndim(counter_rho_check) > 0 and counter_rho_check.size > 0 else 0
    actual_iterations = int(counter_rho_check[0])
    
    rho_history_1d = np.zeros(1 * 1 * actual_iterations, np.float32)
    simulator.memcpy_d2h(rho_history_1d, symbol_rho_history, 0, 0, 1, 1, actual_iterations,
                        streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

    return rho_history_1d, actual_iterations

def print_configuration_summary(
    args,
    device_solver,
    # host_iterations,
    # host_residual,
    device_rho,
    counter_rho_check,
    timing_total_start_end_data,
    ):
    dtype_name = DTYPE.__name__ if hasattr(DTYPE, "__name__") else str(DTYPE)
    # counter_rho_check has shape (1, levels), so access [0, 0] for first level's count
    # device_iterations = int(counter_rho_check[0, 0]) if np.ndim(counter_rho_check) > 0 and counter_rho_check.size > 0 else 0
    # print("ndim", np.ndim(counter_rho_check))
    device_iterations = int(counter_rho_check[0])

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
        ("Pre/Post/Bottom iter", f"{args.pre_iter}/{args.post_iter}/{args.bottom_iter}"),
        ("Datatype", dtype_name),
        ("Jacobi omega", f"{device_solver.omega:.6f}"),
        ("Stencil alpha/beta", f"{device_solver.ALPHA}/{device_solver.BETA}"),
        ("Channels", args.channels),
        ("Block size", args.blockSize),
        ("West/East buffer width", f"{args.width_west_buf}/{args.width_east_buf}"),
        # ("Host iterations", host_iterations),
        # ("Host final rho", f"{host_residual:.3e}"),
        ("Total Solver time (us[cycles])", f"{timing_total_start_end_data['time_send']:10.3f}us({timing_total_start_end_data['cycles_send']:10.0f})"),
        ("Device iterations", device_iterations),
        ("1-V cycle time(Average) (us[cycles])", f"{timing_total_start_end_data['time_send']/device_iterations:10.3f}us({timing_total_start_end_data['cycles_send']/device_iterations:10.0f}cycles)"),
        ("Device final |rho|_inf", f"{device_rho:.3e}"),
    ]

    key_width = 32
    for label, value in config_items:
        print(f"{label:<{key_width}}: {value}")

def profiling(
        args,
        WORDS_PER_TIMESTAMP,
        device_solver, device_rh,
        counter_rho_check,
        simulator, 
        timing_smooth_data, timing_residual_data, timing_restrict_data, timing_interp_data, timing_spmv_total_data, timing_spmv_communication_data, timing_spmv_compute_data, timing_total_start_end_data
    ):
    print("\n" + "="*60)
    print("Performance Timing")
    print("="*60)

    ############################################################
    # Time per operation and level (us[cycles]):
    ############################################################
    
    print("\nTime per operation and level (us[cycles]):")
    operators = [
        ("smooth", timing_smooth_data),
        ("residual", timing_residual_data),
        ("restriction", timing_restrict_data),
        ("interpolation", timing_interp_data),
        # ("setup_init", timing_setup_init_data, counter_setup_init),
        # ("rho_check", timing_rho_check_data, counter_rho_check)
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

        for _, timing_list in operators:
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
    for _, timing_list in operators:
        cycles_sum = sum(entry["cycles_send"] for entry in timing_list)
        time_sum = sum(entry["time_send"] for entry in timing_list)
        total_row_entries.append(f"{time_sum:6.3f}us({cycles_sum:7.0f})".center(col_width))

    grand_total_cycles = sum(level_totals_cycles)
    grand_total_time = sum(level_totals_time)
    total_row_entries.append(f"{grand_total_time:6.3f}us({grand_total_cycles:7.0f})".center(col_width))
    print("|" + "|".join(total_row_entries) + "|")
    print(build_divider(header_cols, "="))
    total_time_us = grand_total_time
    total_time_cycles = grand_total_cycles

    # ------------------------------------------------------------
    # Compute vs Communication Timing Summary(7-pt Stencil)
    # ------------------------------------------------------------
    print("\n7-pt Stencil Compute vs Communication Time per Level (us[cycles]):")
    print("Note: Triangle inequality: max(x+y) <= max(x) + max(y), where x,y are 2D timings across wafer.")
    compute_comm_header = ["level", "Total SpMV Time", "Communication Time", "Compute Time", "L1/L0(Communication)", "L1-L0(Communication)", "L1/L0(Compute)"]
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

        # Calculate ratio: level(i+1)/level(i)
        if level + 1 < args.levels:
            next_comm_time = timing_spmv_communication_data[level + 1]["time_send"]
            if spmv_communication_time_level > 0:
                comm_ratio = next_comm_time / spmv_communication_time_level
                ratio_str = f"{comm_ratio:6.3f}".center(col_width)
            else:
                ratio_str = f"{'N/A':^{col_width}}"
            # Calculate difference: L(i+1) - L(i)
            comm_diff = next_comm_time - spmv_communication_time_level
            diff_str = f"{comm_diff:6.3f}us".center(col_width)
            # Compute ratio: compute[i+1]/compute[i]
            next_compute_time = timing_spmv_compute_data[level + 1]["time_send"]
            if spmv_compute_time_level > 0:
                compute_ratio = next_compute_time / spmv_compute_time_level
                compute_ratio_str = f"{compute_ratio:6.3f}".center(col_width)
            else:
                compute_ratio_str = f"{'N/A':^{col_width}}"
        else:
            ratio_str = f"{'N/A':^{col_width}}"
            diff_str = f"{'N/A':^{col_width}}"
            compute_ratio_str = f"{'N/A':^{col_width}}"

        row = [
            f"{level:^{col_width}}",
            f"{spmv_total_time_level:6.3f}us({spmv_total_cycles_level:7.0f})".center(col_width),
            f"{spmv_communication_time_level:6.3f}us({spmv_communication_cycles_level:7.0f})".center(col_width),
            f"{spmv_compute_time_level:6.3f}us({spmv_compute_cycles_level:7.0f})".center(col_width),
            ratio_str,
            diff_str,
            compute_ratio_str,
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
        f"{sum_spmv_compute_time:6.3f}us({sum_spmv_compute_cycles:7.0f})".center(col_width),
        f"{'N/A':^{col_width}}",
        f"{'N/A':^{col_width}}",
        f"{'N/A':^{col_width}}"
    ]
    print("|" + "|".join(total_row) + "|")
    print(build_divider(compute_comm_header, "="))

    ############################################################
    # Total time and bandwidth:
    ############################################################
    print("=" * 100)
    print(f"Total Upper bound V-cycle time (sum of operations): {total_time_us:10.3f} us ({total_time_cycles:10.0f} cycles)")
    print(f"Total V-cycle time (Kernel Launch + V-cycle time): {timing_total_start_end_data[0]['time_send']:10.3f} us ({timing_total_start_end_data[0]['cycles_send']:10.0f} cycles)")
    print(f"Choose max")
    print("=" * 100)

    print_configuration_summary(
        args,
        device_solver,
        # host_iterations,
        # host_residual,
        device_rh,
        counter_rho_check,
        timing_total_start_end_data[0]
    )

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
    symbol_timing_smooth = simulator.get_id("timing_smooth")
    symbol_timing_residual = simulator.get_id("timing_residual")
    symbol_timing_restrict = simulator.get_id("timing_restrict")
    symbol_timing_interp = simulator.get_id("timing_interp")
    symbol_time_total_start_end = simulator.get_id("time_total_start_end")
    symbol_timing_spmv_total = simulator.get_id("timing_spmv_total")
    symbol_timing_spmv_communication = simulator.get_id("timing_spmv_communication")
    symbol_timing_spmv_compute = simulator.get_id("timing_spmv_compute")    
    symbol_counter_rho_check = simulator.get_id("counter_rho_check")
    symbol_rho_history = simulator.get_id("rho_history")
    
    simulator.load()
    simulator.run()

    ############################################################
    # Warmup + reset
    print("1. Warmup(3 iterations) + reset...")
    ############################################################
    warmup_iterations = 3
    simulator.launch("f_gmg_vcycle", 
                    # np.int16(zDim), 
                    np.int16(args.levels),
                    np.int16(args.pre_iter),
                    np.int16(args.post_iter),
                    np.int16(args.bottom_iter),
                    np.int16(warmup_iterations),  # max_iter parameter
                    np.float32(device_solver.rel_tolerance),  # tolerance parameter (relative tolerance, will be squared in kernel)
                    nonblock=False)
    simulator.launch("f_reset", nonblock=False)
    
    ############################################################
    # Copy data to device
    print("2. Copying initial data to device...")
    ############################################################
   # Timing measures inside the function
    copy_data_h2d(height, width, zDim, memcpy_dtype, memcpy_order, simulator, 
                 symbol_u, symbol_f, symbol_hx_array, symbol_hy_array, symbol_hz_array, symbol_jacobi_coeff_array, device_solver, args)

    ############################################################
    # Kernel launch
    ############################################################
    # Enable timing and synchronize PEs
    print("3. Enabling timer...")
    simulator.launch("f_enable_timer", nonblock=False)
    print("4. Synchronizing PEs...")
    simulator.launch("f_sync", nonblock=False)
    print("5. Tic total...")
    simulator.launch("f_tic_total", nonblock=True)
    print(f"6. Running GMG V-cycle(max_iter={args.max_ite}, levels={args.levels}) on device...")
    simulator.launch("f_gmg_vcycle", 
                    np.int16(args.levels),
                    np.int16(args.pre_iter),
                    np.int16(args.post_iter),
                    np.int16(args.bottom_iter),
                    np.int16(args.max_ite),  # max_iter parameter
                    np.float32(device_solver.rel_tolerance),  # tolerance parameter (relative tolerance, will be squared in kernel)
                    nonblock=False)
    print("7. Toc total...")
    simulator.launch("f_toc_total", nonblock=False)

    ############################################################
    # Copy timers and counters
    print("8. Copying timers and counters from device...")
    ############################################################
    timing_smooth_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_smooth, WORDS_PER_TIMESTAMP, "smooth")
    timing_residual_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_residual, WORDS_PER_TIMESTAMP, "residual")
    timing_restrict_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_restrict, WORDS_PER_TIMESTAMP, "restriction")
    timing_interp_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_interp, WORDS_PER_TIMESTAMP, "interpolation")
    timing_spmv_total_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_spmv_total, WORDS_PER_TIMESTAMP, "spmv_total")
    timing_spmv_communication_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_spmv_communication, WORDS_PER_TIMESTAMP, "spmv_communication")
    timing_spmv_compute_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_spmv_compute, WORDS_PER_TIMESTAMP, "spmv_compute")
    # counter_one = np.array([1]) # Because we have only 1 level
    # ones = np.ones(args.levels, dtype=int)
    timing_total_start_end_data = copy_timing_make_48bit(height, width,1, simulator, symbol_time_total_start_end, WORDS_PER_TIMESTAMP, "total")

    counter_rho_check = copy_counters(height, width, args.levels, simulator, symbol_counter_rho_check)  # This line breaks!
    
    ###########################################################
    # Convergence
    print("9. Checking convergence...")
    ###########################################################
    # Copy rho_history after counter is available to minimize data transfer
    rho_history_array, actual_iterations = copy_rho_history(height, width, memcpy_dtype, memcpy_order, simulator, symbol_rho_history, counter_rho_check, args)
    if actual_iterations > 0:
        device_solver.grids[0]['rho_up'] = rho_history_array[actual_iterations - 1]
    else:
        device_solver.grids[0]['rho_up'] = 0.0

    device_rh = device_solver.grids[0]['rho_up']

    # # Print rho values after each iteration
    print("\n" + "="*60)
    print("Rho values after each iteration")
    print("="*60)
    if len(rho_history_array) > 0:
        if actual_iterations > 0:
            print(f"Total iterations performed: {actual_iterations}")
            print(f"{'Iteration':<12} {'|rho|_max':<20}")
            print("-" * 60)
            for i in range(actual_iterations):
                rho_val = rho_history_array[i]
                print(f"{i+1:<12} {rho_val:>19.6e}")
        else:
            print("No iterations were performed.")
    else:
        print("No iterations were performed.")

    print(f"[GMG] rho = |b-A*x|_inf = {device_rh:.6e}")
    # Use rel_tolerance^2 for convergence check (matching solve_iterative pattern)
    tolerance = device_solver.rel_tolerance
    print(f"  Tolerance = {tolerance:.6e}")
    converged = device_rh <= tolerance
    print(f"  Converged: {'Yes' if converged else 'No'}")

    ############################################################
    # Profiling
    print("10. Profiling...")
    ############################################################

    # counter_rho_check = 0
    # device_rh = 0.0
    profiling(args,
        WORDS_PER_TIMESTAMP,
        device_solver, device_rh,
        counter_rho_check,
        simulator, 
        timing_smooth_data, timing_residual_data, timing_restrict_data, timing_interp_data, timing_spmv_total_data, timing_spmv_communication_data, timing_spmv_compute_data, timing_total_start_end_data
        )  
    
    ###########################################################
    # Clean up
    print("10. Stopping simulator...")
    ###########################################################
    simulator.stop()
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

