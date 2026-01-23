#!/usr/bin/env python3
"""
Geometric Multigrid (GMG) V-cycle solver using CSL with state machine
Runs complete V-cycle on device
"""

import math
import numpy as np

# sys.path.append(os.path.join(os.path.dirname(__file__), '..', "python_gmg"))
from python_gmg.gmgoscar import SimpleGMG as SimpleGMGOSCAR
# from gmgoscar import SimpleGMG as SimpleGMGOSCAR
from cmd_parser import parse_args, print_arguments
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyOrder, MemcpyDataType
from util import csr_7_pt_stencil, hwl_2_oned_colmajor, oned_to_hwl_colmajor


DTYPE = np.float32    
WORDS_PER_TIMESTAMP = 3
WORDS_PER_START_END = WORDS_PER_TIMESTAMP * 2


def get_exponent(A: int) -> int:
    return int(math.log2(A))

def make_u48(words):
    """Convert three u16 words to 48-bit timestamp"""
    return words[0] + (words[1] << 16) + (words[2] << 32)

def copy_timing_make_48bit(height, width, simulator, symbol_timing, operation_name):
    """Copy timing data from device using same pattern as reference times"""
    timing_1d = np.zeros((height*width*3), np.uint32)
    simulator.memcpy_d2h(timing_1d, symbol_timing, 0, 0, width, height, 3,
                        streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                        order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    timing_hwl = oned_to_hwl_colmajor(height, width, 3, timing_1d, np.uint16)
    # Extract timestamp from PE(0,0) and convert to cycles/time
    words = timing_hwl[0, 0, :]
    cycles_send = make_u48(words)
    time_send = (cycles_send / 0.875) * 1.e-3
    return {
        'operation': operation_name,
        'cycles_send': cycles_send,
        'time_send': time_send,
        'timing_hwl': timing_hwl
    }


def copy_counters(height, width, levels, simulator, symbol_counter):
    """Copy operation counter data from device using same pattern as reference times"""
    counter_1d = np.zeros(1*1*levels, dtype=np.uint32)
    simulator.memcpy_d2h(counter_1d, symbol_counter, 0, 0, 1, 1, levels,
                        streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, 
                        order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    # counter_hwl = oned_to_hwl_colmajor(height, width, levels, counter_1d, np.uint32)
    # Extract counters from PE(0,0)
    # counter_values = counter_hwl[0, 0, :]
    return counter_1d

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

    # Initialize device
    print("\n" + "="*60)
    print("Device calculations...")
    print("="*60)
    
    memcpy_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.COL_MAJOR
    simulator = SdkRuntime(logs_dir, cmaddr=args.cmaddr, simfab_numthreads=64)
   
    # timing
    symbol_timing_smooth = simulator.get_id("timing_smooth")
    symbol_timing_residual = simulator.get_id("timing_residual")
    symbol_timing_restrict = simulator.get_id("timing_restrict")
    symbol_timing_interp = simulator.get_id("timing_interp")
    symbol_time_total_start_end = simulator.get_id("time_total_start_end")
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
    # symbol_rho = simulator.get_id("rho")
    symbol_rho_history = simulator.get_id("rho_history")
    
    simulator.load()
    simulator.run()

    print("  6.0. Initializing timers and counters...")
    simulator.launch("f_init_timers_and_counters", nonblock=False)
############################################################
# Copy data to device
############################################################
    print("  6.1. Copying timing data...")
    timing_smooth_data = copy_timing_make_48bit(height, width, simulator, symbol_timing_smooth, "smooth")
    timing_residual_data = copy_timing_make_48bit(height, width, simulator, symbol_timing_residual, "residual")
    timing_restrict_data = copy_timing_make_48bit(height, width, simulator, symbol_timing_restrict, "restriction")
    timing_interp_data = copy_timing_make_48bit(height, width, simulator, symbol_timing_interp, "interpolation")
    timing_spmv_total_data = copy_timing_make_48bit(height, width, simulator, symbol_timing_spmv_total, "spmv_total")
    timing_spmv_communication_data = copy_timing_make_48bit(height, width, simulator, symbol_timing_spmv_communication, "spmv_communication")
    timing_spmv_compute_data = copy_timing_make_48bit(height, width, simulator, symbol_timing_spmv_compute, "spmv_compute")
    timing_total_start_end_data = copy_timing_make_48bit(height, width, simulator, symbol_time_total_start_end, "total")

    # Print timing data summary
    print(f"timing_smooth_data: {timing_smooth_data['time_send']:.3f}us ({timing_smooth_data['cycles_send']} cycles)")
    print(f"timing_residual_data: {timing_residual_data['time_send']:.3f}us ({timing_residual_data['cycles_send']} cycles)")
    print(f"timing_restrict_data: {timing_restrict_data['time_send']:.3f}us ({timing_restrict_data['cycles_send']} cycles)")
    print(f"timing_interp_data: {timing_interp_data['time_send']:.3f}us ({timing_interp_data['cycles_send']} cycles)")
    print(f"timing_spmv_total_data: {timing_spmv_total_data['time_send']:.3f}us ({timing_spmv_total_data['cycles_send']} cycles)")
    print(f"timing_spmv_communication_data: {timing_spmv_communication_data['time_send']:.3f}us ({timing_spmv_communication_data['cycles_send']} cycles)")
    print(f"timing_spmv_compute_data: {timing_spmv_compute_data['time_send']:.3f}us ({timing_spmv_compute_data['cycles_send']} cycles)")
    print(f"timing_total_start_end_data: {timing_total_start_end_data['time_send']:.3f}us ({timing_total_start_end_data['cycles_send']} cycles)")

    print("  6.2. Copying operation counters...")
    counter_smooth = copy_counters(height, width, args.levels, simulator, symbol_counter_smooth)
    counter_residual = copy_counters(height, width, args.levels, simulator, symbol_counter_residual)
    counter_setup_init = copy_counters(height, width, args.levels, simulator, symbol_counter_setup_init) # hang
    counter_restrict = copy_counters(height, width, args.levels, simulator, symbol_counter_restrict)
    counter_apply_op = copy_counters(height, width, args.levels, simulator, symbol_counter_apply_op)
    counter_interp = copy_counters(height, width, args.levels, simulator, symbol_counter_interp)
    counter_rho_check = copy_counters(height, width, args.levels, simulator, symbol_counter_rho_check)  # This line breaks!
    print(f"counter_smooth: {counter_smooth}")
    print(f"counter_residual: {counter_residual}")
    print(f"counter_setup_init: {counter_setup_init}")
    print(f"counter_restrict: {counter_restrict}")
    print(f"counter_apply_op: {counter_apply_op}")
    print(f"counter_interp: {counter_interp}")
    print(f"counter_rho_check: {counter_rho_check}")


    simulator.stop()

if __name__ == "__main__":
    main()

