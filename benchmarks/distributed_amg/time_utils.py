import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
from typing import Dict, Tuple, Optional
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


def float_to_hex(f):
  return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def cast_uint32(x):
  if isinstance(x, (np.float16, np.int16, np.uint16)):
    z = x.view(np.uint16)
    val = np.uint32(z)
  elif isinstance(x, (np.float32, np.int32, np.uint32)):
    val = x.view(np.uint32)
  elif isinstance(x, int):
    val = np.uint32(x)
  elif isinstance(x, float):
    z = np.float32(x)
    val = z.view(np.uint32)
  else:
    raise RuntimeError(f"type of x {type(x)} is not supported")

  return val

def time_analysis_noref(height, width, time_memcpy_hwl, time_ref_hwl):
    # time_start = start time of H2D/D2H
    time_start = np.zeros((height, width)).astype(int)
    # time_end = end time of H2D/D2H
    time_end = np.zeros((height, width)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(width):
        for h in range(height):
            hex_t0 = int(float_to_hex(time_memcpy_hwl[(h, w, 0)]), base=16)
            hex_t1 = int(float_to_hex(time_memcpy_hwl[(h, w, 1)]), base=16)
            hex_t2 = int(float_to_hex(time_memcpy_hwl[(h, w, 2)]), base=16)
            word[0] = hex_t0 & 0x0000FFFF
            word[1] = (hex_t0 >> 16) & 0x0000FFFF
            word[2] = hex_t1 & 0x0000FFFF
            time_start[(h, w)] = make_u48(word)
            word[0] = (hex_t1 >> 16) & 0x0000FFFF
            word[1] = hex_t2 & 0x0000FFFF
            word[2] = (hex_t2 >> 16) & 0x0000FFFF
            time_end[(h, w)] = make_u48(word)

    # time_ref = reference clock
    time_ref = np.zeros((height, width)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(width):
        for h in range(height):
            hex_t0 = int(float_to_hex(time_ref_hwl[(h, w, 0)]), base=16)
            hex_t1 = int(float_to_hex(time_ref_hwl[(h, w, 1)]), base=16)
            word[0] = hex_t0 & 0x0000FFFF
            word[1] = (hex_t0 >> 16) & 0x0000FFFF
            word[2] = hex_t1 & 0x0000FFFF
            time_ref[(h, w)] = make_u48(word)
    # adjust the reference clock by the propagation delay, this is the manhatten distance for grid.
    for py in range(height):
        for px in range(width):
            time_ref[(py, px)] = time_ref[(py, px)] - ((width+height-2)-(px + py))

    # print(time_start)
    # print(time_end)
    # print(time_ref)
    # # shift time_start and time_end by time_ref
    time_start = time_start - time_ref
    time_end = time_end - time_ref


    # cycles_send = time_end[(h,w)] - time_start[(h,w)]
    # 850MHz --> 1 cycle = (1/0.85) ns = (1/0.85)*1.e-3 us
    # time_send = (cycles_send / 0.85) *1.e-3 us
    # bandwidth = (((wvlts-1) * 4)/time_send) MBS
    # wvlts = pw * ph * pe_length
    # Find the earliest start time across all PEs
    min_time_start = time_start.min()
    
    # Find the latest end time across all PEs
    max_time_end = time_end.max() 
    
    # Calculate total cycles from first PE starting to last PE finishing
    cycles_send = max_time_end - min_time_start
    
    # Convert cycles to microseconds:
    # - Hardware runs at 850MHz (0.85 GHz)
    # - 1 cycle = 1/0.85 nanoseconds
    # - Multiply by 1e-3 to convert nanoseconds to microseconds
    time_send = (cycles_send / 0.85) * 1.0e-3
    return {
        'cycles': cycles_send,  # Seems like some bug was introduced here using np.mean()
        'kernel_time_us': time_send,
        # 'time_start': time_start,
        # 'time_end': time_end,
    }
         
def timing_analysis_2d(height, width, time_memcpy_hwl, time_ref_hwl):
    """
    Timing analysis for 2D AMG problem
    Args:
        height: Number of rows in the 2D grid
        width: Number of columns in the 2D grid
        time_memcpy_hwl: Timestamp data for memory operations (h,w,l format)
        time_ref_hwl: Reference clock data (h,w,l format)
    """
    # Initialize timing arrays for 2D grid
    time_start = np.zeros((height, width)).astype(int)
    time_end = np.zeros((height, width)).astype(int)
    # Initialize reference clock array
    time_ref = np.zeros((height, width)).astype(int)
    word = np.zeros(3).astype(np.uint16)

    # Extract start and end times for each PE in 2D grid
    for h in range(height):
        for w in range(width):  # ROW MAJOR ORDER   
            # Get start time
            word[0] = time_memcpy_hwl[(h, w, 0)]
            word[1] = time_memcpy_hwl[(h, w, 1)]
            word[2] = time_memcpy_hwl[(h, w, 2)]
            time_start[(h,w)] = make_u48(word)
            
            # Get end time
            word[0] = time_memcpy_hwl[(h, w, 3)]
            word[1] = time_memcpy_hwl[(h, w, 4)]
            word[2] = time_memcpy_hwl[(h, w, 5)]
            time_end[(h,w)] = make_u48(word)

            # Extract reference clock for each PE
            word[0] = time_ref_hwl[(h, w, 0)]
            word[1] = time_ref_hwl[(h, w, 1)]
            word[2] = time_ref_hwl[(h, w, 2)]
            time_ref[(h, w)] = make_u48(word)

    # Adjust reference clock for 2D grid propagation delay(f_sync)
    # The right-bottom PE (h=height-1, w=width-1) is the reference PE that signals other PEs
    # The signal propagates one hop per cycle to reach other PEs
    # Manhattan distance measures the minimum number of hops needed from reference PE to each PE:
    #   - From (height-1, width-1) to (h,w) takes |(height-1)-h| + |(width-1)-w| cycles
    #   - Simplified to: (height-1-h) + (width-1-w) since reference PE is at max indices
    # Example: For a 4x4 grid, PE(0,0) is 6 hops from reference PE(3,3): |3-0| + |3-0| = 6 cycles
    for h in range(height):
        for w in range(width):
            manhattan_distance = (height - 1 - h) + (width - 1 - w)
            time_ref[(h, w)] = time_ref[(h, w)] - manhattan_distance

    # Adjust timestamps relative to reference clock
    time_start = time_start - time_ref
    time_end = time_end - time_ref

    # Calculate timing metrics
    min_time_start = time_start.min()
    max_time_end = time_end.max()
    cycles_total = max_time_end - min_time_start
    
    # Convert cycles to time (850MHz clock)
    time_us = (cycles_total / 0.85) * 1e-3  # Convert to microseconds
    
    return {
        'cycles': np.mean(cycles_total),
        'time_us': np.mean(time_us),
        'time_start': time_start,
        'time_end': time_end,
        'time_ref': time_ref
    }

def write_timing_data(timing_map, filename="timing_data.csv"):
    """Write timing data to CSV file."""
    rows = []
    for iteration, direction_data in timing_map.items():
        for direction, level_data in direction_data.items():
            for level, timing_data in level_data.items():
                rows.append({
                    'Iteration': iteration,
                    'Direction': direction,
                    'Level': level,
                    'Cycles': timing_data['cycles'],
                    'Time (us)': timing_data['time_us']
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Timing data written to {filename}")
    return df

def write_performance_data(perf_metrics, filename="timing_data.csv"):
    perf_metrics_list = [perf_metrics]
    df = pd.DataFrame(perf_metrics_list)
    
    # Check if file exists to determine mode and header
    file_exists = os.path.isfile(filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Write DataFrame to CSV file
    with open(filename, 'a' if file_exists else 'w') as f:
        if not file_exists:
            df.to_csv(f, index=False)
        else:
            df.to_csv(f, index=False, header=False)

    action = "appended to" if file_exists else "written to"
    print(f"\tTiming data {action} {filename}")
    return df

# AMG timing data class
@dataclass
class OperatorTiming:
    """Timing data for AMG operators"""
    smoothing_time: float = 0.0
    residual_time: float = 0.0
    restriction_time: float = 0.0
    prolongation_time: float = 0.0
    matrix_multiply_time: float = 0.0
    total_time: float = 0.0



@dataclass
class HostProfiling:
    # Structure: iteration -> level -> direction -> timing
    timings: Dict[int, Dict[int, Dict[str, OperatorTiming]]] = None
    current_iteration: int = 0
    current_level: int = 0
    current_direction: str = ""

    def __init__(self):
        self.timings = {}
        
    def add_timing(self, level_index: int, timing: OperatorTiming, direction: str):
        """Add timing data for current iteration, level and direction"""
        if self.current_iteration not in self.timings:
            self.timings[self.current_iteration] = {}
            
        if level_index not in self.timings[self.current_iteration]:
            self.timings[self.current_iteration][level_index] = {}
            
        self.timings[self.current_iteration][level_index][direction] = timing

    def add_other_info(self, iteration: int, level_index: int, level_direction: str):
        """Update current iteration, level and direction context"""
        self.current_iteration = iteration
        self.current_level = level_index
        self.current_direction = level_direction
        
    def print_timing_summary(self):
        """Print a summary of timing information and save to CSV for plotting"""
        # Print formatted table using tabulate
        from tabulate import tabulate
        import pandas as pd
        import json
        from pathlib import Path

        # Convert timing data to list of dictionaries for DataFrame
        timing_records = []
        for iteration in sorted(self.timings.keys()):
            for level in sorted(self.timings[iteration].keys()):
                level_timings = self.timings[iteration][level]
                
                record = {
                    'iteration': iteration,
                    'level': level
                }

                if "down" in level_timings:
                    down = level_timings["down"]
                    record.update({
                        'down_pre_smooth': down.smoothing_time,
                        'down_residual': down.residual_time,
                        'down_restrict': down.restriction_time,
                        'down_total': down.total_time
                    })

                if "up" in level_timings:
                    up = level_timings["up"]
                    record.update({
                        'up_prolong': up.prolongation_time,
                        'up_post_smooth': up.smoothing_time,
                        'up_total': up.total_time
                    })

                timing_records.append(record)

        # Create DataFrame
        df = pd.DataFrame(timing_records)

        # Generate analysis and plots
        cpu_output_path = "timing_results/cpu/plots"
        summary = analyze_and_plot_timings(df, output_path=cpu_output_path)
        
        # Print formatted table
        print("\nAMG Host Timing Summary:")
        headers = ['Iteration', 'Level', 'Down Pre-Smooth', 'Down Residual', 
                  'Down Restrict', 'Down Total', 'Up Prolong', 'Up Post-Smooth', 'Up Total']
        if len(df) > 10:
            print(tabulate(df.head(10), headers=headers, floatfmt='.6f', tablefmt='grid'))
            print("...")
        else:
            print(tabulate(df, headers=headers, floatfmt='.6f', tablefmt='grid'))
        
        # Save to CSV
        output_dir = Path(cpu_output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / 'timing_data.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nTiming data saved to {csv_path}")
        
        print(f"\nDetailed analysis and plots saved in {cpu_output_path}/")

def analyze_and_plot_timings(df, output_path):
    """Generate operation breakdown plot from timing data using Plotly"""
    import plotly.graph_objects as go
    from pathlib import Path
    
    output_dir = Path(output_path)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # First, group by level and sum over iterations
    df_level = df.groupby('level').sum().reset_index()

    # Operation time breakdown per level
    operations = [
        ('Pre-smoothing', 'down_pre_smooth'),
        ('Residual', 'down_residual'), 
        ('Restriction', 'down_restrict'),
        ('Prolongation', 'up_prolong'),
        ('Post-smoothing', 'up_post_smooth')
    ]
    
    # Create operation breakdown plot
    fig = go.Figure()
    x = sorted(df_level['level'].unique())
    
    for label, col in operations:
        means = df_level[col]
        fig.add_trace(go.Bar(
            name=label,
            x=x,
            y=means,
        ))
    
    fig.update_layout(
        barmode='group',
        title='Operation Time Breakdown per Level (Summed over Iterations)',
        xaxis_title='Level',
        yaxis_title='Total Time (seconds)',
        # yaxis_type='log',
        showlegend=True
    )
    
    fig.write_html(output_dir / 'operation_breakdown.html')

    # Create total time plot
    fig_total = go.Figure()
    
    # Calculate total time per level
    df_level['total_time'] = df_level['down_total'] + df_level['up_total']
    
    fig_total.add_trace(go.Bar(
        x=df_level['level'],
        y=df_level['total_time'],
        name='Total Time'
    ))
    
    fig_total.update_layout(
        title='Total Solve Time(Up + Down) (Summed over Iterations)',
        xaxis_title='Level',
        yaxis_title='Total Solve Time (seconds)',
        # yaxis_type='log',
        showlegend=True,
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,  # Force integer ticks
            tickformat='d'  # Display as integers
        )
    )
    
    fig_total.write_html(output_dir / 'total_time_per_level.html')

    print(f"\nAnalysis files generated in {output_dir}:")
    print("- Interactive plots (HTML files)")
    
    # Return basic summary statistics
    summary = {
        'per_level': df_level[['down_total', 'up_total', 'total_time']].round(6)
    }
    return summary

# Collect: H2D, D2H, V-cycle down, V-cycle up, Memory used
@dataclass 
class DeviceOperatorTiming:
    """Timing data for AMG device operators"""
    h2d_data_size: int  # H2D data size in bytes
    d2h_data_size: int  # D2H data size in bytes    
    h2d_time: float     # Host to device transfer time in seconds
    d2h_time: float     # Device to host transfer time in seconds
    kernel_vcycle_down_time: float  # V-cycle down time from hardware timer (seconds)
    kernel_coarse_solve_time: float  # Coarse solve time from CPU as of now
    kernel_vcycle_up_time: float    # V-cycle up time from hardware timer (seconds)
    memory_used: int    # Memory used on device (kilobytes)
    # other information
    total_PEs: Tuple[int, int] = (0, 0)
    problem_size: Tuple[int, int] = (0, 0)

    def __init__(self):
        self.h2d_data_size = 0
        self.d2h_data_size = 0
        self.h2d_time = 0.0
        self.d2h_time = 0.0
        self.kernel_vcycle_down_time = 0.0
        self.kernel_coarse_solve_time = 0.0
        self.kernel_vcycle_up_time = 0.0
        self.memory_used = 0
        self.total_PEs = (0, 0)
        self.problem_size = (0, 0)
    
    # Setters
    def set_h2d_data_size(self, size: int):
        self.h2d_data_size += size

    def set_d2h_data_size(self, size: int):
        self.d2h_data_size += size

    def set_h2d_time(self, time: float):
        self.h2d_time += time

    def set_d2h_time(self, time: float):
        self.d2h_time += time

    def set_kernel_vcycle_down_time(self, time: float):
        self.kernel_vcycle_down_time += time
    
    def set_kernel_coarse_solve_time(self, time: float):
        self.kernel_coarse_solve_time += time

    def set_kernel_vcycle_up_time(self, time: float):
        self.kernel_vcycle_up_time += time

    def set_memory_used(self, memory: int):
        self.memory_used += memory

    def set_total_PEs(self, PEs: Tuple[int, int]):
        self.total_PEs = PEs
    
    def set_problem_size(self, size: Tuple[int, int]):
        self.problem_size = size
    
    # For getters directly access the attributes

@dataclass
class DeviceProfiling:
    """Class to track device-side timing information"""
    # Structure: iteration -> timing
    timings: Dict[int, DeviceOperatorTiming] = None
    current_iteration: int = 0

    def __init__(self):
        self.timings = {}
        
    def add_timing_object(self, iteration: int, timing: DeviceOperatorTiming):
        """Add timing data for current iteration"""
        self.current_iteration = iteration
        self.timings[self.current_iteration] = timing

    def print_timing_summary(self):
        """Print a summary of timing information and save to CSV for plotting"""
        from tabulate import tabulate
        import pandas as pd
        from pathlib import Path

        # Convert timing data to list of dictionaries for DataFrame
        timing_records = []
        for iteration in sorted(self.timings.keys()):
            timing = self.timings[iteration]
            record = {
                'iteration': iteration,
                'problem_size': timing.problem_size,
                'total_PEs': timing.total_PEs,
                'memory_used': timing.memory_used,                
                'h2d_time': timing.h2d_time,
                'd2h_time': timing.d2h_time,
                'h2d_data_size': timing.h2d_data_size,
                'd2h_data_size': timing.d2h_data_size,
                'h2d_bandwidth': timing.h2d_data_size / timing.h2d_time,
                'd2h_bandwidth': timing.d2h_data_size / timing.d2h_time,
                'v_cycle_down': timing.kernel_vcycle_down_time,
                'v_cycle_up': timing.kernel_vcycle_up_time,
            }
            timing_records.append(record)

        # Create DataFrame
        df = pd.DataFrame(timing_records)

        # Generate analysis and plots
        device_output_path = "timing_results/device/plots"
        # summary = analyze_and_plot_device_timings(df, output_path=device_output_path)
        
        # Print formatted table
        print("\nAMG Device Timing Summary:")
        headers = ['Iteration', 'I/P Problem Size', 'Total PEs', 'Memory per PE(kB)', 'H2D(s)', 'D2H(s)', 'H2D Data Size(B)', 'D2H Data Size(B)', 'H2D Bandwidth(B/s)', 'D2H Bandwidth(B/s)', 'V-Cycle Down(s)', 'V-Cycle Up(s)']
        if len(df) > 10:
            print(tabulate(df.head(10), headers=headers, floatfmt='.6f', tablefmt='grid'))
            print("...")
        else:
            print(tabulate(df, headers=headers, floatfmt='.6f', tablefmt='grid'))
        
        # Save to CSV
        output_dir = Path(device_output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / 'device_timing_data.csv'
        file_exists = csv_path.exists()
        with open(csv_path, 'a' if file_exists else 'w') as f:
            df.to_csv(f, index=False, header=not file_exists)
        print(f"\nDevice timing data {'appended to' if file_exists else 'saved to'} {csv_path}")
        print(f"Detailed analysis and plots saved in {device_output_path}/")

# def analyze_and_plot_device_timings(df, output_path):
#     """Generate device timing analysis plots using Plotly"""
#     import plotly.graph_objects as go
#     from pathlib import Path
    
#     output_dir = Path(output_path)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Sum over all iterations
#     df_total = df.sum(numeric_only=True)

#     # Create transfer time plot
#     fig_transfer = go.Figure()
#     fig_transfer.add_trace(go.Bar(
#         x=['Total'],
#         y=[df_total['h2d_time'] + df_total['d2h_time']],
#         name='Total Transfer Time'
#     ))
#     fig_transfer.update_layout(
#         title='Total Memcpy Time (Summed over Iterations)',
#         xaxis_title='Total',
#         yaxis_title='Time (seconds)',
#         showlegend=False,
#         xaxis=dict(
#             tickmode='linear',
#             tick0=0,
#             dtick=1,
#             tickformat='d'
#         )
#     )
    
#     fig_transfer.write_html(output_dir / 'device_transfer_times.html')

#     # Create kernel time plot
#     fig_kernel = go.Figure()
#     fig_kernel.add_trace(go.Bar(
#         x=['Total'],
#         y=[df_total['kernel_time_us']],
#         name='Total Kernel Time'
#     ))
#     fig_kernel.update_layout(
#         title='Total Kernel Time (Summed over Iterations)',
#         xaxis_title='Total',
#         yaxis_title='Time (us)',
#         showlegend=False,
#         xaxis=dict(
#             tickmode='linear',
#             tick0=0,
#             dtick=1,
#             tickformat='d'
#         )
#     )
    
#     fig_kernel.write_html(output_dir / 'device_kernel_times.html')

#     return {
#         'total': df_total[['h2d_time', 'd2h_time', 'kernel_time_us']].round(6)
#     }