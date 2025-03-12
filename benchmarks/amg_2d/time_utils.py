import struct
import numpy as np


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
    
    # Optional: Create timing heatmap
    timing_heatmap = time_end - time_start
    return {
        'cycles_total': cycles_total,
        'time_us': time_us,
        'timing_heatmap': timing_heatmap
    }

def visualize_timing_heatmap(timing_data, filename='timing_heatmap.png'):
    """
    Visualize timing data as a heatmap
    Args:
        timing_data: Dictionary containing timing analysis results
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(timing_data['timing_heatmap'], 
                cmap='viridis',
                annot=True, 
                fmt='.0f',
                cbar_kws={'label': 'Cycles'})
    plt.title('PE Execution Time Distribution')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.savefig(filename)
    plt.close()
    
def write_timing_data(timing_map_all_iterations, filename='timing_data.csv'):
    with open(filename, 'w') as f:
        f.write("Iteration,Direction,Level,Cycles,Time (us)\n")
        for iteration, vcycle_data in timing_map_all_iterations.items():
            for direction, level_data in vcycle_data.items():
                for level_index, timing_data in level_data.items():
                    f.write(f"{iteration},{direction},{level_index},{timing_data['cycles_total']},{timing_data['time_us']}\n")
