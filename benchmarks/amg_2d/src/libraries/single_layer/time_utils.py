import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)

def time_analysis_noref(height, width, time_memcpy_hwl):
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

    # # time_ref = reference clock
    # time_ref = np.zeros((height, width)).astype(int)
    # word = np.zeros(3).astype(np.uint16)
    # for w in range(width):
    #     for h in range(height):
    #         hex_t0 = int(float_to_hex(time_ref_hwl[(h, w, 0)]), base=16)
    #         hex_t1 = int(float_to_hex(time_ref_hwl[(h, w, 1)]), base=16)
    #         word[0] = hex_t0 & 0x0000FFFF
    #         word[1] = (hex_t0 >> 16) & 0x0000FFFF
    #         word[2] = hex_t1 & 0x0000FFFF
    #         time_ref[(h, w)] = make_u48(word)
    # # adjust the reference clock by the propagation delay
    # for py in range(height):
    #     for px in range(width):
    #         time_ref[(py, px)] = time_ref[(py, px)] - (px + py)

    # # shift time_start and time_end by time_ref
    # time_start = time_start - time_ref
    # time_end = time_end - time_ref


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
        'cycles': np.mean(cycles_send),
        'time_us': np.mean(time_send),
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
        # 'time_start': time_start,
        # 'time_end': time_end,
        # 'time_ref': time_ref
    }
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

## Experimental all the following functions
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
    
def plot_timing_heatmap(df, output_file="timing_heatmap.html"):
    """Create an interactive heatmap of execution times across iterations and levels."""
    pivot_df = df.pivot_table(
        values='Time (us)', 
        index=['Iteration', 'Direction'], 
        columns='Level', 
        aggfunc='sum'
    )
    
    fig = px.imshow(pivot_df,
                    labels=dict(x="Level", y="Iteration-Direction", color="Time (μs)"),
                    title="Execution Time Heatmap",
                    aspect="auto")
    
    fig.write_html(output_file)
    print(f"Heatmap saved to {output_file}")

def plot_timing_breakdown(df, output_file="timing_breakdown.html"):
    """Create an interactive stacked bar chart showing timing breakdown per iteration."""
    fig = px.bar(df, 
                 x='Iteration', 
                 y='Time (us)',
                 color='Direction',
                 barmode='stack',
                 title='Timing Breakdown per Iteration',
                 labels={'Time (us)': 'Time (μs)'})
    
    fig.write_html(output_file)
    print(f"Timing breakdown saved to {output_file}")

def plot_convergence_analysis(df, output_file="convergence_analysis.html"):
    """Create an interactive line plot showing timing patterns across iterations."""
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Total Time per Iteration', 
                                      'Time Distribution by Level'))
    
    # Total time per iteration
    total_time = df.groupby('Iteration')['Time (us)'].sum()
    fig.add_trace(
        go.Scatter(x=total_time.index, y=total_time.values,
                  mode='lines+markers',
                  name='Total Time'),
        row=1, col=1
    )
    
    # Time distribution by level
    for level in df['Level'].unique():
        level_data = df[df['Level'] == level].groupby('Iteration')['Time (us)'].sum()
        fig.add_trace(
            go.Scatter(x=level_data.index, y=level_data.values,
                      mode='lines+markers',
                      name=f'Level {level}'),
            row=2, col=1
        )
    
    fig.update_layout(height=800, title_text="Convergence Analysis")
    fig.write_html(output_file)
    print(f"Convergence analysis saved to {output_file}")

def generate_timing_report(df, output_file="timing_report.html"):
    """Generate a comprehensive HTML report with all visualizations and statistics."""
    
    # Calculate summary statistics
    total_time = df['Time (us)'].sum()
    avg_iteration_time = df.groupby('Iteration')['Time (us)'].sum().mean()
    time_by_direction = df.groupby('Direction')['Time (us)'].sum()
    time_by_level = df.groupby('Level')['Time (us)'].sum()
    
    # Create HTML report
    html_content = f"""
    <html>
    <head>
        <title>AMG Timing Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
            .stat {{ margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>AMG Timing Analysis Report</h1>
        
        <div class="section">
            <h2>Summary Statistics</h2>
            <div class="stat">Total Execution Time: {total_time:.2f} μs ({total_time/1e6:.2f} s)</div>
            <div class="stat">Average Iteration Time: {avg_iteration_time:.2f} μs</div>
        </div>
        
        <div class="section">
            <h2>Time Distribution by Direction</h2>
            <table>
                <tr><th>Direction</th><th>Time (μs)</th><th>Percentage</th></tr>
                {''.join(f"<tr><td>{dir}</td><td>{time:.2f}</td><td>{(time/total_time)*100:.1f}%</td></tr>" 
                        for dir, time in time_by_direction.items())}
            </table>
        </div>
        
        <div class="section">
            <h2>Time Distribution by Level</h2>
            <table>
                <tr><th>Level</th><th>Time (μs)</th><th>Percentage</th></tr>
                {''.join(f"<tr><td>{level}</td><td>{time:.2f}</td><td>{(time/total_time)*100:.1f}%</td></tr>"
                        for level, time in time_by_level.items())}
            </table>
        </div>
        
        <div class="section">
            <h2>Visualizations</h2>
            <iframe src="timing_heatmap.html" width="100%" height="600px"></iframe>
            <iframe src="timing_breakdown.html" width="100%" height="600px"></iframe>
            <iframe src="convergence_analysis.html" width="100%" height="800px"></iframe>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    print(f"Comprehensive report saved to {output_file}")

def analyze_timing_data(csv_file="timing_data.csv"):
    """Main function to generate all visualizations and reports."""
    df = pd.read_csv(csv_file)
    
    # Create visualizations
    plot_timing_heatmap(df)
    plot_timing_breakdown(df)
    plot_convergence_analysis(df)
    
    # Generate comprehensive report
    generate_timing_report(df)
    
    return df
