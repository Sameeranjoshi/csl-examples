# python optimized_vs_unoptimized.py out_3_3_6.txt out_3_3_6_unoptimized.txt
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def parse_summary_table(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Restrict to Performance Data Summary section (from plot_gmg_performance output)
    if 'Performance Data Summary' in content:
        section = content.split('Performance Data Summary', 1)[-1]
        if 'Per-Operation Timing' in section:
            section = section.split('Per-Operation Timing', 1)[0]
        content = section

    # Regex to capture the summary table rows accurately
    pattern = re.compile(
        r"(\d+x\d+x\d+)\s+[\d,]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+([\d.]+)\s+[\d.e+-]+\s+\w+\s+([\d.]+|N/A)\s+([\d.]+)"
    )
    
    matches = pattern.findall(content)
    data = []
    for m in matches:
        data.append({
            'Size': m[0],
            'CommTime': float(m[1]),
            'ComputeTime': float(m[2]),
            'VcycleTotal': float(m[3]),
            'AvgVcycle': float(m[4]),
            'CompileTime': 0.0 if m[5] == 'N/A' else float(m[5])
        })
    
    return pd.DataFrame(data)

def generate_plots(opt_file, unopt_file):
    df_opt = parse_summary_table(opt_file)
    df_unopt = parse_summary_table(unopt_file)

    if df_opt.empty or 'Size' not in df_opt.columns:
        raise ValueError(
            "Could not parse Performance Data Summary from input files. "
            "Ensure inputs are plot_gmg_performance output (e.g. out_3_3_6.txt) "
            "containing the 'Performance Data Summary' table."
        )
    if df_unopt.empty or 'Size' not in df_unopt.columns:
        raise ValueError("Could not parse unoptimized file. Check it contains the Performance Data Summary table.")

    # Filter to 16³ to 512³ problem sizes only
    def keep_size(s):
        dim = int(s.split('x')[0])
        return 16 <= dim <= 512
    mask = df_opt['Size'].apply(keep_size)
    df_opt = df_opt[mask].reset_index(drop=True)
    df_unopt = df_unopt[mask].reset_index(drop=True)
    
    # Calculate speedup for the speedup scaling plot
    speedup = df_unopt['AvgVcycle'] / df_opt['AvgVcycle']
    
    # --- metrics_comparison.png ---
    metrics = [
        ('CommTime', 'Comm Time (spmv) [us]', 'linear', 'bar'),
        ('ComputeTime', 'Compute Time (spmv) [us]', 'linear', 'bar'),
        ('Speedup', 'V-Cycle Speedup Scaling (Unoptimized / Optimized)', 'linear', 'speedup'),
        ('CompileTime', 'Compile Time [s]', 'linear', 'bar')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    x = np.arange(len(df_opt['Size']))
    width = 0.35

    for i, (col, title, scale, plot_type) in enumerate(metrics):
        if plot_type == 'speedup':
            # Speedup scaling plot - use numeric x positions like bar charts
            # Use same color scheme as bar charts (light blue for optimized performance)
            axes[i].plot(x, speedup, marker='o', linewidth=2.5, color='#66b3ff', label='Speedup Factor', alpha=0.8)
            axes[i].axhline(y=1, color='gray', linestyle='--', alpha=0.6, label='Baseline (1.0x)')
            if col == "CompileTime":
                axes[i].legend(loc='top center', bbox_to_anchor=(0, 1))
            else:
                axes[i].legend(loc='upper left', bbox_to_anchor=(0, 1))
            
            
            # Add speedup factor labels on each point
            for j, (size, sp) in enumerate(zip(df_opt['Size'], speedup)):
                axes[i].annotate(f'{sp:.2f}x', (x[j], sp), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=10, color='black')
            
            # Set x-axis ticks and labels to match other plots
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(df_opt['Size'], rotation=30)
            
            # Adjust y-axis limits to accommodate labels
            y_min, y_max = axes[i].get_ylim()
            axes[i].set_ylim(y_min, y_max * 1.12)
            # axes[i].set_xlabel('Grid Size ($N^3$)', fontsize=11)
            axes[i].set_ylabel('Speedup Factor', fontsize=11)
            axes[i].grid(True, linestyle=':', alpha=0.7)
        else:
            # Use bar chart for other metrics
            bars_unopt = axes[i].bar(x - width/2, df_unopt[col], width, label='Unoptimized', color='#ff9999', alpha=0.8)
            bars_opt = axes[i].bar(x + width/2, df_opt[col], width, label='Optimized', color='#66b3ff', alpha=0.8)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(df_opt['Size'], rotation=30)
            
            # Add speedup factor labels for CommTime and ComputeTime
            if col in ['CommTime', 'ComputeTime']:
                speedup_factors = df_unopt[col] / df_opt[col]
                for j, (bar_unopt, bar_opt, sp_factor) in enumerate(zip(bars_unopt, bars_opt, speedup_factors)):
                    # Find the maximum height of the two bars to position label above
                    max_height = max(bar_unopt.get_height(), bar_opt.get_height())
                    axes[i].annotate(f'{sp_factor:.2f}x', 
                                    xy=(x[j], max_height), 
                                    xytext=(0, 3), 
                                    textcoords="offset points",
                                    ha='center', va='bottom', 
                                    fontsize=10, color='black')
                # Adjust y-axis limits to accommodate labels
                y_min, y_max = axes[i].get_ylim()
                axes[i].set_ylim(y_min, y_max * 1.08)
        
        axes[i].set_title(title, fontsize=13, fontweight='bold')
        if plot_type != 'speedup':
            axes[i].set_yscale(scale)
        axes[i].legend(loc="upper left")
        if plot_type != 'speedup':
            axes[i].grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle('Performance Metric Comparison: Optimized vs. Unoptimized', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('metrics_comparison.png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    print("Generated: metrics_comparison.png")


if __name__ == "__main__":
    # Ensure these names match your exact file names
    import sys

    if len(sys.argv) != 3:
        print("Usage: python optimized_vs_unoptimized.py <optimized_results_file> <unoptimized_results_file>")
        sys.exit(1)

    opt_file = sys.argv[1]
    unopt_file = sys.argv[2]
    generate_plots(opt_file, unopt_file)