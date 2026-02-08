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

    # Regex: full (with Code/Data) and short (without)
    pattern_full = re.compile(
        r"(\d+x\d+x\d+)\s+[\d,]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+([\d.]+)\s+[\d.e+-]+\s+\w+\s+([\d.]+|N/A)\s+([\d.]+)\s+([\d,]+|N/A)\s+([\d,]+|N/A)"
    )
    pattern_short = re.compile(
        r"(\d+x\d+x\d+)\s+[\d,]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+([\d.]+)\s+[\d.e+-]+\s+\w+\s+([\d.]+|N/A)\s+([\d.]+)"
    )

    def parse_bytes(s):
        s = str(s).strip()
        if not s or s == 'N/A':
            return None
        return int(s.replace(',', ''))

    data = []
    for line in content.splitlines():
        m = pattern_full.search(line)
        if m:
            data.append({
                'Size': m.group(1),
                'CommTime': float(m.group(2)),
                'ComputeTime': float(m.group(3)),
                'VcycleTotal': float(m.group(4)),
                'AvgVcycle': float(m.group(5)),
                'CompileTime': 0.0 if m.group(6) == 'N/A' else float(m.group(6)),
                'Code_bytes': parse_bytes(m.group(8)),
                'Data_bytes': parse_bytes(m.group(9)),
            })
        else:
            m = pattern_short.search(line)
            if m:
                data.append({
                    'Size': m.group(1),
                    'CommTime': float(m.group(2)),
                    'ComputeTime': float(m.group(3)),
                    'VcycleTotal': float(m.group(4)),
                    'AvgVcycle': float(m.group(5)),
                    'CompileTime': 0.0 if m.group(6) == 'N/A' else float(m.group(6)),
                    'Code_bytes': None,
                    'Data_bytes': None,
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
    
    # Code/Data stacked bar from opt_file only (rows with valid Code and Data)
    REF_48KB = 49152
    mask_codedata = df_opt['Code_bytes'].notna() & df_opt['Data_bytes'].notna()
    df_codedata = df_opt[mask_codedata].reset_index(drop=True)

    # --- metrics_comparison.png --- (paper-sized: larger figure, fonts, 300 DPI)
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })
    metrics = [
        ('CommTime', 'Comm Time (spmv) [µs]', 'linear', 'bar'),
        ('ComputeTime', 'Compute Time (spmv) [µs]', 'linear', 'bar'),
        ('Speedup', 'V-Cycle Speedup', 'linear', 'speedup'),
        ('CodeData', 'Code+Data vs 48KB', 'linear', 'stacked')
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    axes = np.array(axes).flatten()  # ensure 1D array of 4 axes for side-by-side
    x = np.arange(len(df_opt['Size']))
    width = 0.35

    for i, (col, title, scale, plot_type) in enumerate(metrics):
        if plot_type == 'speedup':
            # Speedup scaling plot
            axes[i].plot(x, speedup, marker='o', linewidth=2.5, color='#66b3ff', label='Speedup Factor', alpha=0.8)
            axes[i].axhline(y=1, color='gray', linestyle='--', alpha=0.6, label='Baseline (1.0x)')
            axes[i].legend(loc='upper left', bbox_to_anchor=(0, 1))
            for j, (size, sp) in enumerate(zip(df_opt['Size'], speedup)):
                axes[i].annotate(f'{sp:.2f}x', (x[j], sp), textcoords="offset points",
                                xytext=(0, 10), ha='center', fontsize=13, color='black')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(df_opt['Size'], rotation=30)
            y_min, y_max = axes[i].get_ylim()
            axes[i].set_ylim(y_min, y_max * 1.12)
            axes[i].set_ylabel('Speedup Factor', fontsize=15)
            axes[i].grid(True, linestyle=':', alpha=0.7)
        elif plot_type == 'stacked':
            # Code + Data stacked bar from opt_file, 48KB line, % inside bars
            if df_codedata.empty:
                axes[i].text(0.5, 0.5, 'No Code/Data', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_xticks([])
            else:
                x_cd = np.arange(len(df_codedata['Size']))
                code = df_codedata['Code_bytes'].values
                data = df_codedata['Data_bytes'].values
                axes[i].bar(x_cd, code, width=0.6, label='Code Section', color='#ff9999', alpha=0.9, bottom=0)
                axes[i].bar(x_cd, data, width=0.6, label='Data Section', color='#66b3ff', alpha=0.9, bottom=code)
                axes[i].axhline(y=REF_48KB, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.8, label='48 KB')
                # % of 48KB inside each segment
                for j in range(len(df_codedata)):
                    code_pct = (code[j] / REF_48KB) * 100
                    data_pct = (data[j] / REF_48KB) * 100
                    if code[j] > 800:
                        axes[i].text(x_cd[j], code[j] / 2, f'{code_pct:.0f}%', ha='center', va='center', fontsize=12, color='#333')
                    if data[j] > 400:
                        axes[i].text(x_cd[j], code[j] + data[j] / 2, f'{data_pct:.0f}%', ha='center', va='center', fontsize=12, color='#333')
                axes[i].set_xticks(x_cd)
                axes[i].set_xticklabels(df_codedata['Size'], rotation=30)
                axes[i].set_ylabel('Bytes', fontsize=15)
                axes[i].set_ylim(0, max(REF_48KB * 1.1, (code + data).max() * 1.1))
                axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        else:
            # Bar chart for CommTime, ComputeTime
            bars_unopt = axes[i].bar(x - width/2, df_unopt[col], width, label='Unoptimized', color='#ff9999', alpha=0.8)
            bars_opt = axes[i].bar(x + width/2, df_opt[col], width, label='Optimized', color='#66b3ff', alpha=0.8)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(df_opt['Size'], rotation=30)
            if col in ['CommTime', 'ComputeTime']:
                speedup_factors = df_unopt[col] / df_opt[col]
                for j, (bar_unopt, bar_opt, sp_factor) in enumerate(zip(bars_unopt, bars_opt, speedup_factors)):
                    max_height = max(bar_unopt.get_height(), bar_opt.get_height())
                    axes[i].annotate(f'{sp_factor:.2f}x', xy=(x[j], max_height), xytext=(0, 3),
                                    textcoords="offset points", ha='center', va='bottom', fontsize=12, color='black')
                y_min, y_max = axes[i].get_ylim()
                axes[i].set_ylim(y_min, y_max * 1.08)

        axes[i].set_title(title, fontsize=17, fontweight='bold')
        if plot_type not in ('speedup', 'stacked'):
            axes[i].set_yscale(scale)
        axes[i].legend(loc="upper left")
        if plot_type not in ('speedup', 'stacked'):
            axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        elif plot_type == 'stacked' and not df_codedata.empty:
            axes[i].legend(loc="upper left")

    plt.suptitle('Performance Metric Comparison: Optimized vs. Unoptimized', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig('metrics_comparison.png', bbox_inches='tight', dpi=300)
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