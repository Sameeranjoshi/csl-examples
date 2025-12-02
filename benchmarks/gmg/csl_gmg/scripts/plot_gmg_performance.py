#!/usr/bin/env python3
"""
Plot performance data from GMG timing experiments.
Extracts Communication Time, Compute Time, and V-cycle times from output files.
"""

import re
import sys
from typing import List, Dict

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/numpy not found. Install with: pip install matplotlib numpy")


def parse_configuration_summary(text: str) -> List[Dict]:
    """
    Parse Configuration Summary sections to extract:
    - Grid size (PE tiles)
    - Device iterations
    - Device final |rho|_inf
    - Converged status
    """
    results = []
    
    # Split by device runs - each run ends with Configuration Summary
    # Look for pattern: "Device calculations..." -> ... -> "Configuration Summary"
    device_runs = re.split(r'Device calculations\.\.\.', text)
    
    for run_text in device_runs[1:]:  # Skip first split (before first device run)
        # Find Configuration Summary in this run
        config_match = re.search(
            r'Configuration Summary\s*============================================================\s*(.*?)(?=\n\n|\nProcessing on host|\Z)',
            run_text,
            re.DOTALL
        )
        
        if not config_match:
            continue
            
        section = config_match.group(1)
        data = {}
        
        # Extract grid size
        grid_match = re.search(r'HeightxWidthxZDim\s*:\s*(\d+)x(\d+)x(\d+)', section)
        if grid_match:
            width, height, zdim = map(int, grid_match.groups())
            data['grid_size'] = f"{width}x{height}x{zdim}"
            data['total_grid_size'] = width * height * zdim
            data['pe_tiles'] = f"{width}x{height}x{zdim}"
        
        # Extract device iterations
        iter_match = re.search(r'Device iterations\s*:\s*(\d+)', section)
        if iter_match:
            data['device_iterations'] = int(iter_match.group(1))
        
        # Extract device final |rho|_inf
        rho_match = re.search(r'Device final \|rho\|_inf\s*:\s*([\d.eE+-]+)', section)
        if rho_match:
            data['device_rho_inf'] = float(rho_match.group(1))
        
        # Find converged status in this device run (before Configuration Summary)
        # Pattern: "Converged: Yes" or "Converged: No" appears after rho values
        converged_match = re.search(r'Converged:\s*(Yes|No)', run_text, re.IGNORECASE)
        if converged_match:
            data['converged'] = converged_match.group(1).lower() == 'yes'
        else:
            data['converged'] = None
        
        if 'grid_size' in data:
            results.append(data)
    
    return results


def parse_spmv_totals(text: str) -> List[Dict]:
    """
    Parse SPMV Compute vs Communication Time tables from the output file.
    Returns list of dictionaries with grid_size, comm_time, compute_time.
    """
    results = []
    
    # Pattern to find grid size from Configuration Summary
    grid_pattern = re.compile(r'HeightxWidthxZDim\s*:\s*(\d+)x(\d+)x(\d+)')
    
    # Find all SPMV table sections
    sections = text.split('SPMV Compute vs Communication Time per Level')
    
    for section in sections[1:]:  # Skip first empty split
        # Extract total row
        total_match = re.search(
            r'\|\s*total\s*\|\s*([\d.]+)us.*?\|\s*([\d.]+)us.*?\|\s*([\d.]+)us',
            section,
            re.DOTALL
        )
        
        if total_match:
            comm_time = float(total_match.group(2))
            compute_time = float(total_match.group(3))
            
            # Find grid size in this section
            grid_match = grid_pattern.search(section)
            if grid_match:
                width, height, zdim = map(int, grid_match.groups())
                grid_size = f"{width}x{height}x{zdim}"
                
                results.append({
                    'grid_size': grid_size,
                    'comm_time_us': comm_time,
                    'compute_time_us': compute_time,
                })
    
    return results


def parse_vcycle_times(text: str) -> List[Dict]:
    """
    Parse V-cycle time from output file.
    Extracts max of the two V-cycle time values.
    Returns list of dictionaries with grid_size and vcycle_time.
    """
    results = []
    
    # Pattern to find V-cycle time section
    vcycle_pattern = re.compile(
        r'Total V-cycle time \(sum of operations\):\s*([\d.]+)\s*us.*?\n'
        r'Total V-cycle time \(\(No kernel launch\)V-cycle time\):\s*([\d.]+)\s*us',
        re.DOTALL
    )
    
    # Pattern to find grid size
    grid_pattern = re.compile(r'HeightxWidthxZDim\s*:\s*(\d+)x(\d+)x(\d+)')
    
    # Find all V-cycle time sections
    for match in vcycle_pattern.finditer(text):
        end = match.end()
        # Look ahead to find the corresponding grid size (it appears after V-cycle time)
        next_text = text[end:min(len(text), end+2000)]
        grid_match = grid_pattern.search(next_text)
        if grid_match:
            width, height, zdim = map(int, grid_match.groups())
            grid_size = f"{width}x{height}x{zdim}"
            
            time1 = float(match.group(1))
            time2 = float(match.group(2))
            vcycle_time = max(time1, time2)  # Take maximum as instructed
            
            results.append({
                'grid_size': grid_size,
                'vcycle_time_us': vcycle_time
            })
    
    return results


def parse_all_data(text: str) -> List[Dict]:
    """
    Parse all data from the output file and combine results.
    Returns sorted list of combined data dictionaries.
    """
    # Step 1: Parse all data sources
    spmv_data = parse_spmv_totals(text)
    vcycle_data = parse_vcycle_times(text)
    config_data = parse_configuration_summary(text)
    
    # Step 2: Combine data by grid size
    combined = {}
    
    # Add SPMV data
    for item in spmv_data:
        grid = item['grid_size']
        combined[grid] = {
            'pe_tiles': grid,
            'grid_size': grid,
            'comm_time_us': item['comm_time_us'],
            'compute_time_us': item['compute_time_us'],
        }
    
    # Add V-cycle data
    for item in vcycle_data:
        grid = item['grid_size']
        if grid in combined:
            combined[grid]['vcycle_time_us'] = item['vcycle_time_us']
        else:
            combined[grid] = {
                'pe_tiles': grid,
                'grid_size': grid,
                'vcycle_time_us': item['vcycle_time_us']
            }
    
    # Add configuration data
    for item in config_data:
        grid = item['grid_size']
        if grid in combined:
            combined[grid].update({
                'total_grid_size': item.get('total_grid_size', 0),
                'device_iterations': item.get('device_iterations', None),
                'device_rho_inf': item.get('device_rho_inf', None),
                'converged': item.get('converged', None)
            })
        else:
            combined[grid] = {
                'pe_tiles': item.get('pe_tiles', grid),
                'grid_size': grid,
                'total_grid_size': item.get('total_grid_size', 0),
                'device_iterations': item.get('device_iterations', None),
                'device_rho_inf': item.get('device_rho_inf', None),
                'converged': item.get('converged', None)
            }
    
    # Step 3: Convert to sorted list (by total_grid_size)
    sorted_data = sorted(combined.values(), key=lambda x: x.get('total_grid_size', 0))
    
    return sorted_data


def plot_comm_vs_compute(data: List[Dict], output_file: str = 'comm_vs_compute_time.png'):
    """
    Create plot showing Communication/Compute ratio to demonstrate relative scaling.
    Shows how communication overhead scales relative to compute time.
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib not available. Cannot create plots.")
        return None
    
    grid_sizes = [d['grid_size'] for d in data]
    comm_times = [d['comm_time_us'] for d in data]  # Keep in microseconds
    compute_times = [d['compute_time_us'] for d in data]  # Keep in microseconds
    
    # Extract grid dimension (assuming cubic grids like 4x4x4 -> 4)
    grid_dims = []
    for grid in grid_sizes:
        dim = int(grid.split('x')[0])
        grid_dims.append(dim)
    
    # Calculate ratios (comm/compute)
    ratios = []
    for comm, comp in zip(comm_times, compute_times):
        if comp > 0:
            ratios.append(comm / comp)
        else:
            ratios.append(0)
    
    x_labels = [f"{dim}³" for dim in grid_dims]
    x_pos = np.arange(len(grid_sizes))
    
    # Create figure with two subplots: ratio plot and normalized stacked bars
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top plot: Communication/Compute Ratio
    bars = ax1.bar(x_pos, ratios, color='#D62728', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Grid Size (Subdomain Dimension)', fontsize=12)
    ax1.set_ylabel('Communication / Compute Ratio', fontsize=12)
    ax1.set_title('Communication Overhead Relative to Compute (Comm/Compute Ratio)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='1:1 ratio')
    
    # Add value labels on top of bars
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.legend(fontsize=10)
    
    # Bottom plot: Normalized stacked bars (each bar normalized to 100%)
    # This shows the relative proportion without absolute time differences
    comm_percentages = []
    comp_percentages = []
    for comm, comp in zip(comm_times, compute_times):
        total = comm + comp
        if total > 0:
            comm_percentages.append((comm / total) * 100)
            comp_percentages.append((comp / total) * 100)
        else:
            comm_percentages.append(0)
            comp_percentages.append(0)
    
    bars1 = ax2.bar(x_pos, comm_percentages, label='Communication (%)', color='#FF7F0E', alpha=0.8)
    bars2 = ax2.bar(x_pos, comp_percentages, bottom=comm_percentages, label='Compute (%)', 
                    color='#1F77B4', alpha=0.8)
    
    # Add percentage labels
    for i, (comm_pct, comp_pct) in enumerate(zip(comm_percentages, comp_percentages)):
        # Label communication percentage in the middle of its segment
        if comm_pct > 5:  # Only label if segment is large enough
            ax2.text(i, comm_pct / 2, f'{comm_pct:.0f}%', ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
        # Label compute percentage at the top of the bar (100% line) - like the sample image
        if comp_pct > 0:
            ax2.text(i, 100, f'{int(comp_pct)}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='black')
    
    ax2.set_xlabel('Grid Size (Subdomain Dimension)', fontsize=12)
    ax2.set_ylabel('Relative Proportion (%)', fontsize=12)
    ax2.set_title('Normalized Time Distribution (Each Bar = 100%)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax2.set_ylim(0, 110)  # Add extra space at top for labels
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=11)  # Move legend to avoid overlap
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    return fig


def plot_vcycle_time(data: List[Dict], output_file: str = 'vcycle_time.png'):
    """
    Create log-log plot showing V-cycle time vs grid size.
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib not available. Cannot create plots.")
        return None
    
    grid_sizes = [d['grid_size'] for d in data]
    vcycle_times = [d.get('vcycle_time_us', 0) for d in data]  # Keep in microseconds
    
    # Extract grid dimensions
    grid_dims = []
    for grid in grid_sizes:
        dim = int(grid.split('x')[0])
        grid_dims.append(dim)
    
    # Filter out entries without vcycle_time
    filtered = [(dim, time, size) for dim, time, size in zip(grid_dims, vcycle_times, grid_sizes) if time > 0]
    if not filtered:
        print("Warning: No V-cycle time data found")
        return None
    
    dims, times, sizes = zip(*filtered)
    
    # Create figure with single log-log plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Log-log scale plot
    ax.loglog(dims, times, 'o-', linewidth=2, markersize=8, color='#D62728')
    ax.set_xlabel('Grid Size (Subdomain Dimension)', fontsize=12)
    ax.set_ylabel('V-cycle Time (microseconds)', fontsize=12)
    ax.set_title('V-cycle Time vs Grid Size(log-log scale)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.set_xticks(dims)
    ax.set_xticklabels([f"{d}³" for d in dims])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    return fig


def print_summary_table(data: List[Dict]):
    """
    Print a summary table of the parsed data with updated column names and units.
    """
    print("\n" + "="*120)
    print("Performance Data Summary")
    print("="*120)
    header = (f"{'PE tiles':<15} {'Total Grid size':<18} {'Comm Time(spmv) (us)':<25} "
              f"{'Compute Time(spmv) (us)':<28} {'V-cycle Time (us)':<20} "
              f"{'Iterations':<12} {'|rho|_inf':<15} {'Converged':<10}")
    print(header)
    print("-"*120)
    
    for d in data:
        pe_tiles = d.get('pe_tiles', 'N/A')
        total_grid = d.get('total_grid_size', 0)
        comm = d.get('comm_time_us', 0)
        comp = d.get('compute_time_us', 0)
        vcycle = d.get('vcycle_time_us', 0)
        iterations = d.get('device_iterations', None)
        rho_inf = d.get('device_rho_inf', None)
        converged = d.get('converged', None)
        
        iter_str = str(iterations) if iterations is not None else 'N/A'
        rho_str = f"{rho_inf:.3e}" if rho_inf is not None else 'N/A'
        conv_str = 'Yes' if converged else 'No' if converged is False else 'N/A'
        
        print(f"{pe_tiles:<15} {total_grid:<18,} {comm:<25.2f} {comp:<28.2f} "
              f"{vcycle:<20.2f} {iter_str:<12} {rho_str:<15} {conv_str:<10}")
    
    print("="*120 + "\n")


def main():
    """
    Main function: Read file, parse data, print summary, generate plots.
    """
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <gmg_output_file.txt>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Step 1: Read file
    print(f"Reading: {input_file}")
    with open(input_file, 'r') as f:
        text = f.read()
    
    # Step 2: Parse all data
    print("Parsing data...")
    data = parse_all_data(text)
    
    if not data:
        print("Error: No data found in file")
        sys.exit(1)
    
    # Step 3: Print summary table
    print_summary_table(data)
    
    # Step 4: Generate plots if matplotlib is available
    if HAS_MATPLOTLIB:
        print("Generating plots...")
        plot_comm_vs_compute(data, 'comm_vs_compute_time.png')
        plot_vcycle_time(data, 'vcycle_time.png')
        print("\n✓ All plots generated successfully!")
    else:
        print("\n⚠ Plots not generated. Install matplotlib and numpy to create visualizations:")
        print("   pip install matplotlib numpy")


if __name__ == "__main__":
    main()
