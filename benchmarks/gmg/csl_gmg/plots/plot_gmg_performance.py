#!/usr/bin/env python3
"""
Plot performance data from GMG timing experiments.
Extracts Communication Time, Compute Time, and V-cycle times from output files.
ls -d out_dir_S*x* | sed 's/.*S\([0-9]*\)x.*/\1 &/' | sort -n | cut -d' ' -f2- | xargs -I{} cat {}/response.txt > all_responses_6_6_6.txt
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
    Parse Configuration Summary sections to extract all configuration parameters:
    - Grid size (PE tiles)
    - Levels, Max iterations, Tolerances
    - Pre/Post/Bottom iterations
    - Datatype, Jacobi omega, Stencil parameters
    - Channels, Block size, Buffer widths
    - Device iterations, Device final |rho|_inf
    - Converged status, Compile time, Run time
    """
    results = []
    
    # Split by device runs - each run ends with Configuration Summary
    # Look for pattern: "Device calculations..." -> ... -> "Configuration Summary"
    device_runs = re.split(r'Device calculations\.\.\.', text)
    
    for run_idx, run_text in enumerate(device_runs[1:], 1):  # Skip first split (before first device run)
        # Find Configuration Summary in this run
        config_match = re.search(
            r'Configuration Summary\s*============================================================\s*(.*?)(?=\n\n|\nProcessing on host|\nCompile command|\Z)',
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
        
        # Extract all configuration parameters
        levels_match = re.search(r'Levels\s*:\s*(\d+)', section)
        if levels_match:
            data['levels'] = int(levels_match.group(1))
        
        max_iter_match = re.search(r'Max iterations\s*:\s*(\d+)', section)
        if max_iter_match:
            data['max_iterations'] = int(max_iter_match.group(1))
        
        tol_abs_match = re.search(r'Tolerance \(abs\)\s*:\s*([\d.eE+-]+)', section)
        if tol_abs_match:
            data['tolerance_abs'] = tol_abs_match.group(1)
        
        tol_rel_match = re.search(r'Tolerance \(rel\)\s*:\s*([\d.eE+-]+)', section)
        if tol_rel_match:
            data['tolerance_rel'] = tol_rel_match.group(1)
        
        pre_post_bottom_match = re.search(r'Pre/Post/Bottom iter\s*:\s*(\d+)/(\d+)/(\d+)', section)
        if pre_post_bottom_match:
            data['pre_iter'] = int(pre_post_bottom_match.group(1))
            data['post_iter'] = int(pre_post_bottom_match.group(2))
            data['bottom_iter'] = int(pre_post_bottom_match.group(3))
            data['pre_post_bottom_iter'] = f"{data['pre_iter']}/{data['post_iter']}/{data['bottom_iter']}"
        
        datatype_match = re.search(r'Datatype\s*:\s*(\w+)', section)
        if datatype_match:
            data['datatype'] = datatype_match.group(1)
        
        jacobi_omega_match = re.search(r'Jacobi omega\s*:\s*([\d.]+)', section)
        if jacobi_omega_match:
            data['jacobi_omega'] = float(jacobi_omega_match.group(1))
        
        stencil_match = re.search(r'Stencil alpha/beta\s*:\s*([\d.-]+)/([\d.]+)', section)
        if stencil_match:
            data['stencil_alpha'] = stencil_match.group(1)
            data['stencil_beta'] = stencil_match.group(2)
            data['stencil_alpha_beta'] = f"{data['stencil_alpha']}/{data['stencil_beta']}"
        
        block_size_match = re.search(r'Block size\s*:\s*(\d+)', section)
        if block_size_match:
            data['block_size'] = int(block_size_match.group(1))
        
        # Extract device iterations
        iter_match = re.search(r'Device iterations\s*:\s*(\d+)', section)
        if iter_match:
            data['device_iterations'] = int(iter_match.group(1))
        
        # Extract 1-V cycle time(Average)
        vcycle_avg_match = re.search(r'1-V cycle time\(Average\)\s*\(us\[cycles\]\):\s*([\d.]+)\s*us', section)
        if vcycle_avg_match:
            data['vcycle_avg_time_us'] = float(vcycle_avg_match.group(1))
        else:
            data['vcycle_avg_time_us'] = None
        
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
        
        # Extract compile time and run time (appear after Configuration Summary)
        compile_match = re.search(r'Compile time \(s\):\s*([\d.]+)', run_text, re.IGNORECASE)
        if not compile_match:
            compile_match = re.search(r'COMPILE TIME:\s*([\d.]+)\s*seconds', run_text, re.IGNORECASE)
        if compile_match:
            data['compile_time_s'] = float(compile_match.group(1))
        else:
            data['compile_time_s'] = None
        
        run_match = re.search(r'Run time \(s\):\s*([\d.]+)', run_text, re.IGNORECASE)
        if not run_match:
            run_match = re.search(r'RUN TIME:\s*([\d.]+)\s*seconds', run_text, re.IGNORECASE)
        if run_match:
            data['run_time_s'] = float(run_match.group(1))
        else:
            data['run_time_s'] = None
        
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
    sections = text.split('7-pt Stencil Compute vs Communication Time per Level')
    
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
    Extracts the "Total V-cycle time (Kernel Launch + V-cycle time)" value.
    Returns list of dictionaries with grid_size and vcycle_time.
    """
    results = []
    
    # Pattern to find V-cycle time section - matches actual log format:
    # "Total Upper bound V-cycle time (sum of operations):   1071.178 us (    937281 cycles)"
    # "Total V-cycle time (Kernel Launch + V-cycle time):    899.928 us (    787437 cycles)"
    vcycle_pattern = re.compile(
        r'Total Upper bound V-cycle time \(sum of operations\):\s*([\d.]+)\s*us.*?\n'
        r'Total V-cycle time \(Kernel Launch \+ V-cycle time\):\s*([\d.]+)\s*us',
        re.DOTALL
    )
    
    # Pattern to find grid size
    grid_pattern = re.compile(r'HeightxWidthxZDim\s*:\s*(\d+)x(\d+)x(\d+)')
    
    # Find all V-cycle time sections
    for match in vcycle_pattern.finditer(text):
        end = match.end()
        # Look backward and forward to find the corresponding grid size
        # Grid size appears in Configuration Summary which comes after V-cycle time
        next_text = text[end:min(len(text), end+2000)]
        grid_match = grid_pattern.search(next_text)
        if grid_match:
            width, height, zdim = map(int, grid_match.groups())
            grid_size = f"{width}x{height}x{zdim}"
            
            time1 = float(match.group(1))  # Upper bound (sum of operations)
            time2 = float(match.group(2))  # Actual V-cycle time (Kernel Launch + V-cycle)
            # Use the actual V-cycle time (time2) which is the "Kernel Launch + V-cycle time"
            vcycle_time = time2
            
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
                'vcycle_avg_time_us': item.get('vcycle_avg_time_us', None),
                'device_rho_inf': item.get('device_rho_inf', None),
                'converged': item.get('converged', None),
                'compile_time_s': item.get('compile_time_s', None),
                'run_time_s': item.get('run_time_s', None)
            })
        else:
            combined[grid] = {
                'pe_tiles': item.get('pe_tiles', grid),
                'grid_size': grid,
                'total_grid_size': item.get('total_grid_size', 0),
                'device_iterations': item.get('device_iterations', None),
                'vcycle_avg_time_us': item.get('vcycle_avg_time_us', None),
                'device_rho_inf': item.get('device_rho_inf', None),
                'converged': item.get('converged', None),
                'compile_time_s': item.get('compile_time_s', None),
                'run_time_s': item.get('run_time_s', None)
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
    ax1.set_title('Laplacian Communication Overhead Relative to Compute (Comm/Compute Ratio)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    # ax1.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    ax1.grid(False)
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
    ax2.set_title('Laplacian Normalized Time Distribution', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax2.set_ylim(0, 130)  # Add extra space at top for labels
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=11)
    # ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.grid(False)
    
    plt.tight_layout() 
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    return fig


def parse_per_operation_timing(text: str) -> List[Dict]:
    """
    Parse "Time per operation and level" tables from the output file.
    Returns list of dictionaries, each containing grid_size and per-level operator timings.
    """
    results = []
    
    # Pattern to find grid size from Configuration Summary
    grid_pattern = re.compile(r'HeightxWidthxZDim\s*:\s*(\d+)x(\d+)x(\d+)')
    
    # Split text by "Time per operation and level" tables
    sections = text.split('Time per operation and level (us[cycles]):')
    
    for section in sections[1:]:  # Skip first empty split
        # Extract data from the table
        # Pattern to match each level row: | level | smooth | residual | restriction | interpolation | total | ...
        # Example: |         0          | 63.338us(  55421)  |  4.687us(   4101)  | ...
        
        level_data = {}
        rows = []
        
        # Find all level rows (skip header and total rows)
        # Pattern: | level | smooth | residual | restriction | interpolation | total | ...
        # Example: |         0          | 63.338us(  55421)  |  4.687us(   4101)  | ...
        # Note: The last column is "total" not "setup_init" in the new format
        row_pattern = re.compile(
            r'\|\s*(\d+)\s*\|'  # level number
            r'\s*([\d.]+)us\([^)]+\)\s*\|'  # smooth
            r'\s*([\d.]+)us\([^)]+\)\s*\|'  # residual
            r'\s*([\d.]+)us\([^)]+\)\s*\|'  # restriction
            r'\s*([\d.]+)us\([^)]+\)\s*\|'  # interpolation
            r'\s*([\d.]+)us\([^)]+\)\s*\|'  # total (was setup_init)
        )
        
        # Split section into lines and process each line
        lines = section.split('\n')
        for line in lines:
            match = row_pattern.search(line)
            if match:
                # Additional check: make sure this is not the "total" summary row
                # The total row has "total" as the first column value, not a number
                if re.search(r'\|\s*total\s*\|', line, re.IGNORECASE):
                    continue
                level = int(match.group(1))
                smooth = float(match.group(2))
                residual = float(match.group(3))
                restriction = float(match.group(4))
                interpolation = float(match.group(5))
                total = float(match.group(6))  # This is the total column, not setup_init
                
                rows.append({
                    'level': level,
                    'smooth': smooth,
                    'residual': residual,
                    'restriction': restriction,
                    'interpolation': interpolation,
                    'setup_init': 0.0  # setup_init is no longer in this table, set to 0
                })
        
        # Find grid size in this section
        grid_match = grid_pattern.search(section)
        if grid_match and rows:
            width, height, zdim = map(int, grid_match.groups())
            grid_size = f"{width}x{height}x{zdim}"
            
            results.append({
                'grid_size': grid_size,
                'levels': rows
            })
    
    return results


def plot_per_operation_timing(timing_data: Dict, output_file: str = None):
    """
    Plot per-operation timing by level for a single grid size.
    Simple and clean visualization similar to pp.py style.
    
    Args:
        timing_data: Dictionary with 'grid_size' and 'levels' (list of level data)
        output_file: Optional output filename. If None, auto-generates from grid_size.
    
    Returns:
        Figure object or None
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib not available. Cannot create plots.")
        return None
    
    if not timing_data or 'levels' not in timing_data:
        print("Warning: No timing data provided")
        return None
    
    grid_size = timing_data['grid_size']
    levels_data = timing_data['levels']
    
    if not levels_data:
        print(f"Warning: No level data for grid {grid_size}")
        return None
    
    # Sort levels by level number
    levels_data = sorted(levels_data, key=lambda x: x['level'])
    
    # Extract data - simple and clean
    levels = [d['level'] for d in levels_data]
    smooth = [d['smooth'] for d in levels_data]
    residual = [d['residual'] for d in levels_data]
    restriction = [d['restriction'] for d in levels_data]
    interpolation = [d['interpolation'] for d in levels_data]
    setup_init = [d['setup_init'] for d in levels_data]
    
    # Find last non-zero index for residual, restriction, and interpolation
    # (they have 0 at the last level, we don't want to plot those)
    def find_last_nonzero_index(values):
        """Find the last index with non-zero value."""
        for i in range(len(values) - 1, -1, -1):
            if values[i] > 0:
                return i
        return -1
    
    residual_end = find_last_nonzero_index(residual)
    restriction_end = find_last_nonzero_index(restriction)
    interpolation_end = find_last_nonzero_index(interpolation)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each operator - truncate residual, restriction, interpolation at last non-zero
    # Add markers to show exact data points
    ax.plot(levels, smooth, 'o-', label='smooth', linewidth=2, markersize=6)
    ax.plot(levels, setup_init, 's-', label='setup_init', linewidth=2, markersize=6)
    
    if residual_end >= 0:
        ax.plot(levels[:residual_end+1], residual[:residual_end+1], '^-', label='residual', linewidth=2, markersize=6)
    if restriction_end >= 0:
        ax.plot(levels[:restriction_end+1], restriction[:restriction_end+1], 'v-', label='restriction', linewidth=2, markersize=6)
    if interpolation_end >= 0:
        ax.plot(levels[:interpolation_end+1], interpolation[:interpolation_end+1], 'd-', label='interpolation', linewidth=2, markersize=6)
    
    # Simple labels
    ax.set_xlabel('Level', fontsize=12)
    ax.set_ylabel('Time (µs, log scale)', fontsize=12)
    ax.set_title(f'Timing Breakdown per Level - Grid: {grid_size}', fontsize=14, fontweight='bold')
    ax.set_yscale('log')  # Log scale to handle wide range of values
    ax.set_xticks(levels)  # Set x-axis ticks to integer levels only
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    ax.grid(True, which='major', linestyle='-', alpha=0.15)
    ax.grid(False, which='minor')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig


def plot_all_per_operation_timing(all_timing_data: List[Dict], output_dir: str = ''):
    """
    Plot per-operation timing for all grid sizes.
    Creates separate plots for each grid size.
    
    Args:
        all_timing_data: List of timing data dictionaries (one per grid size)
        output_dir: Directory to save plots (default: current directory)
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib not available. Cannot create plots.")
        return
    
    if not all_timing_data:
        print("Warning: No timing data provided")
        return
    
    prefix = f"{output_dir}/" if output_dir else ""
    
    for timing_data in all_timing_data:
        grid_size = timing_data['grid_size']
        output_file = f"{prefix}per_operation_timing_{grid_size}.png"
        plot_per_operation_timing(timing_data, output_file)
    
    print(f"\n✓ Generated {len(all_timing_data)} per-operation timing plots")


def print_per_operation_timing_tables(all_timing_data: List[Dict]):
    """
    Print per-operation timing tables for all grid sizes.
    Shows time per operation and level in a readable format.
    """
    for timing_data in all_timing_data:
        grid_size = timing_data['grid_size']
        levels_data = timing_data['levels']
        
        if not levels_data:
            continue
        
        # Sort levels by level number
        levels_data = sorted(levels_data, key=lambda x: x['level'])
        
        # Extract grid dimension for subdomain size labels
        grid_dim = int(grid_size.split('x')[0])
        
        print(f"\n{'='*100}")
        print(f"Per-Operation Timing: Grid {grid_size}")
        print(f"{'='*100}")
        
        # Header
        print(f"{'Level':<8} {'Subdomain':<12} {'smooth (us)':<15} {'residual (us)':<15} "
              f"{'restriction (us)':<18} {'interpolation (us)':<20} {'total (us)':<18}")
        print("-"*100)
        
        # Print each level
        for level_data in levels_data:
            level = level_data['level']
            subdomain_size = grid_dim // (2 ** level)
            smooth = level_data['smooth']
            residual = level_data['residual']
            restriction = level_data['restriction']
            interpolation = level_data['interpolation']
            setup_init = level_data.get('setup_init', 0.0)  # May not be present in new format
            # Calculate total if setup_init is 0 (new format)
            total = smooth + residual + restriction + interpolation + setup_init if setup_init > 0 else smooth + residual + restriction + interpolation
            
            print(f"{level:<8} {subdomain_size}³{'':<8} {smooth:<15.2f} {residual:<15.2f} "
                  f"{restriction:<18.2f} {interpolation:<20.2f} {total:<18.2f}")
        
        print(f"{'='*100}\n")


def print_configuration_table(config_data: List[Dict]):
    """
    Print configuration parameters table with problem sizes as columns and config parameters as rows.
    """
    if not config_data:
        return
    
    # Sort by grid size
    sorted_config = sorted(config_data, key=lambda x: x.get('total_grid_size', 0))
    
    # Define all configuration parameters to display
    config_params = [
        ('HeightxWidthxZDim', 'grid_size', None),
        ('Levels', 'levels', None),
        ('Max iterations', 'max_iterations', None),
        ('Tolerance (abs)', 'tolerance_abs', None),
        ('Tolerance (rel)', 'tolerance_rel', None),
        ('Pre/Post/Bottom iter', 'pre_post_bottom_iter', None),
        ('Datatype', 'datatype', None),
        ('Jacobi omega', 'jacobi_omega', lambda x: f"{x:.6f}" if x is not None else 'N/A'),
        ('Stencil alpha/beta', 'stencil_alpha_beta', None),
        ('Block size', 'block_size', None),
    ]
    
    # Get grid sizes (columns)
    grid_sizes = [d.get('grid_size', 'N/A') for d in sorted_config]
    
    print("\n" + "="*150)
    print("Configuration Parameters by Problem Size")
    print("="*150)
    
    # Print header
    header = f"{'Parameter':<30}"
    for grid_size in grid_sizes:
        header += f" {grid_size:<15}"
    print(header)
    print("-"*150)
    
    # Print each parameter row
    for param_name, param_key, formatter in config_params:
        row = f"{param_name:<30}"
        for config in sorted_config:
            value = config.get(param_key, None)
            if formatter:
                value_str = formatter(value)
            elif value is None:
                value_str = 'N/A'
            else:
                value_str = str(value)
            row += f" {value_str:<15}"
        print(row)
    
    print("="*150 + "\n")


def print_summary_table(data: List[Dict]):
    """
    Print a summary table of the parsed data with updated column names and units.
    """
    print("\n" + "="*150)
    print("Performance Data Summary")
    print("="*150)
    header = (f"{'PE tiles':<15} {'Total Grid size':<18} {'Comm Time(spmv) (us)':<25} "
              f"{'Compute Time(spmv) (us)':<28} {'V-cycle Time (us)':<20} "
              f"{'Iterations':<12} {'1-V cycle time(Average) (us)':<30} {'|rho|_inf':<15} {'Converged':<10} "
              f"{'Compile Time (s)':<18} {'Run Time (s)':<15}")
    print(header)
    print("-"*150)
    
    for d in data:
        pe_tiles = d.get('pe_tiles', 'N/A')
        total_grid = d.get('total_grid_size', 0)
        comm = d.get('comm_time_us', 0)
        comp = d.get('compute_time_us', 0)
        vcycle = d.get('vcycle_time_us', 0)
        iterations = d.get('device_iterations', None)
        vcycle_avg = d.get('vcycle_avg_time_us', None)
        rho_inf = d.get('device_rho_inf', None)
        converged = d.get('converged', None)
        compile_time = d.get('compile_time_s', None)
        run_time = d.get('run_time_s', None)
        
        iter_str = str(iterations) if iterations is not None else 'N/A'
        vcycle_avg_str = f"{vcycle_avg:.3f}" if vcycle_avg is not None else 'N/A'
        rho_str = f"{rho_inf:.3e}" if rho_inf is not None else 'N/A'
        conv_str = 'Yes' if converged else 'No' if converged is False else 'N/A'
        compile_str = f"{compile_time:.3f}" if compile_time is not None else 'N/A'
        run_str = f"{run_time:.3f}" if run_time is not None else 'N/A'
        
        print(f"{pe_tiles:<15} {total_grid:<18,} {comm:<25.2f} {comp:<28.2f} "
              f"{vcycle:<20.2f} {iter_str:<12} {vcycle_avg_str:<30} {rho_str:<15} {conv_str:<10} "
              f"{compile_str:<18} {run_str:<15}")
    
    print("="*150 + "\n")


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
    
    # Step 2.5: Parse and print configuration table
    print("Parsing configuration data...")
    config_data = parse_configuration_summary(text)
    if config_data:
        print_configuration_table(config_data)
    
    # Step 3: Print summary table
    print_summary_table(data)
    
    # Step 4: Parse per-operation timing data
    print("Parsing per-operation timing data...")
    all_timing_data = parse_per_operation_timing(text)
    
    # Step 5: Print per-operation timing tables
    if all_timing_data:
        print_per_operation_timing_tables(all_timing_data)
    
    # Step 6: Generate plots if matplotlib is available
    if HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        plot_comm_vs_compute(data, 'comm_vs_compute_time.png')
        if all_timing_data:
            plot_all_per_operation_timing(all_timing_data)
        print("\n✓ All plots generated successfully!")
    else:
        print("\n⚠ Plots not generated. Install matplotlib and numpy to create visualizations:")
        print("   pip install matplotlib numpy")


if __name__ == "__main__":
    main()
