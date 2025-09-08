"""
Example integration of RooflineCalculator with existing CSL benchmarks.

This script shows how to modify your existing benchmark code to generate
roofline plot data automatically.
"""

import numpy as np
import pandas as pd
from roofline_calculator import RooflineCalculator


def integrate_roofline_with_existing_benchmark():
    """
    Example of how to integrate RooflineCalculator with your existing benchmark code.
    
    This function shows the pattern you should follow in your actual benchmark scripts.
    """
    
    # Example parameters (replace with your actual values)
    width, height = 64, 32
    pe_length = 1
    loop_count = 100
    frequency = 0.85  # GHz
    isCS2 = False  # True for CS2, False for CS3
    
    # Matrix dimensions for your benchmark
    N, K, M = 1024, 1024, 64
    density = 20  # 80% sparse
    
    # Matrix format being tested
    matrix_format = "CSR"  # or "COO", "CSC", "ELLPACK", "GEMM"
    operation_type = "SpMM"  # or "GeMM"
    
    print("="*80)
    print("ROOFLINE INTEGRATION EXAMPLE")
    print("="*80)
    
    # Step 1: Initialize RooflineCalculator
    print("Step 1: Initialize RooflineCalculator")
    # Note: You'll need to pass your actual runner object here
    # calc = RooflineCalculator(runner, width, height, pe_length, loop_count, frequency, isCS2,
    #                          matrix_format, operation_type)
    
    # For this example, we'll create a mock calculator
    calc = RooflineCalculator(None, width, height, pe_length, loop_count, frequency, isCS2,
                             matrix_format, operation_type)
    
    # Step 2: Set matrix dimensions
    print("Step 2: Set matrix dimensions")
    calc.set_matrix_dimensions(N, K, M, density)
    print(f"  Matrix: {N}x{K} x {K}x{M} = {N}x{M}, Density: {density}%")
    
    # Step 3: Calculate theoretical metrics
    print("Step 3: Calculate theoretical metrics")
    flops = calc.calculate_flops()
    rel_acc, abs_acc = calc.calculate_memory_accesses(matrix_format.lower())
    
    print(f"  FLOPS: {flops:,}")
    print(f"  Memory Accesses: {abs_acc:,} bytes")
    print(f"  Intensity: {flops/abs_acc:.6f} flops/byte")
    
    # Step 4: Simulate timing measurements (replace with your actual timing code)
    print("Step 4: Simulate timing measurements")
    # In your actual code, you would do:
    # calc.tic()
    # # ... run your kernel ...
    # calc.toc()
    # cycles_array = calc.copy_back_compute_time(symbol_time, width, height, data_type, layout)
    
    # For this example, simulate some cycle measurements
    np.random.seed(42)  # For reproducible results
    base_cycles = 15000 if matrix_format == "GEMM" else 20000
    cycles_array = np.random.normal(base_cycles, base_cycles * 0.1, (height, width))
    
    # Step 5: Calculate performance metrics
    print("Step 5: Calculate performance metrics")
    metrics = calc.calculate_performance_metrics(cycles_array)
    
    print(f"  Avg Cycles: {metrics['avg_cycles']:.0f}")
    print(f"  Performance: {metrics['performance']:.6f} flops/cycle")
    
    # Step 6: Generate CSV data
    print("Step 6: Generate CSV data")
    csv_row = calc.generate_roofline_csv_row()
    print("  CSV Row generated with fields:")
    for key, value in csv_row.items():
        print(f"    {key}: {value}")
    
    # Step 7: Save to CSV (commented out to avoid creating files in example)
    print("Step 7: Save to CSV")
    # calc.save_roofline_csv(f"{matrix_format}_benchmark.csv")
    print("  (CSV save commented out for example)")
    
    # Step 8: Print summary
    print("Step 8: Print summary")
    calc.print_roofline_summary()
    
    return calc, csv_row


def create_benchmark_template():
    """
    Create a template for modifying your existing benchmark scripts.
    """
    
    template = '''
# TEMPLATE FOR INTEGRATING ROOFLINE CALCULATOR
# Add this to your existing benchmark scripts

from roofline_calculator import RooflineCalculator

# 1. Initialize calculator (add after your existing TimingCalculator setup)
calc = RooflineCalculator(runner, width, height, pe_length, loop_count, frequency, isCS2,
                         matrix_format="CSR", operation_type="SpMM")

# 2. Set matrix dimensions (add after you know your matrix size)
calc.set_matrix_dimensions(N=1024, K=1024, M=64, density=20)  # 80% sparse

# 3. Calculate theoretical metrics (add before running kernel)
flops = calc.calculate_flops()
rel_acc, abs_acc = calc.calculate_memory_accesses("csr")

# 4. Replace your existing timing code with:
calc.tic()
# ... your kernel execution ...
calc.toc()
cycles_array = calc.copy_back_compute_time(symbol_time, width, height, data_type, layout)

# 5. Calculate performance metrics (add after timing)
metrics = calc.calculate_performance_metrics(cycles_array)

# 6. Save results (add at the end of your benchmark)
calc.save_roofline_csv("CSR_benchmark.csv")
calc.print_roofline_summary()

# 7. Optional: Generate consolidated CSV for multiple runs
# from roofline_calculator import create_roofline_csv_from_multiple_runs
# all_runs_data = [calc1.generate_roofline_csv_row(), calc2.generate_roofline_csv_row(), ...]
# create_roofline_csv_from_multiple_runs(all_runs_data, "all_benchmarks.csv")
'''
    
    print("="*80)
    print("BENCHMARK INTEGRATION TEMPLATE")
    print("="*80)
    print(template)


def demonstrate_multiple_formats():
    """
    Demonstrate how to benchmark multiple sparse formats and generate consolidated data.
    """
    
    print("="*80)
    print("MULTIPLE FORMAT BENCHMARKING EXAMPLE")
    print("="*80)
    
    # Common parameters
    width, height = 64, 32
    N, K, M = 1024, 1024, 64
    density = 20  # 80% sparse
    
    formats = ["GEMM", "COO", "CSR", "CSC", "ELLPACK"]
    all_results = []
    
    for fmt in formats:
        print(f"\nBenchmarking {fmt} format...")
        
        # Create calculator for this format
        calc = RooflineCalculator(None, width, height, 1, 100, 0.85, False, fmt, "SpMM")
        calc.set_matrix_dimensions(N, K, M, density)
        
        # Calculate metrics
        flops = calc.calculate_flops()
        rel_acc, abs_acc = calc.calculate_memory_accesses(fmt.lower())
        
        # Simulate different performance for different formats
        base_cycles = {
            "GEMM": 15000,
            "COO": 25000, 
            "CSR": 20000,
            "CSC": 22000,
            "ELLPACK": 18000
        }
        
        cycles_array = np.random.normal(base_cycles[fmt], base_cycles[fmt] * 0.1, (height, width))
        metrics = calc.calculate_performance_metrics(cycles_array)
        
        # Generate CSV row
        csv_row = calc.generate_roofline_csv_row()
        all_results.append(csv_row)
        
        print(f"  {fmt}: {metrics['performance']:.6f} flops/cycle, intensity: {flops/abs_acc:.6f}")
    
    # Create consolidated CSV
    print(f"\nCreating consolidated CSV with {len(all_results)} formats...")
    df = pd.DataFrame(all_results)
    
    # Save to file (commented out for example)
    # df.to_csv("all_formats_benchmark.csv", index=False)
    print("Consolidated data ready for roofline plotting!")
    
    return df


if __name__ == "__main__":
    # Run examples
    calc, csv_row = integrate_roofline_with_existing_benchmark()
    print("\n")
    create_benchmark_template()
    print("\n")
    df = demonstrate_multiple_formats()
    
    print("\n" + "="*80)
    print("INTEGRATION COMPLETE!")
    print("="*80)
    print("Next steps:")
    print("1. Copy roofline_calculator.py to your timing library")
    print("2. Modify your existing benchmark scripts using the template above")
    print("3. Run your benchmarks to generate CSV data")
    print("4. Use the generated CSV files with your existing roofline_plot.py")
    print("="*80)
