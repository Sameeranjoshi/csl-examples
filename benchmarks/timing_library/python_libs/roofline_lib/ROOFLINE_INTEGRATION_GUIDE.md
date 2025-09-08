# Roofline Integration Guide for CSL Benchmarks

## Overview

This guide explains how to integrate roofline plot data generation with your existing CSL benchmarks using the `TimingCalculator` library.

## What You Need for Roofline Plots

### Required Metrics
1. **FLOPS (Floating Point Operations)**: Total number of floating point operations
2. **Memory Accesses**: Total bytes accessed (absolute and relative)
3. **Cycles**: Execution time in cycles
4. **Matrix Dimensions**: N, K, M for matrix multiplication
5. **Sparsity**: Density percentage (100 = dense, 20 = 80% sparse)

### Roofline Calculations
- **Intensity (I)**: `FLOPS / Memory_Bytes`
- **Performance (P)**: `FLOPS / Total_Cycles`

## Files Created

### 1. `roofline_calculator.py`
Extended `TimingCalculator` class that adds:
- FLOPS calculation for SpMM/GeMM operations
- Memory access counting for different sparse formats
- Performance metrics calculation
- CSV data generation for roofline plotting

### 2. `roofline_integration_example.py`
Example code showing how to integrate the roofline calculator with your existing benchmarks.

## Quick Start

### Step 1: Import the RooflineCalculator
```python
from roofline_calculator import RooflineCalculator
```

### Step 2: Initialize with your benchmark parameters
```python
calc = RooflineCalculator(runner, width, height, pe_length, loop_count, frequency, isCS2,
                         matrix_format="CSR", operation_type="SpMM")
```

### Step 3: Set matrix dimensions
```python
calc.set_matrix_dimensions(N=1024, K=1024, M=64, density=20)  # 80% sparse
```

### Step 4: Calculate theoretical metrics
```python
flops = calc.calculate_flops()
rel_acc, abs_acc = calc.calculate_memory_accesses("csr")
```

### Step 5: Run your timing measurements
```python
calc.tic()
# ... your kernel execution ...
calc.toc()
cycles_array = calc.copy_back_compute_time(symbol_time, width, height, data_type, layout)
```

### Step 6: Calculate performance metrics
```python
metrics = calc.calculate_performance_metrics(cycles_array)
```

### Step 7: Save results
```python
calc.save_roofline_csv("CSR_benchmark.csv")
calc.print_roofline_summary()
```

## Supported Sparse Formats

- **Dense (GEMM)**: Standard dense matrix multiplication
- **COO**: Coordinate format
- **CSR**: Compressed Sparse Row
- **CSC**: Compressed Sparse Column  
- **ELLPACK**: ELLPACK format

## CSV Output Format

The generated CSV files match the format expected by your existing `roofline_plot.py`:

```csv
width,height,N,K,M,density,avg_cycles,min_cycles,max_cycles,total_relative_accesses,total_absolute_accesses,total_flops,matrix_format,operation_type,WSE,frequency
64,32,1024,1024,64,20,20000,18000,22000,52428800,209715200,16777216,CSR,SpMM,CS3,0.85
```

## Integration with Existing Benchmarks

### Before (your current code):
```python
from my_timeit import TimingCalculator

calc = TimingCalculator(runner, width, height, pe_length, loop_count, frequency, isCS2)
calc.tic()
# ... kernel execution ...
calc.toc()
cycles_array = calc.copy_back_compute_time(symbol_time, width, height, data_type, layout)
```

### After (with roofline integration):
```python
from roofline_calculator import RooflineCalculator

calc = RooflineCalculator(runner, width, height, pe_length, loop_count, frequency, isCS2,
                         matrix_format="CSR", operation_type="SpMM")
calc.set_matrix_dimensions(N=1024, K=1024, M=64, density=20)
flops = calc.calculate_flops()
rel_acc, abs_acc = calc.calculate_memory_accesses("csr")

calc.tic()
# ... kernel execution ...
calc.toc()
cycles_array = calc.copy_back_compute_time(symbol_time, width, height, data_type, layout)

metrics = calc.calculate_performance_metrics(cycles_array)
calc.save_roofline_csv("CSR_benchmark.csv")
```

## Multiple Format Benchmarking

To benchmark multiple sparse formats and create consolidated data:

```python
formats = ["GEMM", "COO", "CSR", "CSC", "ELLPACK"]
all_results = []

for fmt in formats:
    calc = RooflineCalculator(runner, width, height, pe_length, loop_count, frequency, isCS2,
                             matrix_format=fmt, operation_type="SpMM")
    calc.set_matrix_dimensions(N, K, M, density)
    
    # ... run benchmark ...
    
    csv_row = calc.generate_roofline_csv_row()
    all_results.append(csv_row)

# Create consolidated CSV
from roofline_calculator import create_roofline_csv_from_multiple_runs
create_roofline_csv_from_multiple_runs(all_results, "all_formats_benchmark.csv")
```

## Using with Existing Roofline Plot Script

The generated CSV files are compatible with your existing `roofline_plot.py`. Simply update the file paths in the script:

```python
# In roofline_plot.py, update these lines:
filepath_gemm = "../benchmarks/GEMM_benchmark.csv"
filepath_coo = "../benchmarks/COO_benchmark.csv"
filepath_csc = "../benchmarks/CSC_benchmark.csv"
filepath_csr = "../benchmarks/CSR_benchmark.csv"
filepath_ellpack = "../benchmarks/ELLPACK_benchmark.csv"
```

## Key Benefits

1. **Reusable**: Works with any CSL benchmark
2. **Automatic**: Calculates FLOPS and memory accesses automatically
3. **Compatible**: Generates CSV format compatible with existing roofline plots
4. **Comprehensive**: Supports all major sparse formats
5. **Extensible**: Easy to add new formats or metrics

## Example Output

```
============================================================
ROOFLINE PERFORMANCE SUMMARY
============================================================
Matrix Format: CSR
Operation: SpMM
WSE: CS3
Dimensions: 1024x1024 x 1024x64 = 1024x64
Density: 20%
Grid Size: 64x32
------------------------------------------------------------
Total FLOPS: 16,777,216
Memory Accesses: 209,715,200 bytes
Intensity: 0.080000 flops/byte
Avg Cycles: 20,000
Performance: 0.013107 flops/cycle
============================================================
```

## Next Steps

1. Copy `roofline_calculator.py` to your timing library directory
2. Modify your existing benchmark scripts using the integration template
3. Run your benchmarks to generate CSV data
4. Use the generated CSV files with your existing `roofline_plot.py`
5. Generate beautiful roofline plots showing performance across different sparse formats!

## Troubleshooting

### Common Issues

1. **Matrix dimensions not set**: Call `set_matrix_dimensions()` before calculating FLOPS
2. **Unknown sparse format**: Use one of: "dense", "coo", "csr", "csc", "ellpack"
3. **Missing runner object**: Pass your actual runner object when initializing

### Debug Tips

- Use `calc.print_roofline_summary()` to verify all calculations
- Check that FLOPS and memory access values are reasonable
- Verify CSV output format matches your roofline plot script expectations
