# Single Layer Implementation

This folder contains the implementation of a single layer for the Algebraic Multigrid (AMG) solver, specifically focusing on the Jacobi iteration method. The code is designed to run on Cerebras hardware and includes tools for performance analysis and visualization.

## Overview

The single layer implementation serves as a building block for more complex multilayer AMG solvers. It implements:
- Jacobi iteration for solving linear systems (Ax = b)
- Matrix operations in a distributed computing environment
- Performance monitoring and analysis
- Interactive visualization of results

## Files and Their Functions

### Core Implementation
- `single_layer_layout.csl`: Defines the hardware layout and configuration for the Cerebras system
- `single_layer_pe.csl`: Contains the Processing Element (PE) implementation code
- `single_layer_run.py`: Main runner script that orchestrates the execution on hardware
- `jacobi_only.py`: Pure Python implementation of the Jacobi iteration method (used for validation)

### Analysis Tools
- `analyze_sim_log.py`: Analyzes simulation logs and generates static performance visualizations
- `interactive_analysis.py`: Creates interactive dashboards for performance analysis
- `time_utils.py`: Utilities for timing analysis and performance metrics
- `util.py`: General utility functions for matrix operations and data handling

### Output Files
The following files are generated during execution:
- `performance_analysis.png`: Static visualization of performance metrics
- `simulation_dashboard.html`: Interactive dashboard showing detailed performance analysis
- `simulation_dashboard.png`: Static version of the dashboard
- `single_layer_timing_runs.csv`: Raw timing data in CSV format

## Workflow

1. **Setup and Configuration**
   - Define matrix dimensions and kernel sizes
   - Configure hardware layout using `single_layer_layout.csl`
   - Set up processing elements using `single_layer_pe.csl`

2. **Execution**
   - Run the solver using `single_layer_run.py`
   - Monitor execution through simulation logs
   - Collect timing and performance data

3. **Analysis**
   - Process simulation logs using `analyze_sim_log.py`
   - Generate interactive visualizations with `interactive_analysis.py`
   - Analyze timing data using utilities in `time_utils.py`

## How to Run

1. Basic execution:
Runs various configurations and generates the output in the `out` directory.
```bash
./run_single_layer_wse2.sh
```

## Performance Analysis

The implementation includes comprehensive performance analysis tools that provide:
- Instruction distribution analysis
- PE utilization metrics
- Pipeline usage statistics
- Operation type distribution
- Cycle-level performance data
- Timing breakdowns

### How to Run

1. Generate performance analysis:   
```bash
python analyze_sim_log.py --log_file <path_to_sim_log>
```


2. Run interactive analysis:
```bash
python interactive_analysis.py --log_file <path_to_sim_log>
```


### Visualization Types
1. **Static Analysis** (`analyze_sim_log.py`):
   - Instruction distribution
   - PE utilization
   - Pipeline usage
   - Operation type distribution
   - Instructions per cycle

2. **Interactive Dashboard** (`interactive_analysis.py`):
   - Real-time performance metrics
   - Drill-down capabilities
   - Dynamic filtering
   - Custom view configurations

## Dependencies

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly (for interactive visualizations)

## Notes
- Input matrix dimensions should be square.
- PE dimensions should be square.(TODO: Make non-square in the future)
- Use interactive dashboard for detailed performance analysis
- Refer to timing data in .csv file


## Core Operators:
```
1. Jacobi Iteration (compute_jacobi task):
   x = (I - ωD⁻¹A)x + ωD⁻¹b
   where:
   - D is diagonal of A
   - ω is relaxation parameter
   - Implementation splits this into:
     a. term1 = (I - ωD⁻¹A)x
     b. term2 = ωD⁻¹b
     c. x_new = term1 + term2

2. Residual Computation (compute_residual task):
   r = b - Ax

3. Restriction (compute_restriction task):
   b_next = R × r
```

## Taskflow:
```
main() → compute_jacobi() → bcast_jacobi() → compute_residual() → bcast_residual() → compute_restriction() → EXIT
```



