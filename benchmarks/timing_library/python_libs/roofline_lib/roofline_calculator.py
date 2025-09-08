"""
Roofline Calculator for CSL Performance Analysis

This module extends the TimingCalculator to generate roofline plot data
by calculating FLOPS, memory accesses, and performance metrics.
"""

import numpy as np
import pandas as pd
import datetime
import sys
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from my_timeit import TimingCalculator


class RooflineCalculator(TimingCalculator):
    """
    Extended TimingCalculator that generates roofline plot data.
    
    Calculates:
    - FLOPS (Floating Point Operations)
    - Memory accesses (relative and absolute)
    - Performance metrics (intensity, cycles, bandwidth)
    """
    
    def __init__(self, runner, width, height, pe_length, loop_count, frequency, isCS2, 
                 matrix_format="unknown", operation_type="unknown"):
        super().__init__(runner, width, height, pe_length, loop_count, frequency, isCS2)
        
        # Matrix and operation metadata
        self.matrix_format = matrix_format  # "GEMM", "COO", "CSR", "CSC", "ELLPACK"
        self.operation_type = operation_type  # "SpMM", "GeMM", etc.
        
        # Roofline-specific metrics
        self.total_flops = 0
        self.total_relative_accesses = 0
        self.total_absolute_accesses = 0
        self.avg_cycles = 0
        self.min_cycles = 0
        self.max_cycles = 0
        
        # Matrix dimensions for FLOPS calculation
        self.N = 0  # Matrix A rows
        self.K = 0  # Matrix A cols / Matrix B rows  
        self.M = 0  # Matrix B cols
        self.density = 100  # Sparsity percentage (100 = dense)
        
    def set_matrix_dimensions(self, N: int, K: int, M: int, density: int = 100):
        """
        Set matrix dimensions for FLOPS calculation.
        
        Args:
            N: Number of rows in matrix A
            K: Number of columns in matrix A (and rows in matrix B)
            M: Number of columns in matrix B
            density: Sparsity percentage (100 = dense, 20 = 80% sparse)
        """
        self.N = N
        self.K = K
        self.M = M
        self.density = density
        
    def calculate_flops(self) -> int:
        """
        Calculate total FLOPS for the operation.
        
        For SpMM/GeMM: FLOPS = 2 * N * K * M * (density/100)
        (2 because each multiply-add counts as 2 operations)
        
        Returns:
            Total number of floating point operations
        """
        if self.N == 0 or self.K == 0 or self.M == 0:
            raise ValueError("Matrix dimensions not set. Call set_matrix_dimensions() first.")
            
        # Base FLOPS for dense matrix multiplication
        base_flops = 2 * self.N * self.K * self.M
        
        # Adjust for sparsity
        self.total_flops = int(base_flops * (self.density / 100.0))
        return self.total_flops
        
    def calculate_memory_accesses(self, sparse_format: str = "dense") -> Tuple[int, int]:
        """
        Calculate memory accesses for different sparse formats.
        
        Args:
            sparse_format: "dense", "coo", "csr", "csc", "ellpack"
            
        Returns:
            Tuple of (relative_accesses, absolute_accesses)
        """
        if self.N == 0 or self.K == 0 or self.M == 0:
            raise ValueError("Matrix dimensions not set. Call set_matrix_dimensions() first.")
            
        # Dense matrix memory accesses
        if sparse_format.lower() == "dense":
            # A: N*K, B: K*M, C: N*M (read A, read B, write C)
            relative_accesses = self.N * self.K + self.K * self.M + self.N * self.M
            absolute_accesses = relative_accesses * 4  # 4 bytes per float32
            
        # COO (Coordinate) format
        elif sparse_format.lower() == "coo":
            # Only access non-zero elements
            nnz = int(self.N * self.K * (self.density / 100.0))
            # COO: (row, col, val) + B matrix + C matrix
            relative_accesses = nnz * 3 + self.K * self.M + self.N * self.M
            absolute_accesses = relative_accesses * 4
            
        # CSR (Compressed Sparse Row) format  
        elif sparse_format.lower() == "csr":
            nnz = int(self.N * self.K * (self.density / 100.0))
            # CSR: row_ptr (N+1), col_ind (nnz), val (nnz) + B + C
            relative_accesses = (self.N + 1) + nnz + nnz + self.K * self.M + self.N * self.M
            absolute_accesses = relative_accesses * 4
            
        # CSC (Compressed Sparse Column) format
        elif sparse_format.lower() == "csc":
            nnz = int(self.N * self.K * (self.density / 100.0))
            # CSC: col_ptr (K+1), row_ind (nnz), val (nnz) + B + C
            relative_accesses = (self.K + 1) + nnz + nnz + self.K * self.M + self.N * self.M
            absolute_accesses = relative_accesses * 4
            
        # ELLPACK format
        elif sparse_format.lower() == "ellpack":
            nnz = int(self.N * self.K * (self.density / 100.0))
            # ELLPACK: col_ind (N*max_nnz_per_row), val (N*max_nnz_per_row) + B + C
            max_nnz_per_row = int(nnz / self.N) + 1  # Approximate
            relative_accesses = self.N * max_nnz_per_row * 2 + self.K * self.M + self.N * self.M
            absolute_accesses = relative_accesses * 4
            
        else:
            raise ValueError(f"Unknown sparse format: {sparse_format}")
            
        self.total_relative_accesses = relative_accesses
        self.total_absolute_accesses = absolute_accesses
        return relative_accesses, absolute_accesses
        
    def calculate_performance_metrics(self, cycles_array: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics from cycles data.
        
        Args:
            cycles_array: Array of cycle measurements
            
        Returns:
            Dictionary with performance metrics
        """
        self.avg_cycles = float(np.mean(cycles_array))
        self.min_cycles = float(np.min(cycles_array))
        self.max_cycles = float(np.max(cycles_array))
        
        # Calculate roofline metrics
        total_cycles = self.avg_cycles * self.width * self.height
        
        # Intensity = FLOPS / Bytes
        intensity = self.total_flops / self.total_absolute_accesses if self.total_absolute_accesses > 0 else 0
        
        # Performance = FLOPS / Cycles  
        performance = self.total_flops / total_cycles if total_cycles > 0 else 0
        
        return {
            "intensity": intensity,
            "performance": performance,
            "avg_cycles": self.avg_cycles,
            "min_cycles": self.min_cycles,
            "max_cycles": self.max_cycles,
            "total_cycles": total_cycles
        }
        
    def generate_roofline_csv_row(self) -> Dict[str, any]:
        """
        Generate a single row of data for roofline CSV.
        
        Returns:
            Dictionary with all required fields for roofline plotting
        """
        return {
            "width": self.width,
            "height": self.height,
            "N": self.N,
            "K": self.K,
            "M": self.M,
            "density": self.density,
            "avg_cycles": self.avg_cycles,
            "min_cycles": self.min_cycles,
            "max_cycles": self.max_cycles,
            "total_relative_accesses": self.total_relative_accesses,
            "total_absolute_accesses": self.total_absolute_accesses,
            "total_flops": self.total_flops,
            "matrix_format": self.matrix_format,
            "operation_type": self.operation_type,
            "WSE": self.WSE,
            "frequency": self.frequency
        }
        
    def save_roofline_csv(self, filename: Optional[str] = None, append: bool = True):
        """
        Save roofline data to CSV file.
        
        Args:
            filename: Output filename (auto-generated if None)
            append: Whether to append to existing file or create new one
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"roofline_{self.matrix_format}_{self.WSE}_{timestamp}.csv"
            
        # Generate data row
        data_row = self.generate_roofline_csv_row()
        
        # Create DataFrame
        df = pd.DataFrame([data_row])
        
        # Save to CSV
        if append and os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, index=False)
            
        print(f"Roofline data saved to: {filename}")
        
    def print_roofline_summary(self):
        """Print a summary of roofline metrics."""
        print("\n" + "="*60)
        print("ROOFLINE PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Matrix Format: {self.matrix_format}")
        print(f"Operation: {self.operation_type}")
        print(f"WSE: {self.WSE}")
        print(f"Dimensions: {self.N}x{self.K} x {self.K}x{self.M} = {self.N}x{self.M}")
        print(f"Density: {self.density}%")
        print(f"Grid Size: {self.width}x{self.height}")
        print("-"*60)
        print(f"Total FLOPS: {self.total_flops:,}")
        print(f"Memory Accesses: {self.total_absolute_accesses:,} bytes")
        print(f"Intensity: {self.total_flops/self.total_absolute_accesses:.6f} flops/byte")
        print(f"Avg Cycles: {self.avg_cycles:.0f}")
        print(f"Performance: {self.total_flops/(self.avg_cycles*self.width*self.height):.6f} flops/cycle")
        print("="*60)

    def plot_roofline_from_csv(self,
                               csv_path: str,
                               savefile: str):
        """
        Plot a roofline using a single CSV input, following the reference style.
        Thanks to https://github.com/pr0f3ss/SpMM_Cerebras/blob/main/plots/roofline_plot.py !!

        Args:
            csv_path: Path to the CSV containing columns: width,height,density,avg_cycles,total_absolute_accesses,total_flops.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        figtext = "Cerebras WSE-" + self.WSE
        xlabel = "I(n) [flops/byte]"
        ylabel = "P(n) [flops/cycle]"
        title = f"Performance of {self.operation_type} implementations with {100 - self.density}% sparsity."
        ax = plt.gca()

        # initialize plot
        plt.xlabel(xlabel, loc="center")
        plt.ylabel(ylabel, rotation="horizontal", labelpad=0, alpha=0.75, loc="top")
        ax.yaxis.set_label_coords(0.1, 1.015)
        plt.figtext(0.73, 0.895, figtext, alpha=0.75)
        plt.title(title, y=1.06, loc="center")

        # retrieve dataset
        df = pd.read_csv(csv_path)
        # check if this is correct entry
        df = df[df["density"] == self.density]

        # Plot data
        gemm_flops = df["total_flops"]
        gemm_bytes = df["total_absolute_accesses"]
        gemm_cycles = df["avg_cycles"]*df["width"]*df["height"]
        # Operational Intensity
        x1 = gemm_flops / gemm_bytes
        # Performance
        y1 = gemm_flops / gemm_cycles
        
        # Plot
        fmt1 = "-v"
        plt.plot(x1, y1, fmt1, color="orangered", label=f"{self.matrix_format}")

        # Other plot settings, labels, Memory bound line, legend etc...
        plt.axhline(y=2, color='k', ls=':')
        plt.axvline(x=2/12, color='k', ls='--', alpha=0.6)
        plt.text(0.05, 0.95,'Peak Performance', color='k', fontdict={'size': 'smaller'},  transform=ax.transAxes)
        plt.text(0.1, 0.8,'Memory Bandwidth', color='k', fontdict={'size': 'smaller'},  transform=ax.transAxes, rotation=6.0)
        plt.text(0.835, 0.35,'Memory/Compute Bound', color='k', fontdict={'size': 'smaller'},  transform=ax.transAxes, rotation=270.0)
        # Draw mem line
        mem_x = np.linspace(0.15, 0.17, 200)
        mem_y = 12.0*mem_x
        plt.plot(mem_x, mem_y, color='k', alpha=0.6)
        # Draw legend
        plt.legend(loc=(1.04, 0.65))
        plt.grid()
        # Set limits
        plt.xlim([0.15, 0.17])
        plt.ylim([0.425, 2.1])
        # save
        plt.savefig(savefile, bbox_inches='tight', format='png')

# Example usage functions
def example_usage_function():
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
        calc.plot_roofline_from_csv("CSR_benchmark.csv", savefile="CSR_benchmark.png")
    '''
    
    print("="*80)
    print("BENCHMARK INTEGRATION TEMPLATE")
    print("="*80)
    print(template)

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
    
    # Step 7: Save to CSV
    print("Step 7: Save to CSV")
    calc.save_roofline_csv(f"{matrix_format}_benchmark.csv")
    
    # Step 8: Print summary
    print("Step 8: Print summary")
    calc.print_roofline_summary()

    # Step 9: Plot roofline
    print("Step 9: Plot roofline")
    calc.plot_roofline_from_csv(f"{matrix_format}_benchmark.csv", savefile=f"{matrix_format}_benchmark.png")
    print(f"Roofline plot saved to: {matrix_format}_benchmark.png")
    
    return calc, csv_row


if __name__ == "__main__":
    example_usage_function()
    # Integrated example
    calc, csv_row = integrate_roofline_with_existing_benchmark()
    print("\n")
    
    print("\n" + "="*80)
    print("INTEGRATION COMPLETE!")
    print("="*80)
    print("Next steps:")
    print("1. Import roofline_calculator.py to your timing library")
    print("2. Modify your existing benchmark scripts using the template/example above")
    print("3. Run your benchmarks to generate CSV data")
    print("4. Use the generated CSV files to plot the roofline")
    print("="*80)
