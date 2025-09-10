"""
Roofline Calculator for CSL Performance Analysis

This module extends the TimingCalculator to generate roofline plot data
by calculating FLOPS, memory accesses, and performance metrics.
"""

import time
import numpy as np
import pandas as pd
import datetime
import sys
import os
import seaborn as sns
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
        
        # Roofline-specific metrics(actual)
        # self.total_flops = 0
        # self.total_relative_accesses = 0
        # self.total_absolute_accesses = 0
        self.empirical_avg_cycles = 0
        self.empirical_min_cycles = 0
        self.empirical_max_cycles = 0

        self.empirical_flops = 0  # FLOPs
        self.empirical_accesses = 0  # Bytes
        self.empirical_cycles = 0 # Cycles  # need to decide what type it is
        self.empirical_time = 0 # Seconds
        self.empirical_bw = 0 # Bytes/Second
        self.empirical_performance = 0    # FLOPs/Second = FLOPS
        self.empirical_AI = 0  # FLOPs/Bytes


        # Theoretical performance - modeling metrics
        # FLOPs = count 
        # FLOPS = FLOPs / time
        self.theoretical_flops = 0  # FLOPs
        self.theoretical_accesses = 0  # Bytes
        self.theoretical_cycles = 0 # Cycles
        self.theoretical_time = 0 # Seconds
        self.theoretical_bw = 0 # Bytes/Second
        self.theoretical_performance = 0    # FLOPs/Second = FLOPS
        self.theoretical_AI = 0  # FLOPs/Bytes
        
        # Input problem
        # Matrix dimensions for FLOPS calculation
        self.M = 0  # Matrix A rows
        self.K = 0  # Matrix A cols / Matrix B rows  
        self.N = 0  # Matrix B cols
        self.density = 100  # Sparsity percentage (100 = dense)
        self.matrix_format = matrix_format  # "DENSE", "COO", "CSR", "CSC", "ELLPACK"
        self.operation_type = operation_type  # "SpMM", "GeMV", etc.

    def get_theoretical_metrics(self):
        """
        Getter for all theoretical values as a dictionary.
        Returns:
            dict: Dictionary containing all theoretical performance metrics.
        """
        return {
            "theoretical_flops": self.theoretical_flops,
            "theoretical_accesses": self.theoretical_accesses,
            "theoretical_cycles": self.theoretical_cycles,
            "theoretical_time": self.theoretical_time,
            "theoretical_bw": self.theoretical_bw,
            "theoretical_performance": self.theoretical_performance,
            "theoretical_AI": self.theoretical_AI
        }

    def get_empirical_metrics(self):
        """
        Getter for all empirical values as a dictionary.
        Returns:
            dict: Dictionary containing all empirical performance metrics.
        """
        return {
            "empirical_flops": self.empirical_flops,
            "empirical_accesses": self.empirical_accesses,
            "empirical_cycles": self.empirical_cycles,
            "empirical_min_cycles": self.empirical_min_cycles,
            "empirical_max_cycles": self.empirical_max_cycles,
            "empirical_avg_cycles": self.empirical_avg_cycles,
            "empirical_time": self.empirical_time,
            "empirical_bw": self.empirical_bw,
            "empirical_performance": self.empirical_performance,
            "empirical_AI": self.empirical_AI
        }

    def get_problem_dimensions(self):
        """
        Getter for all problem dimensions as a dictionary.
        Returns:
            dict: Dictionary containing all problem dimensions.
        """
        return {
            "M": self.M,
            "K": self.K,
            "N": self.N,
            "density": self.density,
            "matrix_format": self.matrix_format,
            "operation_type": self.operation_type
        }

    def get_hardware_configuration(self):
        """
        Getter for all hardware configuration as a dictionary.
        Returns:
            dict: Dictionary containing all hardware configuration.
        """
        return {
            "width": self.width,
            "height": self.height,
            "pe_length": self.pe_length,
            "loop_count": self.loop_count,
            "frequency": self.frequency,
            "WSE": self.WSE
        }

    def set_matrix_dimensions(self, M: int, K: int, N: int, density: int = 100):
        """
        Set matrix dimensions for FLOPS calculation.
        
        Args:
            M: Number of rows in matrix A
            K: Number of columns in matrix A (and rows in matrix B)
            N: Number of columns in matrix B
            density: Sparsity percentage (100 = dense, 20 = 80% sparse)
        """
        self.M = M
        self.K = K
        self.N = N
        self.density = density
    
    # Theoretical
    def calculate_theoretical_flops(self):
        """
        Calculate total FLOPS for the operation.
        
        For GeMV: FLOPS = M*(2*K-1) * (density/100)
        Per row: K multiply + (K-1) adds
        Total: Total rows * Per row
        Total: M*(K + (K-1))
        Total FLOPS : M*(2*K-1) * (density/100)
        Returns:
            Total number of floating point operations
        """
        if self.N == 0 or self.K == 0 or self.M == 0:
            raise ValueError("Matrix dimensions not set. Call set_matrix_dimensions() first.")
            
        # Base FLOPS for dense matrix vector multiplication
        base_flops = self.M * (2 * self.K - 1)/(self.height*self.width)
        
        # Adjust for sparsity
        self.theoretical_flops = int(base_flops * (self.density / 100.0))

        # scale to program rectangle
        self.theoretical_flops = self.theoretical_flops * (self.width * self.height)
        
    def calculate_theoretical_memory_accesses(self, sparse_format: str = "dense"):
        """
        Calculate memory accesses for different sparse formats.
        
        Args:
            sparse_format: "dense", "csr"
            
        Returns:
            Tuple of (relative_accesses, absolute_accesses)
        """
        if self.N == 0 or self.K == 0 or self.M == 0:
            raise ValueError("Matrix dimensions not set. Call set_matrix_dimensions() first.")
            
        # Dense matrix memory accesses
        if sparse_format.lower() == "dense":
            # A: M*K, B: K*N, C: M*N (read A, read B, write C)
            self.theoretical_accesses = self.M * self.K + self.K * self.N + self.M * self.N
            self.theoretical_accesses = self.theoretical_accesses * 4  # 4 bytes per float32
            
        # CSR (Compressed Sparse Row) format
        # Todo: fixit might be wrong
        elif sparse_format.lower() == "csr":
            nnz = int(self.N * self.K * (self.density / 100.0))
            # CSR: row_ptr (N+1), col_ind (nnz), val (nnz) + B + C
            self.theoretical_accesses = (self.N + 1) + nnz + nnz + self.K * self.M + self.N * self.M
            self.theoretical_accesses = self.theoretical_accesses * 4
            
        else:
            raise ValueError(f"Unknown sparse format: {sparse_format}")
        
        # scale to program rectangle
        self.theoretical_accesses = self.theoretical_accesses * (self.width * self.height)  

    def calculate_theoretical_metrics(self):
        """
        Calculate theoretical metrics.
        """
        self.calculate_theoretical_flops()
        self.calculate_theoretical_memory_accesses()
        # cycles
        self.theoretical_cycles = 100 # dummy value cycles
        self.theoretical_time = self.theoretical_cycles / self.frequency # seconds
        self.theoretical_bw = self.theoretical_accesses / self.theoretical_time # bytes/second
        self.theoretical_performance = self.theoretical_flops / self.theoretical_time # flops/second
        self.theoretical_AI = self.theoretical_flops / self.theoretical_accesses # flops/byte
        return self.get_theoretical_metrics()

    # Empirical
    def calculate_empirical_flops(self):
        # Actually should be returned from the device
        # TODO: Fix this
        self.empirical_flops = (self.K)*(self.M) + (self.K - 1)*(self.M)
        return self.empirical_flops

    def calculate_empirical_memory_accesses(self):
        # Actually should be returned from the device
        # TODO: Fix this
        self.empirical_accesses = self.K + self.M * self.K + 2 * self.M * self.M
        return self.empirical_accesses

    def calculate_empirical_metrics(self, cycles_array: np.ndarray):
        """
        Calculate empirical metrics from cycles data.
        
        Args:
            cycles_array: Array of cycle measurements
            
        Returns:
            Dictionary with performance metrics
        """
        
        # Calculate roofline metrics
        self.calculate_empirical_flops()
        self.calculate_empirical_memory_accesses()
        self.empirical_avg_cycles = float(np.mean(cycles_array))
        self.empirical_min_cycles = float(np.min(cycles_array))
        self.empirical_max_cycles = float(np.max(cycles_array))
        self.empirical_cycles = self.empirical_max_cycles  
        self.empirical_time = self.empirical_cycles / self.frequency
        self.empirical_bw = self.empirical_accesses / self.empirical_time
        self.empirical_performance = self.empirical_flops / self.empirical_time
        self.empirical_AI = self.empirical_flops / self.empirical_accesses
        return self.get_empirical_metrics()

    # Roofline + Plotting + CSV utilities
    def generate_roofline_csv_row(self) -> Dict[str, any]:
        """
        Generate a single row of data for roofline CSV.
        
        Returns:
            Dictionary with all required fields for roofline plotting
        """
        empirical_metrics = self.get_empirical_metrics()
        theoretical_metrics = self.get_theoretical_metrics()
        problem_dimensions = self.get_problem_dimensions()
        hardware_configuration = self.get_hardware_configuration()
        metrics = {**empirical_metrics, **theoretical_metrics, **problem_dimensions, **hardware_configuration}
        return metrics
        
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
        empirical_metrics = self.get_empirical_metrics()
        theoretical_metrics = self.get_theoretical_metrics()
        problem_dimensions = self.get_problem_dimensions()
        hardware_configuration = self.get_hardware_configuration()
        print("\n" + "="*60)
        print("ROOFLINE PERFORMANCE SUMMARY")
        print("="*60)
        print("empirical Metrics:")
        print(f"  FLOPS(empirical): {empirical_metrics['empirical_flops']:,}")
        print(f"  Memory Accesses(empirical): {empirical_metrics['empirical_accesses']:,} bytes")
        print(f"  Min Cycles(empirical): {empirical_metrics['empirical_min_cycles']:.0f}")
        print(f"  Max Cycles(empirical): {empirical_metrics['empirical_max_cycles']:.0f}")
        print(f"  Avg Cycles(empirical): {empirical_metrics['empirical_avg_cycles']:.0f}")
        print(f"  Time(empirical): {empirical_metrics['empirical_time']:.6f} seconds")
        print(f"  BW(empirical): {empirical_metrics['empirical_bw']:.6f} bytes/second")
        print(f"  Performance(empirical): {empirical_metrics['empirical_performance']:.6f} flops/cycle")
        print(f"  Arithmetic Intensity(empirical): {empirical_metrics['empirical_AI']:.6f} flops/byte")        
        print("-"*60)
        print("Theoretical Metrics:")
        print(f"  FLOPS(Theoretical): {theoretical_metrics['theoretical_flops']:,}")
        print(f"  Memory Accesses(Theoretical): {theoretical_metrics['theoretical_accesses']:,} bytes")
        print(f"  Arithmetic Intensity(Theoretical): {theoretical_metrics['theoretical_AI']:.6f} flops/byte")
        print(f"  Cycles(Theoretical): {theoretical_metrics['theoretical_cycles']:.0f}")
        print(f"  Time(Theoretical): {theoretical_metrics['theoretical_time']:.6f} seconds")
        print(f"  BW(Theoretical): {theoretical_metrics['theoretical_bw']:.6f} bytes/second")
        print(f"  Performance(Theoretical): {theoretical_metrics['theoretical_performance']:.6f} flops/second")
        print("-"*60)
        print("Input Problem :")
        print(f"  M: {problem_dimensions['M']}")
        print(f"  K: {problem_dimensions['K']}")
        print(f"  N: {problem_dimensions['N']}")
        print(f"  Density: {problem_dimensions['density']}%")
        print(f"  Matrix Format: {problem_dimensions['matrix_format']}")
        print(f"  Operation Type: {problem_dimensions['operation_type']}")
        print("-"*60)
        print("Hardware Configuration :")
        print(f"  PE Width: {hardware_configuration['width']}")
        print(f"  PE Height: {hardware_configuration['height']}")
        print(f"  PE Length?: {hardware_configuration['pe_length']}") # Todo: What is this parameter?
        print(f"  Loop Count: {hardware_configuration['loop_count']}")
        print(f"  Frequency: {hardware_configuration['frequency']} GHz")
        print(f"  WSE: {hardware_configuration['WSE']}")
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

    def plot_empirical_vs_theoretical_cycles(self, csv_path: str, savefile: str):
        """
        Plot empirical vs theoretical cycles against number of PEs
        using a tidy dataframe and single y-axis.
        
        Args:
            csv_path: Path to the CSV containing the benchmark data
            savefile: Output path for the plot (PNG)
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"CSV loaded: {len(df)} rows")
        
        # Compute number of PEs
        df['num_PEs'] = df['height'] * df['width']
        
        # Reshape to tidy format (for seaborn plotting)
        df_long = df.melt(
            id_vars=['num_PEs'],
            value_vars=['empirical_cycles', 'theoretical_cycles'],
            var_name='Type', value_name='Cycles'
        )
        
        # Debug print
        print(df_long.head())
        
        # Plot
        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            data=df_long,
            x='num_PEs',
            y='Cycles',
            hue='Type',
            style='Type',
            s=120
        )
        sns.lineplot(
            data=df_long,
            x='num_PEs',
            y='Cycles',
            hue='Type',
            style='Type',
            markers=False,
            dashes=True,
            linewidth=1.5,
            alpha=0.7,
            legend=False
        )
        
        # Labels & title
        plt.xlabel("Number of PEs (Height × Width)", fontsize=12)
        plt.ylabel("Cycles", fontsize=12)
        plt.title("Empirical vs Theoretical Cycles vs Number of PEs", fontsize=14, fontweight='bold')
        # plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Set x-axis to show the PE value but allow natural spread of data points
        unique_pe_values = df_long['num_PEs'].unique()
        if len(unique_pe_values) == 1:
            # If all data points have the same PE count, show that value but allow spread
            pe_value = unique_pe_values[0]
            # Set a reasonable range around the PE value to show the data points clearly
            plt.xlim(pe_value - 20, pe_value + 20)
            # Show the PE value as a tick mark
            plt.xticks([pe_value], [f"{pe_value}"])
        else:
            # If multiple PE values, set range to show all data with some padding
            pe_min, pe_max = unique_pe_values.min(), unique_pe_values.max()
            pe_range = pe_max - pe_min
            plt.xlim(pe_min - pe_range * 0.1, pe_max + pe_range * 0.1)
        
        # Add configuration info (from first row)
        first_row = df.iloc[0]
        legend_text = f"""Configuration:
Density: {first_row['density']}%
Format: {first_row['matrix_format']}
Operation: {first_row['operation_type']}
AX+B; A = {first_row['M']}×{first_row['K']}, X = {first_row['K']}×{first_row['N']}, B = {first_row['M']}×{first_row['N']}
{first_row['WSE']}"""
                            
        plt.text(
            1.02, 0.98, legend_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, fontfamily='monospace'
        )
        
        plt.tight_layout()
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to {savefile}")

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
                                matrix_format="DENSE", operation_type="SpMM")

        # 2. Set matrix dimensions (add after you know your matrix size)
        calc.set_matrix_dimensions(N=1024, K=1024, M=64, density=20)  # 80% sparse

        # 3. Calculate theoretical metrics (add before running kernel)
        flops = calc.calculate_flops()
        theoretical_accesses = calc.calculate_memory_accesses("dense")

        # 4. Replace your existing timing code with:
        calc.tic()
        # ... your kernel execution ...
        calc.toc()
        cycles_array = calc.copy_back_compute_time(symbol_time, width, height, data_type, layout)

        # 5. Calculate performance metrics (add after timing)
        metrics = calc.calculate_performance_metrics(cycles_array)

        # 6. Save results (add at the end of your benchmark)
        calc.save_roofline_csv("DENSE_benchmark.csv")
        calc.print_roofline_summary()
        calc.plot_roofline_from_csv("DENSE_benchmark.csv", savefile="DENSE_benchmark.png")
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
    width, height = 16, 16
    pe_length = 1
    loop_count = 1
    frequency = 0.85  # GHz  # 850MHz = 850000000 Hz = 1/second
    isCS2 = False  # True for CS2, False for CS3
    
    # Matrix dimensions for your benchmark
    M, K, N = 64, 64, 1 # 64x64(Matrix) * 64*1(Vector) = 64x1(Result)
    density = 100
    
    # Matrix format being tested
    matrix_format = "DENSE"  # or "COO", "CSC", "ELLPACK", "DENSE"
    operation_type = "GeMV"  # or "SpMM"
    
    print("="*80)
    print("ROOFLINE INTEGRATION EXAMPLE")
    print("="*80)
    
    # Step 1: Initialize RooflineCalculator
    print("Step 1: Initialize RooflineCalculator")    
    # For this example, we'll create a mock calculator
    calc = RooflineCalculator(None, width, height, pe_length, loop_count, frequency, isCS2,
                             matrix_format, operation_type)
    # Step 2: Set matrix dimensions
    print("Step 2: Set matrix dimensions")
    calc.set_matrix_dimensions(M, K, N, density)
    # Step 3: Calculate theoretical metrics
    print("Step 3: Calculate theoretical metrics")
    calc.calculate_theoretical_metrics()
    # Step 4: Empirical measurements
    print("Step 4: Calculate empirical metrics")
    # In your actual code, you would do:
    # calc.tic()
    # # ... run your kernel ...
    # calc.toc()
    # cycles_array = calc.copy_back_compute_time(symbol_time, width, height, data_type, layout)
    
    # For this example, simulate some cycle measurements
    # Todo: This is dummy data, not real data
    np.random.seed(42)  # For reproducible results
    base_cycles = 1500 if matrix_format == "DENSE" else 2000
    cycles_array = np.random.normal(base_cycles, base_cycles * 0.1, (height, width))

    # Step 5: Calculate empirical metrics
    calc.calculate_empirical_metrics(cycles_array)

    # Step 6: Save to CSV
    print("Step 6: Save to CSV")
    calc.save_roofline_csv(f"{matrix_format}_benchmark.csv")
    
    # Step 7: Print summary
    print("Step 7: Print summary")
    calc.print_roofline_summary()

    # Step 8: Plot empirical vs theoretical 
    print("Step 8: Plot empirical vs theoretical")
    calc.plot_empirical_vs_theoretical_cycles(f"{matrix_format}_benchmark.csv", savefile=f"{matrix_format}_benchmark.png")

    # # Step 8: Plot roofline
    # print("Step 8: Plot roofline")
    # calc.plot_roofline_from_csv(f"{matrix_format}_benchmark.csv", savefile=f"{matrix_format}_benchmark.png")
    # print(f"Roofline plot saved to: {matrix_format}_benchmark.png")
    
    return calc


if __name__ == "__main__":
    example_usage_function()
    # Integrated example
    calc = integrate_roofline_with_existing_benchmark()
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
