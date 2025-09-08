"""
Roofline Calculator for CSL Performance Analysis

This module extends the TimingCalculator to generate roofline plot data
by calculating FLOPS, memory accesses, and performance metrics.
"""

import numpy as np
import pandas as pd
import datetime
from typing import Dict, List, Tuple, Optional
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


def create_roofline_csv_from_multiple_runs(runs_data: List[Dict], output_filename: str):
    """
    Create a consolidated roofline CSV from multiple benchmark runs.
    
    Args:
        runs_data: List of dictionaries containing run data
        output_filename: Output CSV filename
    """
    df = pd.DataFrame(runs_data)
    df.to_csv(output_filename, index=False)
    print(f"Consolidated roofline data saved to: {output_filename}")
    return df


# Example usage functions
def example_spmm_benchmark():
    """
    Example of how to use RooflineCalculator for SpMM benchmarking.
    """
    # This would be called from your actual benchmark code
    print("Example SpMM Roofline Benchmark:")
    print("1. Initialize RooflineCalculator")
    print("2. Set matrix dimensions")
    print("3. Calculate FLOPS and memory accesses")
    print("4. Run timing measurements")
    print("5. Calculate performance metrics")
    print("6. Save to CSV")
    
    # Example code structure:
    """
    # In your benchmark script:
    calc = RooflineCalculator(runner, width, height, pe_length, loop_count, frequency, isCS2, 
                             matrix_format="CSR", operation_type="SpMM")
    
    # Set matrix dimensions
    calc.set_matrix_dimensions(N=768, K=768, M=64, density=20)  # 80% sparse
    
    # Calculate theoretical metrics
    flops = calc.calculate_flops()
    rel_acc, abs_acc = calc.calculate_memory_accesses("csr")
    
    # Run your timing code here...
    # cycles_array = your_timing_measurements()
    
    # Calculate performance
    # metrics = calc.calculate_performance_metrics(cycles_array)
    
    # Save results
    # calc.save_roofline_csv("CSR_benchmark.csv")
    # calc.print_roofline_summary()
    """


if __name__ == "__main__":
    example_spmm_benchmark()
