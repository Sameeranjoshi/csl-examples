#!/usr/bin/env cs_python

# Copyright 2024 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import time
import numpy as np
from jacobi_only import jacobi_iteration, jacobi_iteration_alt, unpad_1d
import time_utils as time_ut
from util import (
    hwl_2_oned_colmajor,
    oned_to_hwl_colmajor,
    laplacian,
    csr_7_pt_stencil,
)
from jacobi_only import pad_A, pad_1d

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime     # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder    # pylint: disable=no-name-in-module

def time_logs(h, w, time_memcpy_hwl, start_time, end_time, filename="timing_data.csv", folder={}):
    cpu_time = end_time - start_time    # measures the (memcpy + kernel time + sdkruntime setup) using CPU time
    perf_metrics = {}
    folder_name = folder.split('/')[-1] if '/' in folder else folder
    problem_size = '_'.join(folder_name.split('_')[1:5])  # Get A16_16_R16_16
    pe_size = '_'.join(folder_name.split('_')[5:])       # Get PE2_2
    perf_metrics['input'] = problem_size
    perf_metrics['PE'] = pe_size
    perf_metrics['cpu_time_seconds'] = cpu_time
    
    # Get timing data for this layer
    timing_data_per_layer = time_ut.time_analysis_noref(h, w, time_memcpy_hwl)
    perf_metrics['cycles'] = timing_data_per_layer['cycles']
    perf_metrics['time_us'] = timing_data_per_layer['time_us']
    df = time_ut.write_performance_data(perf_metrics, filename=filename)
    return df

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
parser.add_argument("--A_rows", help="the number of rows of A")
parser.add_argument("--A_cols", help="the number of cols of A")
parser.add_argument("--R_rows", help="the number of rows of R")
parser.add_argument("--R_cols", help="the number of cols of R")
args = parser.parse_args()

# Get params from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)
# Kernel rectangle and matrix dimensions from compile parameters
kernel_rows = int(compile_data['params']['layer_kernel_rows'])
kernel_cols = int(compile_data['params']['layer_kernel_cols'])
# TODO: Use .json file as params in run.py, for now use arguments from .py.
# matrix_rows = int(compile_data['params']['layer_rows_A']) 
matrix_rows = int(args.A_rows)
# matrix_cols = int(compile_data['params']['layer_cols_A'])
matrix_cols = int(args.A_cols)
# restrict_rows = int(compile_data['params']['layer_rows_R'])
restrict_rows = int(args.R_rows)
# restrict_cols = int(compile_data['params']['layer_cols_R'])
restrict_cols = int(args.R_cols)
# print all the params
print(f"matrix_rows: {matrix_rows}, matrix_cols: {matrix_cols}, restrict_rows: {restrict_rows}, restrict_cols: {restrict_cols}, kernel_rows: {kernel_rows}, kernel_cols: {kernel_cols}")
# Create tensors for A, X, B.
# Use a deterministic seed so that CI results are predictable
np.random.seed(seed=7)

# A = np.array([[4, 1,  3,  3],
#               [4, 1, 0, 1],
#               [2, 2, 0, 4],
#               [0, 4, 0, 3]], dtype=np.float32)
# B = np.array([0,0,0,3], dtype=np.float32)
# X = np.array([2,3,4,4], dtype=np.float32)

A = np.random.rand(matrix_rows, matrix_cols).astype(np.float32)
X = np.random.rand(matrix_cols).astype(np.float32)
B = np.random.rand(matrix_rows).astype(np.float32)
R = np.random.rand(restrict_rows, restrict_cols).astype(np.float32)

# check if R_rows = layer_rows_A
# Add more.
assert restrict_cols == matrix_rows, "restrict_cols must be equal to matrix_rows, example:(R_Coarse,R_Fine) x (R_Fine,..) = (R_Coarse,..)"
assert matrix_rows == matrix_cols, "matrix should be square"    # This is strictly required for Ax=b.
# assert kernel_rows == kernel_cols, "kernel should be square(Limitation of current padding implementation)"

# Delete the variables to prevent accidental use
# del matrix_rows, matrix_cols, restrict_rows, restrict_cols


# Compute expected result
x_copy = X.copy()
omega_host = 1.0/3.0
iterations_host = 5
X_smooth_host = jacobi_iteration(A, B, x_copy, omega_host, iterations_host)
residual = B - (A @ X_smooth_host)
# print("residual:\n", residual)
# Reshape residual to be a column vector for matrix multiplication
b_next_host = R @ residual


# DEVICE
# Padding

start_time = time.time()
# Specify path to ELF files, set up runner
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

symbol_A = runner.get_id("A")
symbol_R = runner.get_id("R")
symbol_x = runner.get_id("x")
symbol_b = runner.get_id("b")
symbol_b_next = runner.get_id("b_next") 
symbol_omega = runner.get_id("omega")
symbol_iterations = runner.get_id("iterations")
symbol_time_memcpy = runner.get_id("time_memcpy")
symbol_time_ref = runner.get_id("time_ref")

runner.load()
runner.run()

print("Copying data...")
# Compute number of rows and cols of A on each PE
print("Padded the input matrices")
A_padded = pad_A(A, kernel_rows, kernel_cols, variable_name="A")
[padded_matrix_rows, padded_matrix_cols] = A_padded.shape
X_padded = pad_1d(X, padded_matrix_cols, variable_name="X")
B_padded = pad_1d(B, padded_matrix_rows, variable_name="B")
R_padded = pad_A(R, kernel_rows, kernel_cols, variable_name="R")
[padded_restrict_rows, padded_restrict_cols] = R_padded.shape

# # per PE - floor division operator (//) rounds down to nearest integer
per_pe_rows = padded_matrix_rows // kernel_rows  # Floor division - rounds down padded matrix rows / kernel rows
per_pe_cols = padded_matrix_cols // kernel_cols  # Floor division - rounds down padded matrix cols / kernel cols
per_pe_restrict_rows = padded_restrict_rows // kernel_rows
per_pe_restrict_cols = padded_restrict_cols // kernel_cols


# # HOST
# # Compute expected result
# x_copy = X_padded.copy()
# A_copy = A_padded.copy()
# R_copy = R_padded.copy()
# B_copy = B_padded.copy()
# omega_host = 1.0/3.0
# iterations_host = 5
# X_smooth_host = jacobi_iteration_alt(A_copy, B_copy, x_copy, omega_host, iterations_host)
# residual = B_copy - (A_copy @ X_smooth_host)
# print(f"R_copy shape: {R_copy.shape}")
# print(f"residual shape: {residual.shape}")
# b_next_host = R_copy @ residual

# This transformation on A creates a flattened array so that the matrix can
# be mapped onto the PEs using MemcpyOrder.ROW_MAJOR copy ordering.
# The arrays holding A on each PE are 1D arrays that store the submatrices
# in row major order.

# As an example, consider A[4, 4], mapped onto a 2x2 grid of PEs:
#
#   Matrix A on host            2 x 2 PE grid, row major A submatrices
# +----+----+----+----+         +----------------+----------------+
# | 0  | 1  | 2  | 3  |         | PE (0, 0):     | PE (1, 0):     |
# +----+----+----+----+         |  0,  1,  4,  5 |  2,  3,  6,  7 |
# | 4  | 5  | 6  | 7  |         |                |                |
# +----+----+----+----+   --->  +----------------+----------------+
# | 8  | 9  | 10 | 11 |         | PE (0, 1):     | PE (1, 1):     |
# +----+----+----+----+         |  8,  9, 12, 13 | 10, 11, 14, 15 |
# | 12 | 13 | 14 | 15 |         |                |                |
# +----+----+----+----+         +----------------+----------------+
#
# MemcpyOrder.ROW_MAJOR copy ordering maps an input array to dimensions h x w x l,
# with l varying fastest, where:
#   - h is kernel_rows (i.e., height of program rectangle)
#   - w is kernel_cols (i.e., width of program rectangle)
#   - l is per_pe_rows * per_pe_cols (i.e., num elems copied to each PE)
#
# So our input array for memcpy_h2d must be ordered as follows:
# [ 0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15 ]

# The transformation here takes the matrix A and:
#   1. splits A into kernel_cols submatrices, along the vertical axis
#   2. stacks them into a 3D array, with kernel_col as 0th dimension
#   3. splits 3D array into kernel_rows subarrays, along the horizontal axis
#   4. stacks them into a 4D array, with kernel_row as 0th dimension
#   5. flattens into a 1D array
data = np.stack(np.split(np.stack(np.split(A_padded, kernel_cols, axis=1)), kernel_rows, axis=1)).ravel()
data_R = np.stack(np.split(np.stack(np.split(R_padded, kernel_cols, axis=1)), kernel_rows, axis=1)).ravel()
# Copy flattened A array onto PEs
runner.memcpy_h2d(symbol_A, data, 0, 0, kernel_cols, kernel_rows, per_pe_rows * per_pe_cols,
                  streaming=False, data_type=memcpy_dtype, nonblock=False,
                  order=memcpy_order)
runner.memcpy_h2d(symbol_R, data_R, 0, 0, kernel_cols, kernel_rows, per_pe_restrict_rows * per_pe_restrict_cols,
                  streaming=False, data_type=memcpy_dtype, nonblock=False,
                  order=memcpy_order)

# x is across rows and b is across cols
runner.memcpy_h2d_colbcast(symbol_x, X_padded, 0, 0, kernel_cols, kernel_rows, per_pe_cols,
                  streaming=False, data_type=memcpy_dtype, nonblock=False, order=memcpy_order)

# B will be distributed but the diagonal will only process the first term.
runner.memcpy_h2d_rowbcast(symbol_b, B_padded, 0, 0, kernel_cols, kernel_rows, per_pe_rows,
                  streaming=False, data_type=memcpy_dtype, nonblock=False, order=memcpy_order)

# omega
# make the size of omega equal to the number of PEs.
omega = np.zeros(kernel_rows * kernel_cols, dtype=np.float32)
omega[:] = omega_host
runner.memcpy_h2d(symbol_omega, omega, 0, 0, kernel_cols, kernel_rows, 1,
                            streaming=False, data_type=memcpy_dtype, nonblock=False, order=memcpy_order)

# iterations
iterations = np.zeros(kernel_rows * kernel_cols, dtype=np.int32)
iterations[:] = iterations_host
runner.memcpy_h2d(symbol_iterations, iterations, 0, 0, kernel_cols, kernel_rows, 1,
                            streaming=False, data_type=memcpy_dtype, nonblock=False, order=memcpy_order)

print("Initializing hardware timing...")
print("Step 1: Enable timer")
runner.launch("f_enable_timer", nonblock=False)
# print("Step 2: sync")
# runner.launch("f_sync", nonblock=False)
print("Step 3: Record initial timestamp (tic)")
runner.launch("f_tic", nonblock=True)
print("Step 4: Launching kernel...")
# runner.launch("layout_print", nonblock=False)
runner.launch("main", nonblock=False) # Run the kernel
print("Step 5: toc() records time_end")
runner.launch("f_toc", nonblock=False)
print("Step 6: prepare (time_start, time_end)")
runner.launch("f_memcpy_timestamps", nonblock=False)
print("Step 7: prepare reference clock")
runner.call("f_reference_timestamps", [], nonblock=False)
print("Step 8: Retrieve timing data")
    
time_memcpy_1d_f32 = np.zeros(kernel_rows*kernel_cols*3, np.float32)
runner.memcpy_d2h(time_memcpy_1d_f32, symbol_time_memcpy, 0, 0, kernel_cols, kernel_rows, 3,
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (kernel_rows, kernel_cols, 3), order='C')
# time_ref is of type u16[3], packed into two f32
time_ref_1d_f32 = np.zeros(kernel_rows*kernel_cols*2, np.float32)
runner.memcpy_d2h(time_ref_1d_f32, symbol_time_ref, 0, 0, kernel_cols, kernel_rows, 2,
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
time_ref_hwl = np.reshape(time_ref_1d_f32, (kernel_rows, kernel_cols, 2), order='C')
# print("time_ref_hwl:\n", time_ref_hwl)

# RIGHTMOST COLUMN
print("Step 9: Copied back result.")
b_next_device = np.zeros(padded_restrict_rows, dtype=np.float32)
runner.memcpy_d2h(b_next_device, symbol_b_next, kernel_cols-1, 0, 1, kernel_rows, per_pe_restrict_rows,
                  streaming=False, data_type=memcpy_dtype, nonblock=False,
                  order=memcpy_order)
b_next_device = unpad_1d(b_next_device, restrict_rows)
# print("b_next_device shape:", b_next_device.shape)
runner.stop()

# Record end time and calculate duration
end_time = time.time()
print("Step 10: Time logs")
time_logs(kernel_rows, kernel_cols, time_memcpy_hwl, start_time, end_time, filename=f"./single_layer_timing_runs.csv", folder=args.name)

# print("b_next_host calculated: ", b_next_host)
# print("b_next_device calculated: ", b_next_device)
np.testing.assert_allclose(b_next_host, b_next_device, atol=0.0, rtol=1e-4)
print("SUCCESS")
