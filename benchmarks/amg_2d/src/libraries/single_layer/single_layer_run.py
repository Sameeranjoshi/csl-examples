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
from jacobi_only import jacobi_iteration, jacobi_iteration_alt

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime     # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder    # pylint: disable=no-name-in-module
  
parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()

# Get params from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)
# Kernel rectangle and matrix dimensions from compile parameters
kernel_rows = int(compile_data['params']['layer_kernel_rows'])
kernel_cols = int(compile_data['params']['layer_kernel_cols'])
matrix_rows = int(compile_data['params']['layer_rows_A']) 
matrix_cols = int(compile_data['params']['layer_cols_A'])
restrict_rows = int(compile_data['params']['layer_rows_R'])
restrict_cols = int(compile_data['params']['layer_cols_R'])

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
assert restrict_cols == matrix_rows, "restrict_cols must be equal to matrix_rows, example:(4,49) x (49,1) = (4,1)"


# Compute expected result
x_copy = X.copy()
omega_host = 1.0/3.0
iterations_host = 5
X_smooth_host = jacobi_iteration(A, B, x_copy, omega_host, iterations_host)
residual = B - (A @ X_smooth_host)
print("residual:\n", residual)
# Reshape residual to be a column vector for matrix multiplication
b_next_host = R @ residual


# DEVICE
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
symbol_x_smooth = runner.get_id("x_smooth")
symbol_x_smooth_src = runner.get_id("x_smooth_src")
symbol_omega = runner.get_id("omega")
symbol_iterations = runner.get_id("iterations")

runner.load()
runner.run()

print("Copying data...")
# Compute number of rows and cols of A on each PE
per_pe_rows = matrix_rows // kernel_rows
per_pe_cols = matrix_cols // kernel_cols

per_pe_restrict_rows = restrict_rows // kernel_rows
per_pe_restrict_cols = restrict_cols // kernel_cols

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
data = np.stack(np.split(np.stack(np.split(A, kernel_cols, axis=1)), kernel_rows, axis=1)).ravel()
data_R = np.stack(np.split(np.stack(np.split(R, kernel_cols, axis=1)), kernel_rows, axis=1)).ravel()
# Copy flattened A array onto PEs
runner.memcpy_h2d(symbol_A, data, 0, 0, kernel_cols, kernel_rows, per_pe_rows * per_pe_cols,
                  streaming=False, data_type=memcpy_dtype, nonblock=False,
                  order=memcpy_order)
runner.memcpy_h2d(symbol_R, data_R, 0, 0, kernel_cols, kernel_rows, per_pe_restrict_rows * per_pe_restrict_cols,
                  streaming=False, data_type=memcpy_dtype, nonblock=False,
                  order=memcpy_order)

# x is across rows and b is across cols
runner.memcpy_h2d_colbcast(symbol_x, X, 0, 0, kernel_cols, kernel_rows, per_pe_cols,
                  streaming=False, data_type=memcpy_dtype, nonblock=False, order=memcpy_order)

# B will be distributed but the diagonal will only process the first term.
runner.memcpy_h2d_rowbcast(symbol_b, B, 0, 0, kernel_cols, kernel_rows, per_pe_rows,
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

print("Launching kernel...")
# Record start time


# Run the kernel
runner.launch("main", nonblock=False)

# TOPRIGHT PE
b_next_device = np.zeros(restrict_rows, dtype=np.float32)
runner.memcpy_d2h(b_next_device, symbol_b_next, kernel_cols-1, 0, 1, kernel_rows, per_pe_restrict_rows,
                  streaming=False, data_type=memcpy_dtype, nonblock=False,
                  order=memcpy_order)

runner.stop()
# Record end time and calculate duration
end_time = time.time()
duration = end_time - start_time
print(f"Kernel execution time: {duration:.3f} seconds")

print("Copied back result.")

print("b_next_host calculated: ", b_next_host)
print("b_next_device calculated: ", b_next_device)
np.testing.assert_allclose(b_next_host, b_next_device, atol=0.0, rtol=1e-4)
print("SUCCESS")
