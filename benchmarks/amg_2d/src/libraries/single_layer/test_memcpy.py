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
from jacobi_only import pad_A, pad_1d, row_row_layout, row_col_layout, col_row_layout, col_col_layout, hwl_2_oned_colmajor

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime     # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder    # pylint: disable=no-name-in-module
  

 
parser = argparse.ArgumentParser()
parser.add_argument("--M", help="the test name")
parser.add_argument("--N", help="the test name")
parser.add_argument("--kernel_rows", help="the test name")
parser.add_argument("--kernel_cols", help="the test name")
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()


# Sample matrix A
M = int(args.M)
N = int(args.N)  # Matrix dimensions
kernel_rows = int(args.kernel_rows)
kernel_cols = int(args.kernel_cols)

A = np.arange(M*N, dtype=np.float32).reshape(M,N).astype(np.float32)
X = np.arange(N, dtype=np.float32).reshape(N,1).astype(np.float32)
B = np.arange(M, dtype=np.float32).reshape(M,1).astype(np.float32)

A_padded = pad_A(A, kernel_rows, kernel_cols)
[padded_M, padded_N] = A_padded.shape
X_padded = pad_1d(X, padded_N)
B_padded = pad_1d(B, padded_M)

# per PE - floor division operator (//) rounds down to nearest integer
per_pe_M = padded_M // kernel_rows  # Floor division - rounds down padded matrix rows / kernel rows
per_pe_N = padded_N // kernel_cols  # Floor division - rounds down padded matrix cols / kernel cols

# Test all layouts
print("kernel_rows:", kernel_rows, "kernel_cols:", kernel_cols)
print("Original A shape:", A.shape, "--> Padded A shape:", A_padded.shape, "per PE shape:", (per_pe_M, per_pe_N))
print("Original X shape:", X.shape, "--> Padded X shape:", X_padded.shape)
print("Original B shape:", B.shape, "--> Padded B shape:", B_padded.shape)

# layouts
print(f"A_row_row = {A_padded}")
A_transposed = A_padded.T.flatten()
print(f"A_row_row_2 = {A_transposed}")


# A_row_col = row_col_layout(A_padded, kernel_rows, kernel_cols)
# A_col_row = col_row_layout(A_padded, kernel_rows, kernel_cols)
# A_col_col = col_col_layout(A_padded, kernel_rows, kernel_cols)

print("\nRow-Row Layout (Row-major blocks, Row-major inside blocks):")
print(A_row_row)

# print("\nRow-Col Layout (Row-major blocks, Column-major inside blocks):")
# print(A_row_col)

# print("\nCol-Row Layout (Column-major blocks, Row-major inside blocks):")
# print(A_col_row)

# print("\nCol-Col Layout (Column-major blocks, Column-major inside blocks):")
# print(A_col_col)



# DEVICE
start_time = time.time()
# Specify path to ELF files, set up runner
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT


symbol_A = runner.get_id("A")


runner.load()
runner.run()

print("Copying data...")
# Compute number of rows and cols of A on each PE
per_pe_rows = per_pe_M
per_pe_cols = per_pe_N

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
# data = np.stack(np.split(np.stack(np.split(A, kernel_cols, axis=1)), kernel_rows, axis=1)).ravel()
# data_R = np.stack(np.split(np.stack(np.split(R, kernel_cols, axis=1)), kernel_rows, axis=1)).ravel()
# Copy flattened A array onto PEs
runner.memcpy_h2d(symbol_A, A_transposed, 0, 0, kernel_cols, kernel_rows, per_pe_rows * per_pe_cols,
                  streaming=False, data_type=memcpy_dtype, nonblock=False,
                  order=MemcpyOrder.COL_MAJOR)

print("Launching kernel...")
# Record start time


# Run the kernel
runner.launch("layout_print", nonblock=False)

runner.stop()
# Record end time and calculate duration
end_time = time.time()
duration = end_time - start_time
print(f"Kernel execution time: {duration:.3f} seconds")

