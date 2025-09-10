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
import numpy as np
import sys
import os
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', "..", "timing_library", "python_libs", "roofline_lib"))
from roofline_calculator import RooflineCalculator

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

# Matrix dimensions
M = 4
N = 6
height = 1
width = 1

# Construct A, x, 
A = np.arange(M*N, dtype=np.float32).reshape(M, N)
x = np.full(shape=N, fill_value=1.0, dtype=np.float32)
b = np.full(shape=M, fill_value=2.0, dtype=np.float32)

# Calculate expected y
y_expected = A@x + b

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
calc = RooflineCalculator(runner, width=width, height=height, pe_length=1, loop_count=1, frequency=850000000, isCS2=True,
                             matrix_format="DENSE", operation_type="GeMV")
calc.set_matrix_dimensions(M=M, K=N, N=1, density=1.0)
calc.calculate_theoretical_metrics()               
# Get symbol for copying y result off device
y_symbol = runner.get_id('y')
symbol_maxmin_time = runner.get_id("maxmin_time")

# Load and run the program
runner.load()
runner.run()

# Launch the init_and_compute function on device
runner.launch('init_and_compute', nonblock=False)

# Copy y back from device
# Arguments to memcpy_d2h:
# - y_result is array on host which will story copied-back array
# - y_symbol is symbol of device tensor to be copied
# - 0, 0, 1, 1 are (starting x-coord, starting y-coord, width, height)
#   of rectangle of PEs whose data is to be copied
# - M is number of elements to be copied from each PE
y_result = np.zeros([1*1*M], dtype=np.float32)
runner.memcpy_d2h(y_result, y_symbol, 0, 0, 1, 1, M, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
  
u48cycles_array = calc.copy_back_compute_time(symbol_maxmin_time, width, height, MemcpyDataType.MEMCPY_32BIT, MemcpyOrder.ROW_MAJOR)
print(f"maxmin_time_hwl = {u48cycles_array}")

# Stop the program
runner.stop()

# Ensure that the result matches our expectation
np.testing.assert_allclose(y_result, y_expected, atol=0.01, rtol=0)
print("SUCCESS!")
