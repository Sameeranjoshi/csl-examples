
import argparse

import numpy as np
import pyamg
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder  # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument('--cmaddr', help='IP:port for CS system')
args = parser.parse_args()
name = args.name

# Simulate ELF files
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
np.random.seed(42)
runner.load()
runner.run()

total_levels_count = 3
layers_data = [
    {"M": 5, "layer_index": 0},
    {"M": 4, "layer_index": 1}, 
    {"M": 3, "layer_index": 2}
]

# fill data in A with shapes from layers_data such that for each layer, the data is filled with the shape of the layer.
A = []
for i in range(total_levels_count):
    A.append(np.random.rand(layers_data[i]["M"]).astype(np.float32))

# print array data
for i in range(total_levels_count):
    print(f"Layer {i}: ", A[i])
    
# transform A which is jagged array into a contiguous array
A_blob = np.concatenate(A)
print("Contiguous Array: ", A_blob)

symbols = {
    'A_blob': runner.get_id("A_blob"),
}

runner.launch("before", nonblock=False)
runner.memcpy_h2d(symbols['A_blob'], A_blob, 0, 0, 1, 1, A_blob.size, streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.launch("before", nonblock=False)
runner.launch("init_data_blob", nonblock=False)
runner.launch("print_data_blob", nonblock=False)
runner.stop()
