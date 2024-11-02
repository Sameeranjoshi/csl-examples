#!/usr/bin/env cs_python

import numpy as np
import argparse

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder, Task

# read out elf file

parser = argparse.ArgumentParser(description="Run a CSL program on Cerebras", add_help=True)
parser.add_argument('--name', required=True, help="The directory containing the compiled output ELF file")
parser.add_argument('--cmaddr', help="IP:port for the CS system (optional if using simulator)")
args = parser.parse_args()

print(f"--name= {args.name}")

M = 4
N = 6
# host A, x, b, y
host_A = np.arange(M*N, dtype=np.float32).reshape(M, N)
host_x = np.full(shape=N, fill_value=1.0, dtype=np.float32)
host_b = np.full(shape=M, fill_value=2.0, dtype=np.float32)
host_y_expected = np.dot(host_A, host_x) + host_b


# Initialize the SdkRuntime runner
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

device_data_y = runner.get_id('y_advertised')

runner.load()
runner.run()
runner.launch('init_and_compute_advertised', nonblock=False)

y_returned = np.zeros([1*1*M], dtype=np.float32)
runner.memcpy_d2h(y_returned, device_data_y, 0, 0, 1, 1, M, streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.stop()

print(f"host_y_expected: {host_y_expected}")
print(f"y_returned: {y_returned}")
np.testing.assert_allclose(host_y_expected, y_returned, rtol=0.01, atol=0)
print("Success!")
