#!/usr/bin/env cs_python

import numpy as np
import argparse
import json

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder, Task

# read out elf file

parser = argparse.ArgumentParser(description="Run a CSL program on Cerebras", add_help=True)
parser.add_argument('--name', required=True, help="The directory containing the compiled output ELF file")
parser.add_argument('--cmaddr', help="IP:port for the CS system (optional if using simulator)")
args = parser.parse_args()


# Params are stored in a file called out/out.json, read them dynamically from it.
with open('out/out.json') as f:
    params = json.load(f)

params = params['params']

# debug print
print(f"--name= {args.name}")
print(f"params: {params}")

M = int(params['M'])
N = int(params['N'])
width = int(params['width'])

# host A, x, b, y
host_A = np.arange(M*N, dtype=np.float32)
host_x = np.full(shape=N, fill_value=2.0, dtype=np.float32)
host_b = np.full(shape=M, fill_value=2.0, dtype=np.float32)



# Initialize the SdkRuntime runner
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

device_data_y = runner.get_id('y_advertised')
device_data_A = runner.get_id('A_advertised')
device_data_x = runner.get_id('x_advertised')
device_data_b = runner.get_id('b_advertised')


runner.load()
runner.run()
#copy from host to device for A
runner.memcpy_h2d(device_data_A, np.tile(host_A, width), 0, 0, width, 1, M*N, streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
#copy from host to device for x
runner.memcpy_h2d(device_data_x, np.tile(host_x, width), 0, 0, width, 1, N, streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
#copy from host to device for b
runner.memcpy_h2d(device_data_b, np.tile(host_b, width), 0, 0, width, 1, M, streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.launch('init_and_compute_advertised', nonblock=False)

y_returned = np.zeros([M*width], dtype=np.float32)
runner.memcpy_d2h(y_returned, device_data_y, 0, 0, width, 1, M, streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.stop()

host_y_expected = host_A.reshape(M,N)@host_x + host_b
print(f"host_y_expected: {host_y_expected}")
print(f"y_returned: {y_returned}")
np.testing.assert_allclose(y_returned, np.tile(host_y_expected, width), atol=0.01, rtol=0)
print("Success!")
