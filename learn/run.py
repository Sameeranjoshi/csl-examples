#!/usr/bin/env cs_python

import argparse
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder  # pylint: disable=no-name-in-module

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run a CSL program on Cerebras")
parser.add_argument('--name', required=True, help="The directory containing the compiled output ELF file")
parser.add_argument('--cmaddr', help="IP:port for the CS system (optional if using simulator)")
args = parser.parse_args()

# host data 
# Initialize data to send to the device
WIDTH = 5
HEIGHT = 1

A = np.array([10, 20, 30, 40, 50], dtype=np.int32)
y_expected = np.array([11, 21, 31, 41, 51], dtype=np.int32)



try:
    # Initialize the SdkRuntime runner
    runner = SdkRuntime(args.name, cmaddr=args.cmaddr, supress_simfab_trace=False, msg_level="WARNING")


    # Get symbols for A, x, b, y on device
    A_symbol = runner.get_id('A')
    y_symbol = runner.get_id('y')

    # Load and execute the program
    print("Loading program...")
    runner.load()

    print("Starting program...")
    runner.run()

    # Copy data to the device
    print("Copying data to the device...")
    # Copy A, x, b to device
    # runner.memcpy_h2d(dest=A_symbol, src=A, px=0, py=0, w=WIDTH, h=HEIGHT, elem_per_pe=5, streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    try:
        runner.memcpy_h2d(
            A_symbol,  # dest
            A,         # src
            0,         # px
            0,         # py
            1,     # w
            1,    # h
            5,         # elem_per_pe
            streaming=False,
            order=MemcpyOrder.ROW_MAJOR,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            nonblock=False
        )
    except TypeError as e:
        print(f"Error during memcpy_h2d: {e}")

    # Launch the main function on the device
    print("Launching main function...")
    runner.launch('main', nonblock=False)

    print("Program executed successfully!")
    
    # copy y from device to host
    print("Copying y from device to host...")
    y_result = np.zeros(5, dtype=np.int32)
    
    try:
        runner.memcpy_d2h(
            y_result,  # dest
            y_symbol,  # src
            0,         # px
            0,         # py
            1,         # w
            1,         # h
            1,         # elem_per_pe
            streaming=False,
            order=MemcpyOrder.ROW_MAJOR,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            nonblock=False
        )
    except TypeError as e:
        print(f"Error during memcpy_d2h: {e}")
finally:
    # Ensure the runner stops cleanly
    print("Stopping the program...")
    runner.stop()
    
    np.testing.assert_array_equal(y_expected, y_result)
    print("Success!")
