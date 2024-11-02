#!/usr/bin/env cs_python

import argparse
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder  # pylint: disable=no-name-in-module

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run a CSL program on Cerebras")
parser.add_argument('--name', required=True, help="The directory containing the compiled output ELF file")
parser.add_argument('--cmaddr', help="IP:port for the CS system (optional if using simulator)")
args = parser.parse_args()

try:
    # Initialize the SdkRuntime runner
    runner = SdkRuntime(args.name, cmaddr=args.cmaddr, supress_simfab_trace=False, msg_level="WARNING")

    # Load and execute the program
    print("Loading program...")
    runner.load()

    print("Starting program...")
    runner.run()

    # Launch the main function on the device
    print("Launching main function...")
    runner.launch('main', nonblock=False)

    print("Program executed successfully!")

finally:
    # Ensure the runner stops cleanly
    print("Stopping the program...")
    runner.stop()
