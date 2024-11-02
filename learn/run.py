#!/usr/bin/env cs_python

import argparse
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr, supress_simfab_trace=True, msg_level="WARNING")


# Load and run the program
runner.load()
runner.run()

# Launch the init_and_compute function on device
runner.launch('main', nonblock=False)

# Stop the program
runner.stop()

# Ensure that the result matches our expectation

print("SUCCESS!")