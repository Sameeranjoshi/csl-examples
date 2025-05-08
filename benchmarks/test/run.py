
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
runner.launch("user", nonblock=False)
runner.stop()
