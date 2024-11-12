#!/usr/bin/env bash

set -e  # Exit on first error

echo "Compiling layout.csl to create out ELF file..."
cslc layout.csl --fabric-dims=9,3 --fabric-offsets=4,1 --params=M:4,N:6,width:2 --memcpy --channels=1 --verbose -o out

echo "Running the host-device interaction through run.py..."
cs_python run.py --name out/

echo "Execution complete."
