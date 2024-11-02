#!/usr/bin/env bash

set -e

# creates a .out elf file which can be examined using cs_readelf
cslc --arch=wse2 ./layout.csl --fabric-dims=8,8 --fabric-offsets=4,1 --memcpy --channel=1 -o out
# RUn the host->device interaction
cs_python run.py --name out
    