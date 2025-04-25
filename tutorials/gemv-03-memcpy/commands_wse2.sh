#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./layout.csl --fabric-dims=11,6 \
--fabric-offsets=4,1 -o out --memcpy --channels 1 --max-inlined-iterations=1000000
cs_python run.py --name out
