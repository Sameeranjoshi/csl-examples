#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./layout.csl --fabric-dims=18,5 \
--fabric-offsets=4,1 --params=kernel_x_dim:5,kernel_y_dim:1,M:4,N:4 \
-o out --memcpy --channels 1 --max-inlined-iterations=1000000
cs_python run.py --name out
