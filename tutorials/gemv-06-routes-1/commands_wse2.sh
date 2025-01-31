#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./layout.csl --fabric-dims=11,3 \
--fabric-offsets=4,1 --params=M:4,N:6 -o out --memcpy --channels 1
cs_python run.py --cmaddr=140.221.80.27:60000 --name out
