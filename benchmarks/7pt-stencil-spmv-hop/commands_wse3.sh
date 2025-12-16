#!/usr/bin/env bash

set -e

cslc ./src/layout.csl --arch wse3 --fabric-dims=74,66 --fabric-offsets=4,1 \
--params=width:64,height:64,MAX_ZDIM:64 --params=BLOCK_SIZE:64 --params=C0_ID:0 \
--params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
--params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out --max-inlined-iterations=1000000 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python ./run.py -m=64 -n=64 -k=64 --level-id=1 --latestlink out --channels=1 \
--width-west-buf=0 --width-east-buf=0 --zDim=64 --run-only
