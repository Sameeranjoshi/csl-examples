#!/usr/bin/env bash

set -e

cslc ./src/layout.csl --arch wse3 --fabric-dims=24,18 --fabric-offsets=4,1 \
--params=width:16,height:16,MAX_ZDIM:16 --params=BLOCK_SIZE:16 --params=C0_ID:0 \
--params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
--params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out --max-inlined-iterations=1000000 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python ./run.py -m=16 -n=16 -k=16 --level-id=3 --latestlink out --channels=1 \
--width-west-buf=0 --width-east-buf=0 --zDim=16 --run-only
