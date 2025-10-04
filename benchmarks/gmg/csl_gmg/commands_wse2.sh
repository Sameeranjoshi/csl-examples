#!/usr/bin/env bash

# set -e

# cslc ./src/layout_gmg.csl --arch wse2 --fabric-dims=12,7 --fabric-offsets=4,1 \
# --params=width:5,height:5,MAX_ZDIM:5 --params=BLOCK_SIZE:5 \
# --params=C0_ID:0 --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 \
# --params=C4_ID:4 --params=C5_ID:5 --params=C6_ID:6 --params=C7_ID:7 \
# --params=C8_ID:8 -o=out --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000

# # Compile and run GMG on WSE2
# cs_python ./run_gmg.py -m=5 -n=5 -k=5 --zDim=5 --latestlink out --channels=1 --width-west-buf=0 \
# --width-east-buf=0 --run-only --max-ite=10



# cslc ./src/layout_gmg.csl --arch wse2 --fabric-dims=25,25 --fabric-offsets=4,1 \
# --params=width:16,height:16,MAX_ZDIM:16 --params=BLOCK_SIZE:16 --params=C0_ID:0 \
# --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
# --params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out \
# --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000
# cs_python ./run_gmg.py -m=16 -n=16 -k=16 --latestlink out --channels=1 \
# --width-west-buf=0 --width-east-buf=0 --zDim=16 --run-only --levels=4


cslc ./src/layout_gmg.csl --arch wse3 --fabric-dims=24,18 --fabric-offsets=4,1 \
--params=width:16,height:16,MAX_ZDIM:16 --params=BLOCK_SIZE:16 --params=C0_ID:0 \
--params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
--params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000
cs_python ./run_gmg.py -m=16 -n=16 -k=16 --latestlink out --channels=1 \
--width-west-buf=0 --width-east-buf=0 --zDim=16 --run-only --levels=5