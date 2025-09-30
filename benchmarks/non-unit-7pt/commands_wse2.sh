#!/usr/bin/env bash

set -e

# Test cases for 8x8x8 with BLOCK_SIZE=2,4,8
# cslc ./src/layout.csl --arch wse2 --fabric-dims=15,14 --fabric-offsets=4,1 \
# --params=width:8,height:8,MAX_ZDIM:8 --params=BLOCK_SIZE:2 --params=C0_ID:0 \
# --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
# --params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out_8x8x8_block2 \
# --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000
# cs_python ./run.py -m=8 -n=8 -k=8 --latestlink out_8x8x8_block2 --channels=1 \
# --width-west-buf=0 --width-east-buf=0 --zDim=8 --run-only

# cslc ./src/layout.csl --arch wse2 --fabric-dims=15,14 --fabric-offsets=4,1 \
# --params=width:8,height:8,MAX_ZDIM:8 --params=BLOCK_SIZE:4 --params=C0_ID:0 \
# --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
# --params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out_8x8x8_block4 \
# --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000
# cs_python ./run.py -m=8 -n=8 -k=8 --latestlink out_8x8x8_block4 --channels=1 \
# --width-west-buf=0 --width-east-buf=0 --zDim=8 --run-only

cslc ./src/layout.csl --arch wse2 --fabric-dims=15,14 --fabric-offsets=4,1 \
--params=width:8,height:8,MAX_ZDIM:8 --params=BLOCK_SIZE:8 --params=C0_ID:0 \
--params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
--params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out_8x8x8_block8 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000
cs_python ./run.py -m=8 -n=8 -k=8 --latestlink out_8x8x8_block8 --channels=1 \
--width-west-buf=0 --width-east-buf=0 --zDim=8 --run-only

# # Test cases for 4x4x4 with BLOCK_SIZE=2,4,8
# cslc ./src/layout.csl --arch wse2 --fabric-dims=11,10 --fabric-offsets=4,1 \
# --params=width:4,height:4,MAX_ZDIM:4 --params=BLOCK_SIZE:2 --params=C0_ID:0 \
# --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
# --params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out_4x4x4_block2 \
# --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000
# cs_python ./run.py -m=4 -n=4 -k=4 --latestlink out_4x4x4_block2 --channels=1 \
# --width-west-buf=0 --width-east-buf=0 --zDim=4 --run-only

# cslc ./src/layout.csl --arch wse2 --fabric-dims=11,10 --fabric-offsets=4,1 \
# --params=width:4,height:4,MAX_ZDIM:4 --params=BLOCK_SIZE:4 --params=C0_ID:0 \
# --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
# --params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out_4x4x4_block4 \
# --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000
# cs_python ./run.py -m=4 -n=4 -k=4 --latestlink out_4x4x4_block4 --channels=1 \
# --width-west-buf=0 --width-east-buf=0 --zDim=4 --run-only

# cslc ./src/layout.csl --arch wse2 --fabric-dims=11,10 --fabric-offsets=4,1 \
# --params=width:4,height:4,MAX_ZDIM:4 --params=BLOCK_SIZE:8 --params=C0_ID:0 \
# --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
# --params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out_4x4x4_block8 \
# --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000
# cs_python ./run.py -m=4 -n=4 -k=4 --latestlink out_4x4x4_block8 --channels=1 \
# --width-west-buf=0 --width-east-buf=0 --zDim=4 --run-only

# # Test cases for 16x16x16 with BLOCK_SIZE=2,4,8
# cslc ./src/layout.csl --arch wse2 --fabric-dims=27,26 --fabric-offsets=4,1 \
# --params=width:16,height:16,MAX_ZDIM:16 --params=BLOCK_SIZE:2 --params=C0_ID:0 \
# --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
# --params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out_16x16x16_block2 \
# --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000
# cs_python ./run.py -m=16 -n=16 -k=16 --latestlink out_16x16x16_block2 --channels=1 \
# --width-west-buf=0 --width-east-buf=0 --zDim=16 --run-only

# cslc ./src/layout.csl --arch wse2 --fabric-dims=27,26 --fabric-offsets=4,1 \
# --params=width:16,height:16,MAX_ZDIM:16 --params=BLOCK_SIZE:4 --params=C0_ID:0 \
# --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
# --params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out_16x16x16_block4 \
# --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000
# cs_python ./run.py -m=16 -n=16 -k=16 --latestlink out_16x16x16_block4 --channels=1 \
# --width-west-buf=0 --width-east-buf=0 --zDim=16 --run-only

cslc ./src/layout.csl --arch wse2 --fabric-dims=27,26 --fabric-offsets=4,1 \
--params=width:16,height:16,MAX_ZDIM:16 --params=BLOCK_SIZE:8 --params=C0_ID:0 \
--params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
--params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out_16x16x16_block8 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000
cs_python ./run.py -m=16 -n=16 -k=16 --latestlink out_16x16x16_block8 --channels=1 \
--width-west-buf=0 --width-east-buf=0 --zDim=16 --run-only

