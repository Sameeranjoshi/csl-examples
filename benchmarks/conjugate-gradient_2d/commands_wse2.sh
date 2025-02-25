#!/usr/bin/env bash

set -e

cslc ./src/layout.csl \
--arch wse2 --fabric-dims=12,7 --fabric-offsets=4,1 \
--params=matrix_rows:4,matrix_cols:4 \
--params=pe_rows:1,pe_cols:2 \
--params=BLOCK_SIZE:2 \
--params=C0_ID:0 --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
--params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 -o=out --max-inlined-iterations=1000000

# # # # Run it
# cs_python ./run.py --max-ite=80 --run-only --latestlink out

# CG - needs matrix to be symmetric and positive definite.