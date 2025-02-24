#!/usr/bin/env bash

set -e

# cslc ./src/layout.csl \
# --arch wse2 --fabric-dims=24,14 --fabric-offsets=4,1 \
# --params=matrix_rows:100,matrix_cols:100 \
# --params=pe_rows:10,pe_cols:10 \
# --params=BLOCK_SIZE:2 \
# --params=C0_ID:0 --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
# --params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out \
# --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 -o out

# # Run it
cs_python ./run.py --max-ite=80 --run-only --name out

# CG - needs matrix to be symmetric and positive definite.