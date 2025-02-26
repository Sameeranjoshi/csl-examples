#!/usr/bin/env bash

set -e

# cslc ./src/layout_amg.csl \
# --arch wse2 --fabric-dims=12,7 --fabric-offsets=4,1 \
# --params=matrix_rows:4,matrix_cols:4 \
# --params=pe_rows:1,pe_cols:1 \
# --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 -o=out --max-inlined-iterations=1000000

cslc ./src/layout_amg.csl \
--params=pe_rows:1,pe_cols:1 \
--arch wse2 --fabric-dims=8,3 --fabric-offsets=4,1 \
--memcpy --channels=1 --max-inlined-iterations=1000000 -o out 




# # # # Run it
# cs_python ./run.py --max-ite=80 --run-only --latestlink out

# CG - needs matrix to be symmetric and positive definite.