#!/usr/bin/env bash

set -e

# Compile and run GMG V-cycle with state machine on WSE3
# With comprehensive performance timing

echo "============================================"
echo "GMG V-Cycle State Machine with Timing"
echo "============================================"
echo ""

# Clean previous runs
# rm -rf out_vcycle sim.log simfab_traces

echo "Compiling..."
cslc ./src/layout_gmg_vcycle.csl --arch wse3 --fabric-dims=37,15 --fabric-offsets=17,1 \
--params=width:4,height:4,MAX_ZDIM:4,LEVELS:2 --params=BLOCK_SIZE:4 \
--params=C0_ID:0 --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 \
--params=C4_ID:4 --params=C5_ID:5 --params=C6_ID:6 --params=C7_ID:7 \
--params=C8_ID:8 -o=out_vcycle \
--memcpy --channels=4 --width-west-buf=13 --width-east-buf=13 --max-inlined-iterations=1000000

cs_python ./run_gmg_vcycle.py -m=4 -n=4 -k=4 --latestlink out_vcycle --channels=4 \
--width-west-buf=13 --width-east-buf=13 --zDim=4 --run-only --levels=2 --max-ite=1 \
--pre-iter=3 --post-iter=3 --bottom-iter=10

echo ""
echo "============================================"
echo "Check output above for timing breakdown!"
echo "============================================"

