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
cslc ./src/layout_gmg_vcycle.csl --arch wse3 --fabric-dims=25,20 --fabric-offsets=4,1 \
--params=width:8,height:8,MAX_ZDIM:8,LEVELS:2,BLOCK_SIZE:8 -o=out_vcycle \
--memcpy --channels=8 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000

cs_python ./run_gmg_vcycle.py -m=8 -n=8 -k=8 --latestlink out_vcycle --channels=8 \
--width-west-buf=0 --width-east-buf=0 --zDim=8 --run-only --levels=2 --max-ite=100

./check_memory_usage.sh out_vcycle 0 0 --summary