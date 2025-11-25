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

# echo "Compiling..."
# cslc ./src/layout_gmg_vcycle.csl --arch wse3 --fabric-dims=18,11 --fabric-offsets=4,1 \
# --params=width:8,height:8,MAX_ZDIM:8,LEVELS:3,BLOCK_SIZE:8 -o=out_vcycle \
# --memcpy --channels=8 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000

# cs_python ./run_gmg_vcycle.py -m=8 -n=8 -k=8 --latestlink out_vcycle --channels=8 \
# --width-west-buf=0 --width-east-buf=0 --zDim=8 --run-only --levels=3 --max-ite=1

# ./check_memory_usage.sh out_vcycle 0 0 --summary


echo "============================================"
echo "Done"
echo "============================================"
echo "Compiling..."
cslc ./src/layout_gmg_vcycle.csl --arch wse3 --fabric-dims=16,12 --fabric-offsets=4,1 \
--params=width:8,height:8,MAX_ZDIM:8,LEVELS:3,BLOCK_SIZE:8 -o=out_vcycle \
--memcpy --channels=4 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000

cs_python ./run_gmg_vcycle.py -m=8 -n=8 -k=8 --latestlink out_vcycle --channels=4 \
--width-west-buf=0 --width-east-buf=0 --zDim=8 --run-only --levels=3 --max-ite=1

./check_memory_usage.sh out_vcycle 0 0 --summary

echo "============================================"
echo "Done"
echo "============================================"

# echo "Compiling..."
# cslc ./src/layout_gmg_vcycle.csl --arch wse3 --fabric-dims=42,35 --fabric-offsets=4,1 \
# --params=width:32,height:32,MAX_ZDIM:32,LEVELS:5,BLOCK_SIZE:32 -o=out_vcycle \
# --memcpy --channels=15 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000

# cs_python ./run_gmg_vcycle.py -m=32 -n=32 -k=32 --latestlink out_vcycle --channels=15 \
# --width-west-buf=0 --width-east-buf=0 --zDim=32 --run-only --levels=5 --max-ite=100

# ./check_memory_usage.sh out_vcycle 0 0 --summary

# echo "============================================"
# echo "Done"
# echo "============================================"

# echo "Compiling..."
# cslc ./src/layout_gmg_vcycle.csl --arch wse3 --fabric-dims=74,67 --fabric-offsets=4,1 \
# --params=width:64,height:64,MAX_ZDIM:64,LEVELS:6,BLOCK_SIZE:64 -o=out_vcycle \
# --memcpy --channels=15 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000

# cs_python ./run_gmg_vcycle.py -m=64 -n=64 -k=64 --latestlink out_vcycle --channels=15 \
# --width-west-buf=0 --width-east-buf=0 --zDim=64 --run-only --levels=6 --max-ite=100

# ./check_memory_usage.sh out_vcycle 0 0 --summary

# echo "============================================"
# echo "Done"
# echo "============================================"