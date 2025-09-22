#!/usr/bin/env bash

set -e

# Compile and run GMG on WSE3
cs_python ./run_gmg.py -m=4 -n=4 -k=4 --latestlink out --channels=1 \
--width-west-buf=0 --width-east-buf=0 --zDim=4 --run-only --max-ite=5
