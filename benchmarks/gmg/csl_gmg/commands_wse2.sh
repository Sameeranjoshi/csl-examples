#!/usr/bin/env bash

set -e

# Compile and run GMG on WSE2
cs_python ./run_gmg.py -m=8 -n=8 -k=8 --latestlink out --channels=1 \
--width-west-buf=0 --width-east-buf=0 --zDim=8 --run-only --max-ite=10


