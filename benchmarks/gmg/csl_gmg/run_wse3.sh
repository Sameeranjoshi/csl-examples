#!/usr/bin/env bash

set -e
# if [ -z "$1" ]; then
#     echo "Error: Simulator value must be specified as the first argument (true or false)."
#     exit 1
# fi
# simulator=$1
python compile.py
# python run_gmg.py -m=8 -n=8 -k=8 --latestlink out --channels=1 --width-west-buf=0 --width-east-buf=0 --zDim=8 --run-only --levels=3
