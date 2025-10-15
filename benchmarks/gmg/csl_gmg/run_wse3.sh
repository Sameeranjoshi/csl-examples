#!/usr/bin/env bash

set -e
# if [ -z "$1" ]; then
#     echo "Error: Simulator value must be specified as the first argument (true or false)."
#     exit 1
# fi
# simulator=$1
python compile.py
python launch_wse3.py
