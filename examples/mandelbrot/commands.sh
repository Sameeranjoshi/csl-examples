#!/usr/bin/env bash

set -e

cslc ./code.csl --fabric-dims=6,6 --fabric-offsets=1,1 -o out
cs_python run.py --name out
