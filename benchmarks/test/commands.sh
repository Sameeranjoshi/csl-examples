#!/usr/bin/env bash

set -e

cslc layout.csl --arch=wse2 --fabric-dims=20,6 --fabric-offsets=4,1 --params=layer_N_0:10,layer_M_0:11,layer_K_0:12,layer_L_0:13,layer_N_1:14,layer_M_1:15,layer_K_1:16,layer_L_1:17,total_levels:2 --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000 -o out
cs_python run.py --name out