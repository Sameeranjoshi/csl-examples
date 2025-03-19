#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./single_layer_layout.csl --fabric-dims=22,12 --fabric-offsets=4,1 \
--params=layer_rows_A:4,layer_cols_A:4 \
--params=layer_rows_R:2,layer_cols_R:4 \
--params=layer_start_x:0,layer_start_y:0,layer_kernel_rows:2,layer_kernel_cols:2,layer_index:0 \
--memcpy --channels=1 --max-inlined-iterations=1000000 -o out_single_layer 
cs_python single_layer_run.py --name out_single_layer
