#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./single_layer_layout.csl --fabric-dims=22,12 --fabric-offsets=4,1 \
--params=layer_rows_A:32,layer_cols_A:32 \
--params=layer_start_x:0,layer_start_y:0,layer_kernel_rows:4,layer_kernel_cols:4,layer_index:0 \
--memcpy --channels=1 -o out_single_layer
cs_python single_layer_run.py --name out_single_layer
