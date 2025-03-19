#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./single_layer_layout.csl --fabric-dims=71,66 --fabric-offsets=4,1 \
--params=layer_rows_A:16,layer_cols_A:16 \
--params=layer_rows_R:8,layer_cols_R:16 \
--params=layer_start_x:0,layer_start_y:0,layer_kernel_rows:4,layer_kernel_cols:4,layer_index:0 \
--memcpy --channels=1 --max-inlined-iterations=1000000 -o out_single_layer 
cs_python single_layer_run.py --name out_single_layer

# 256/8 = 32
# 128/8 = 16
# 512/8 = 64
# 1024/8 = 128
# 2048/8 = 256
# 4096/8 = 512
# 8192/8 = 1024
