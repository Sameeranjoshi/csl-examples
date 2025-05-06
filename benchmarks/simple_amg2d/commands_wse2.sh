#!/usr/bin/env bash

set -e

cslc ./src/auto_layout_amg.csl --arch=wse2 --fabric-dims=11,4 --fabric-offsets=4,1 \
 --params=total_pe_rows:2,total_pe_cols:4,total_levels:2 \
 --params=layer_M_0:50,layer_N_0:50,layer_R_M_0:4,layer_R_N_0:50,layer_start_x_0:0,layer_start_y_0:0,layer_pe_cols_0:2,layer_pe_rows_0:2,layer_index_0:0 \
 --params=layer_M_1:4,layer_N_1:4,layer_R_M_1:2,layer_R_N_1:4,layer_start_x_1:2,layer_start_y_1:0,layer_pe_cols_1:2,layer_pe_rows_1:2,layer_index_1:1 \
 --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000 -o out
cs_python ./run_amg.py