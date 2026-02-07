#!/bin/bash

python plot_gmg_performance.py ../all_responses_6_6_6_interpolation.txt
python optimized_vs_unoptimized.py out_3_3_6.txt out_3_3_6_unoptimized.txt
python h200_vs_cs3.py h200_vs_cs3_feb6.csv
