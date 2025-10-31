#!/bin/bash

# Experiment number 1: Maximum V cycle depth (meaning 2x2 mimimum problem size)
# Everyone reaches at minimum 2x2 size at coarsest level
# python compile_and_run_wse3.py --size 8 --levels 3 --channels 5
# python compile_and_run_wse3.py --size 16 --levels 4 --channels 10
# python compile_and_run_wse3.py --size 32 --levels 5 --channels 16
# python compile_and_run_wse3.py --size 64 --levels 6 --channels 16
# python compile_and_run_wse3.py --size 128 --levels 7 --channels 16
# python compile_and_run_wse3.py --size 256 --levels 8 --channels 16
# python compile_and_run_wse3.py --size 512 --levels 9 --channels 16

# After this step run data_analysis.py to analyze the data on response.txt file on each folder.
# It will generate a excel output.


# Experiment number 2: Don't go maximum depth this time, try to fit the problem size

python compile_and_run_wse3.py --size 8 --levels 3 --channels 5
python compile_and_run_wse3.py --size 16 --levels 4 --channels 10
python compile_and_run_wse3.py --size 32 --levels 5 --channels 16
python compile_and_run_wse3.py --size 64 --levels 4 --channels 16
python compile_and_run_wse3.py --size 128 --levels 5 --channels 16
# python compile_and_run_wse3.py --size 256 --levels 5 --channels 16
python compile_and_run_wse3.py --size 512 --levels 4 --channels 16