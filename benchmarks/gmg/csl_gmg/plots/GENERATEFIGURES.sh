#!/bin/bash


# step1 - collect into 1 file
# Need to go one folder behind to do this and also save in the outer folder
cd ../
ls -d out_dir_S*x*_P6_P6_B100 | sed 's/.*S\([0-9]*\)x.*/\1 &/' | sort -n | cut -d' ' -f2- | xargs -I{} cat {}/response.txt > all_responses_6_6_100.txt
ls -d out_dir_S*x*_P4_P4_B100 | sed 's/.*S\([0-9]*\)x.*/\1 &/' | sort -n | cut -d' ' -f2- | xargs -I{} cat {}/response.txt > all_responses_4_4_100.txt
ls -d out_dir_S*x*_P4_P4_B6 | sed 's/.*S\([0-9]*\)x.*/\1 &/' | sort -n | cut -d' ' -f2- | xargs -I{} cat {}/response.txt > all_responses_4_4_6.txt
ls -d out_dir_S*x*_P6_P6_B6 | sed 's/.*S\([0-9]*\)x.*/\1 &/' | sort -n | cut -d' ' -f2- | xargs -I{} cat {}/response.txt > all_responses_6_6_6.txt
ls -d shallow_* | sed 's/.*S\([0-9]*\)x.*/\1 &/' | sort -n | cut -d' ' -f2- | xargs -I{} cat {}/response.txt > all_responses_6_6_6_shallow.txt
ls -d out_dir_S*unoptimized* | sed 's/.*S\([0-9]*\)x.*/\1 &/' | sort -n | cut -d' ' -f2- | xargs -I{} cat {}/response.txt > all_responses_6_6_6_unoptimized.txt
cd plots/

# step 2: plot the performance
python plot_gmg_performance.py ../all_responses_6_6_100.txt | tee out_6_6_100.txt
python plot_gmg_performance.py ../all_responses_4_4_100.txt | tee out_4_4_100.txt
python plot_gmg_performance.py ../all_responses_4_4_6.txt | tee out_4_4_6.txt
python plot_gmg_performance.py ../all_responses_6_6_6_shallow.txt | tee out_6_6_6_shallow.txt
python plot_gmg_performance.py ../all_responses_6_6_6_unoptimized.txt | tee out_6_6_6_unoptimized.txt
# internal spmv, interpolation, and curves.
python plot_gmg_performance.py ../all_responses_6_6_6.txt | tee out_6_6_6.txt

# step 3: plot the metrics comparison
# metrics comparision 
python optimized_vs_unoptimized.py out_6_6_6.txt out_6_6_6_unoptimized.txt

# step 4: plot the final bar chart speedup
python h200_vs_cs3.py h200_vs_cs3_feb7.csv