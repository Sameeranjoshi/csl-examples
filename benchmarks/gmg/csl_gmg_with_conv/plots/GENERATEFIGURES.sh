#!/bin/bash
# GENERATEFIGURES.sh
#
# Single entry point to regenerate ALL plots for the GMG V-cycle paper.
# Run from this directory (plots/):  ./GENERATEFIGURES.sh
#
# Pipeline:
#   1. Aggregate response.txt files from out_dir_*/ into all_responses_*.txt
#   2. Parse response data with plot_gmg_performance.py -> out_*.txt + internal plots
#   3. Run analysis/comparison scripts that depend on out_*.txt
#   4. Run standalone plot scripts that read response.txt directly
#
# See README.md for script descriptions, outputs, and prerequisites.

set -u  # unset variable -> error (but continue on errors: no -e)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CSL_GMG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${CSL_GMG_DIR}" || exit 1

# Single log file for all script output
LOGFILE="${SCRIPT_DIR}/GENERATEFIGURES.log"
> "${LOGFILE}"  # truncate

# ----------------------------------------------------------------------------
# Step 1: aggregate response.txt files into all_responses_*.txt
# ----------------------------------------------------------------------------
echo "=========================================="
echo "Step 1: Aggregating response files"
echo "=========================================="

# All aggregated all_responses_*.txt files are written into build/ (alongside
# the out_dir_*/ they were aggregated from). Plot scripts read them from build/.
mkdir -p build
collect() {
    local pattern="$1"
    local outfile="build/$2"
    if ls -d ${pattern} 2>/dev/null | head -1 > /dev/null; then
        ls -d ${pattern} \
          | sed 's/.*S\([0-9]*\)x.*/\1 &/' \
          | sort -n | cut -d' ' -f2- \
          | xargs -I{} cat {}/response.txt > "${outfile}" 2>/dev/null
        echo "  -> ${outfile} ($(wc -l < ${outfile}) lines)"
    else
        echo "  SKIP ${outfile}: no matching dirs for pattern '${pattern}'"
    fi
}

# NOTE: out_dir_* and shallow_* directories now live under build/ (see
# compile_and_run_wse3.py's BUILD_DIR). The collect globs reflect that.
collect 'build/out_dir_S*x*_P6_P6_B100' all_responses_6_6_100.txt
collect 'build/out_dir_S*x*_P4_P4_B100' all_responses_4_4_100.txt
collect 'build/out_dir_S*x*_P4_P4_B6'   all_responses_4_4_6.txt
collect 'build/out_dir_S*x*_P6_P6_B6'   all_responses_6_6_6.txt
collect 'build/shallow_*'               all_responses_6_6_6_shallow.txt
collect 'build/out_dir_S*unoptimized*'  all_responses_6_6_6_unoptimized.txt

# ----------------------------------------------------------------------------
# Step 2: generate out_*.txt + internal plots from each aggregated file
#   plot_gmg_performance.py produces:
#     stdout -> out_*.txt (tee'd)
#     spmv_internal.png, interpolation_internal.png, per_operation_timing_*.png
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step 2: Running plot_gmg_performance.py"
echo "=========================================="
cd "${SCRIPT_DIR}"

run_perf() {
    local input="$1"
    local output="$2"
    # Inputs (all_responses_*.txt) live under build/ now.
    if [ -s "../build/${input}" ]; then
        echo "  ${input} -> ${output}"
        python plot_gmg_performance.py "../build/${input}" > "${output}" 2>&1
    else
        echo "  SKIP ${input}: file missing or empty"
    fi
}

run_perf all_responses_6_6_100.txt        out_6_6_100.txt
run_perf all_responses_4_4_100.txt        out_4_4_100.txt
run_perf all_responses_4_4_6.txt          out_4_4_6.txt
run_perf all_responses_6_6_6_shallow.txt  out_6_6_6_shallow.txt
run_perf all_responses_6_6_6_unoptimized.txt out_6_6_6_unoptimized.txt
# 6/6/6 last so its plots (spmv_internal, interpolation_internal, per_operation_timing_512)
# are the ones kept on disk — they are the paper's primary figures.
run_perf all_responses_6_6_6.txt          out_6_6_6.txt

# ----------------------------------------------------------------------------
# Step 3: analysis/comparison scripts that depend on out_*.txt
# ----------------------------------------------------------------------------
# echo ""
# echo "=========================================="
# echo "Step 3: Comparison/analysis plots"
# echo "=========================================="

# if [ -f out_6_6_6.txt ] && [ -f out_6_6_6_unoptimized.txt ]; then
#     echo "  optimized_vs_unoptimized.py -> metrics_comparison.png"
#     { echo "--- optimized_vs_unoptimized.py ---"; python optimized_vs_unoptimized.py out_6_6_6.txt out_6_6_6_unoptimized.txt 2>&1; echo; } | tee -a "${LOGFILE}"
# else
#     echo "  SKIP optimized_vs_unoptimized.py: missing out_6_6_6*.txt"
# fi

# if [ -f h200_vs_cs3_april6.csv ]; then
#     echo "  h200_vs_cs3.py -> hpgmg_speedup_barplot.pdf"
#     { echo "--- h200_vs_cs3.py ---"; python h200_vs_cs3.py h200_vs_cs3_april6.csv 2>&1; echo; } | tee -a "${LOGFILE}"
# else
#     echo "  SKIP h200_vs_cs3.py: missing h200_vs_cs3_april6.csv"
# fi

# ----------------------------------------------------------------------------
# Step 4: standalone plot scripts (read response.txt or aggregates)
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step 4: Standalone plots"
echo "=========================================="

# echo "  time_to_solution.py -> time_to_solution.png + iterations_comparison.png"
# { echo "--- time_to_solution.py ---"; python time_to_solution.py 2>&1; echo; } | tee -a "${LOGFILE}"

#echo "  spider_plot.py -> spider_plot.png"
#{ echo "--- spider_plot.py ---"; python spider_plot.py 2>&1; echo; } | tee -a "${LOGFILE}"

# echo "  wafer_utilization.py -> wafer_utilization.png"
# { echo "--- wafer_utilization.py ---"; python wafer_utilization.py 2>&1; echo; } | tee -a "${LOGFILE}"

# echo "  fixed_tolerance_analysis.py -> fixed_tolerance_convergence.png"
# { echo "--- fixed_tolerance_analysis.py ---"; python fixed_tolerance_analysis.py 2>&1; echo; } | tee -a "${LOGFILE}"

# echo "  plot_convergence.py -> convergence_*.png"
# { echo "--- plot_convergence.py ---"; python plot_convergence.py 2>&1; echo; } | tee -a "${LOGFILE}"

# echo " Memory and utilization table"
# python memory_utilization_table.py out_6_6_6.txt

# echo "  v_vs_w_cycle.py -> W vs V cycle plot"
# { echo "--- v_vs_w_cycle.py ---"; python v_vs_w_cycle.py 2>&1; echo; } | tee -a "${LOGFILE}"

echo "  tts_comparison.py -> time-per-V-cycle table"
{ echo "--- tts_comparison.py ---"; python tts_comparison.py 2>&1; echo; } | tee -a "${LOGFILE}"
# ----------------------------------------------------------------------------
# Step 5: roofline analysis (one per sample problem)
# ----------------------------------------------------------------------------
# echo ""
# echo "=========================================="
# echo "Step 5: Roofline analysis"
# echo "=========================================="

# ROOFLINE_SAMPLE="${CSL_GMG_DIR}/build/out_dir_S512x_L9_M100_P6_P6_B6/response.txt"
# if [ -s "${ROOFLINE_SAMPLE}" ]; then
#     echo "  roofline_analysis.py <- 512³ 6/6/6 -> roofline_plot.png"
#     { echo "--- roofline_analysis.py ---"; python roofline_analysis.py "${ROOFLINE_SAMPLE}" 2>&1; echo; } | tee -a "${LOGFILE}"
# else
#     echo "  SKIP roofline_analysis.py: no 512³ 6/6/6 response.txt"
# fi

python print_results_table.py --build ../build/

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Done — generated figures:"
echo "=========================================="
ls -1 *.png *.pdf 2>/dev/null | sort
