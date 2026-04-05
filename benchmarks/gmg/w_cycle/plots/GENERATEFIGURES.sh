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

# ----------------------------------------------------------------------------
# Step 1: aggregate response.txt files into all_responses_*.txt
# ----------------------------------------------------------------------------
echo "=========================================="
echo "Step 1: Aggregating response files"
echo "=========================================="

collect() {
    local pattern="$1"
    local outfile="$2"
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

collect 'out_dir_S*x*_P6_P6_B100' all_responses_6_6_100.txt
collect 'out_dir_S*x*_P4_P4_B100' all_responses_4_4_100.txt
collect 'out_dir_S*x*_P4_P4_B6'   all_responses_4_4_6.txt
collect 'out_dir_S*x*_P6_P6_B6'   all_responses_6_6_6.txt
collect 'shallow_*'               all_responses_6_6_6_shallow.txt
collect 'out_dir_S*unoptimized*'  all_responses_6_6_6_unoptimized.txt

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
    if [ -s "../${input}" ]; then
        echo "  ${input} -> ${output}"
        python plot_gmg_performance.py "../${input}" > "${output}" 2>&1
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
echo ""
echo "=========================================="
echo "Step 3: Comparison/analysis plots"
echo "=========================================="

if [ -f out_6_6_6.txt ] && [ -f out_6_6_6_unoptimized.txt ]; then
    echo "  optimized_vs_unoptimized.py -> metrics_comparison.png"
    python optimized_vs_unoptimized.py out_6_6_6.txt out_6_6_6_unoptimized.txt
else
    echo "  SKIP optimized_vs_unoptimized.py: missing out_6_6_6*.txt"
fi

if [ -f h200_vs_cs3_feb7.csv ]; then
    echo "  h200_vs_cs3.py -> hpgmg_speedup_barplot.pdf"
    python h200_vs_cs3.py h200_vs_cs3_feb7.csv
else
    echo "  SKIP h200_vs_cs3.py: missing h200_vs_cs3_feb7.csv"
fi

# ----------------------------------------------------------------------------
# Step 4: standalone plot scripts (read response.txt or aggregates)
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step 4: Standalone plots"
echo "=========================================="

echo "  time_to_solution.py -> time_to_solution.png + iterations_comparison.png"
python time_to_solution.py 2>&1 | tail -3

echo "  spider_plot.py -> spider_plot.png"
python spider_plot.py 2>&1 | tail -3

echo "  wafer_utilization.py -> wafer_utilization.png"
python wafer_utilization.py 2>&1 | tail -2

echo "  fixed_tolerance_analysis.py -> fixed_tolerance_convergence.png"
python fixed_tolerance_analysis.py 2>&1 | tail -3

echo "  plot_convergence.py -> convergence_*.png"
python plot_convergence.py 2>&1 | tail -3

# ----------------------------------------------------------------------------
# Step 5: roofline analysis (one per sample problem)
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step 5: Roofline analysis"
echo "=========================================="

ROOFLINE_SAMPLE="${CSL_GMG_DIR}/out_dir_S512x_L9_M100_P6_P6_B6/response.txt"
if [ -s "${ROOFLINE_SAMPLE}" ]; then
    echo "  roofline_analysis.py <- 512³ 6/6/6 -> roofline_plot.png"
    python roofline_analysis.py "${ROOFLINE_SAMPLE}" > /dev/null
else
    echo "  SKIP roofline_analysis.py: no 512³ 6/6/6 response.txt"
fi

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Done — generated figures:"
echo "=========================================="
ls -1 *.png *.pdf 2>/dev/null | sort
