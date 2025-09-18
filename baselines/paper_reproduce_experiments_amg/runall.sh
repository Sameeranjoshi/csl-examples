#!/bin/bash

################################################################################
# AMGX Solver Execution Script
################################################################################
#
# DESCRIPTION:
#   This script runs AMGX (Algebraic Multi-Grid eXtended) solver for solving
#   linear systems using either single GPU or multiple GPU configurations.
#   It's designed to work with the Cerebras AMG baseline experiments.
#
# USAGE:
#   ./runall.sh [SINGLE|MULTIPLE]
#
# PARAMETERS:
#   SINGLE   - Run AMGX solver on a single GPU
#   MULTIPLE - Run AMGX solver across multiple GPUs using MPI
#
# PREREQUISITES:
#   1. AMGX must be compiled and installed
#   2. For MULTIPLE mode: MPI must be available
#   3. GPU launcher script must exist at specified path
#   4. Matrix file (.mtx) must exist at specified path
#   5. Configuration file (.json) must exist at specified path
#
# OUTPUT:
#   - Solver execution output to console
#   - Logs directory created in AMGX folder
#
# EXAMPLE:
#   ./runall.sh SINGLE    # Run on single GPU
#   ./runall.sh MULTIPLE  # Run on multiple GPUs
#
################################################################################

########################################################
# Configuration Variables
########################################################
# MPI launcher command for multi-GPU execution
MPI_LAUNCHER=mpirun

# Input matrix file (Matrix Market format) - 50x50x50 Poisson problem
MTX=/uufs/chpc.utah.edu/common/home/u1418973/other/AMGCerebras/baselines/paper_reproduce_experiments/poisson_50x50x50.mtx

# AMGX solver configuration file (JSON format)
CFG=/uufs/chpc.utah.edu/common/home/u1418973/other/AMGCerebras/baselines/AMGX/src/configs/V.json

# GPU launcher script for multi-GPU setup
GPU_LAUNCHER=/uufs/chpc.utah.edu/common/home/u1418973/other/AMGCerebras/baselines/paper_reproduce_experiments/gpu_launcher.sh

# Single GPU AMGX executable
AMGX_BIN=/uufs/chpc.utah.edu/common/home/u1418973/other/AMGCerebras/baselines/AMGX/install/lib/examples/amgx_capi

# Multi-GPU AMGX executable (MPI-enabled)
AMGX_MPI_BIN=/uufs/chpc.utah.edu/common/home/u1418973/other/AMGCerebras/baselines/AMGX/install/lib/examples/amgx_mpi_capi
########################################################
# Command Line Argument Processing
########################################################
# Parse the first command line argument to determine execution mode
if [ "$1" = "SINGLE" ]; then
    echo "Running in single GPU mode"
    GPU_MODE="single"
elif [ "$1" = "MULTIPLE" ]; then
    echo "Running in multiple GPU mode"
    GPU_MODE="multiple"
else
    echo "Usage: $0 [SINGLE|MULTIPLE]"
    echo "Please specify SINGLE or MULTIPLE as parameter"
    echo ""
    echo "SINGLE   - Execute AMGX solver on a single GPU"
    echo "MULTIPLE - Execute AMGX solver across multiple GPUs using MPI"
    exit 1
fi

########################################################
# Execution Setup and Running
########################################################
echo "Step 1: Changing directory to AMGX"
cd AMGX

# Create logs directory for output storage
echo "Step 2: Creating logs directory"
mkdir --p logs/

# Execute AMGX solver based on selected mode
echo "Step 3: Executing AMGX solver in $GPU_MODE mode"
echo "Matrix file: $MTX"
echo "Config file: $CFG"
echo ""

# Note: The commented loop below was designed to run multiple configurations
# Currently, only the V.json configuration is used
# for configs in ./src/configs/*.json; do
#   echo "=== Running with config $configs ==="

if [ "$GPU_MODE" = "single" ]; then
    # Single GPU execution
    echo "Executing: $AMGX_BIN -mode dDDI -m $MTX -c $CFG"
    $AMGX_BIN -mode dDDI -m $MTX -c $CFG
elif [ "$GPU_MODE" = "multiple" ]; then
    # Multi-GPU execution using MPI
    echo "Executing: $MPI_LAUNCHER $GPU_LAUNCHER $AMGX_MPI_BIN -mode dDDI -m $MTX -c $CFG"
    $MPI_LAUNCHER $GPU_LAUNCHER $AMGX_MPI_BIN -mode dDDI -m $MTX -c $CFG
fi

# Note: Logging output to file was commented out
# echo "Output written to ./logs/output_$(basename $CFG .json)_$GPU_MODE.log"
# done

echo ""
echo "Execution completed!"

################################################################################
# TROUBLESHOOTING NOTES
################################################################################
#
# COMMON ISSUES:
# 1. "Permission denied" - Make sure the script is executable: chmod +x runall.sh
# 2. "No such file or directory" - Verify all paths in configuration variables exist
# 3. "mpirun: command not found" - Install MPI or adjust MPI_LAUNCHER variable
# 4. "CUDA out of memory" - Reduce problem size or use multiple GPUs
# 5. "AMGX initialization failed" - Check GPU availability and CUDA installation
#
# AMGX COMMAND LINE OPTIONS:
#   -mode dDDI: Double precision, Double precision, Double precision, Integer indices
#   -m <file>: Matrix file (Matrix Market format)
#   -c <file>: Configuration file (JSON format)
#
# CONFIGURATION FILES:
#   The script uses V.json configuration. Other configs available in:
#   ./AMGX/src/configs/
#
# LOGGING:
#   Currently, output goes to console. To enable file logging, uncomment the
#   logging line in the execution section.
#
################################################################################