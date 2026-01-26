#!/usr/bin/env bash

set +e

# Compile and run GMG V-cycle with state machine on WSE3
# With comprehensive performance timing

# Setup Python path for Singularity container
# This ensures both host packages (like numba) and container packages (like cerebras.sdk) are accessible
# The container paths MUST be included for cerebras.sdk to work
CONTAINER_PATHS="/cbcore/py_root:/cbcore/py_root/cerebras"

if [ -n "$SINGULARITYENV_PYTHONPATH" ]; then
    # If already set, append container paths (they might not be included)
    if [[ "$SINGULARITYENV_PYTHONPATH" != *"/cbcore/py_root"* ]]; then
        export SINGULARITYENV_PYTHONPATH="$SINGULARITYENV_PYTHONPATH:$CONTAINER_PATHS"
    fi
elif [ -n "$SINGULARITY_PYTHONPATH" ]; then
    # Handle legacy SINGULARITY_PYTHONPATH variable
    if [[ "$SINGULARITY_PYTHONPATH" != *"/cbcore/py_root"* ]]; then
        export SINGULARITYENV_PYTHONPATH="$SINGULARITY_PYTHONPATH:$CONTAINER_PATHS"
    else
        export SINGULARITYENV_PYTHONPATH="$SINGULARITY_PYTHONPATH"
    fi
else
    # Not set, try to detect host Python path for numba
    PYTHON_CMD=$(which python3 2>/dev/null || which python 2>/dev/null || echo "")
    if [ -n "$PYTHON_CMD" ]; then
        # Check if numba is available in host Python
        if $PYTHON_CMD -c "import numba" 2>/dev/null; then
            # Get the site-packages directory from host Python
            HOST_PYTHONPATH=$($PYTHON_CMD -c "import site; print(':'.join(site.getsitepackages()))" 2>/dev/null || echo "")
            if [ -n "$HOST_PYTHONPATH" ]; then
                export SINGULARITYENV_PYTHONPATH="$HOST_PYTHONPATH:$CONTAINER_PATHS"
            else
                export SINGULARITYENV_PYTHONPATH="$CONTAINER_PATHS"
            fi
        else
            export SINGULARITYENV_PYTHONPATH="$CONTAINER_PATHS"
        fi
    else
        export SINGULARITYENV_PYTHONPATH="$CONTAINER_PATHS"
    fi
fi

echo "============================================"
echo "GMG V-Cycle State Machine with Timing"
echo "============================================"
echo ""

# # (16, 4, 1, 1e-5, 6, 6, 100),   # Small problem
echo "Compiling..."
cslc ./src/layout_gmg_vcycle.csl --arch wse3 --fabric-dims=25,18 --fabric-offsets=4,1 \
--params=width:16,height:16,MAX_ZDIM:16,LEVELS:4,BLOCK_SIZE:16 -o=out_vcycle_16 \
--memcpy --channels=15 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000
cs_python ./run_gmg_vcycle.py -m=16 -n=16 -k=16 --latestlink out_vcycle_16 --channels=15 \
--width-west-buf=0 --width-east-buf=0 --zDim=16 --run-only --levels=4 --max-ite=1 --pre-iter=6 --post-iter=6 --bottom-iter=100 --tolerance=1e-5 --blockSize=16
./check_memory_usage.sh out_vcycle_16 0 0 --summary
# Clean previous runs
# rm -rf out_vcycle sim.log simfab_traces


# echo "============================================"
# echo "Done"
# echo "============================================"
# echo "Compiling..."
# cslc ./src/layout_gmg_vcycle.csl --arch wse3 --fabric-dims=25,18 --fabric-offsets=4,1 \
# --params=width:16,height:16,MAX_ZDIM:16,LEVELS:4,BLOCK_SIZE:16 -o=out_vcycle \
# --memcpy --channels=8 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000
# echo "============================================"
# echo "Running with 1 V-cycle, 1 pre-iter, 1 post-iter, 1 bottom-iter"
# cs_python ./run_gmg_vcycle.py -m=16 -n=16 -k=16 --latestlink out_vcycle --channels=8 \
# --width-west-buf=0 --width-east-buf=0 --zDim=16 --run-only --levels=4 --max-ite=1 --pre-iter=1 --post-iter=1 --bottom-iter=1

# echo "============================================"
# echo "Running with 1 V-cycle, 6 pre-iter, 6 post-iter, ? bottom-iter"
# cs_python ./run_gmg_vcycle.py -m=16 -n=16 -k=16 --latestlink out_vcycle --channels=8 \
# --width-west-buf=0 --width-east-buf=0 --zDim=16 --run-only --levels=4 --max-ite=1 --pre-iter=6 --post-iter=6 --bottom-iter=1
# cs_python ./run_gmg_vcycle.py -m=16 -n=16 -k=16 --latestlink out_vcycle --channels=8 \
# --width-west-buf=0 --width-east-buf=0 --zDim=16 --run-only --levels=4 --max-ite=1 --pre-iter=6 --post-iter=6 --bottom-iter=5
# cs_python ./run_gmg_vcycle.py -m=16 -n=16 -k=16 --latestlink out_vcycle --channels=8 \
# --width-west-buf=0 --width-east-buf=0 --zDim=16 --run-only --levels=4 --max-ite=1 --pre-iter=6 --post-iter=6 --bottom-iter=6
# cs_python ./run_gmg_vcycle.py -m=16 -n=16 -k=16 --latestlink out_vcycle --channels=8 \
# --width-west-buf=0 --width-east-buf=0 --zDim=16 --run-only --levels=4 --max-ite=1 --pre-iter=6 --post-iter=6 --bottom-iter=10
# cs_python ./run_gmg_vcycle.py -m=16 -n=16 -k=16 --latestlink out_vcycle --channels=8 \
# --width-west-buf=0 --width-east-buf=0 --zDim=16 --run-only --levels=4 --max-ite=1 --pre-iter=6 --post-iter=6 --bottom-iter=20

# echo "============================================"
# echo "Running with ? V-cycle, 6 pre-iter, 6 post-iter, 10 bottom-iter"
# cs_python ./run_gmg_vcycle.py -m=16 -n=16 -k=16 --latestlink out_vcycle --channels=8 \
# --width-west-buf=0 --width-east-buf=0 --zDim=16 --run-only --levels=4 --max-ite=5 --pre-iter=6 --post-iter=6 --bottom-iter=10
# cs_python ./run_gmg_vcycle.py -m=16 -n=16 -k=16 --latestlink out_vcycle --channels=8 \
# --width-west-buf=0 --width-east-buf=0 --zDim=16 --run-only --levels=4 --max-ite=6 --pre-iter=6 --post-iter=6 --bottom-iter=10
# cs_python ./run_gmg_vcycle.py -m=16 -n=16 -k=16 --latestlink out_vcycle --channels=8 \
# --width-west-buf=0 --width-east-buf=0 --zDim=16 --run-only --levels=4 --max-ite=10 --pre-iter=6 --post-iter=6 --bottom-iter=10

# # ./check_memory_usage.sh out_vcycle 0 0 --summary

# echo "============================================"
# echo "32 32 32 PROBLEM"
# echo "============================================"
# cslc ./src/layout_gmg_vcycle.csl --arch wse3 --fabric-dims=39,34 --fabric-offsets=4,1 \
# --params=width:32,height:32,MAX_ZDIM:32,LEVELS:5,BLOCK_SIZE:32 -o=out_vcycle_32 \
# --memcpy --channels=16 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000

# echo "Running with ? V-cycle, 6 pre-iter, 6 post-iter, 10 bottom-iter"
# echo "============================================"
# cs_python ./run_gmg_vcycle.py -m=32 -n=32 -k=32 --latestlink out_vcycle_32 --channels=16 \
# --width-west-buf=0 --width-east-buf=0 --zDim=32 --run-only --levels=5 --max-ite=1 --pre-iter=6 --post-iter=6 --bottom-iter=10
# cs_python ./run_gmg_vcycle.py -m=32 -n=32 -k=32 --latestlink out_vcycle_32 --channels=16 \
# --width-west-buf=0 --width-east-buf=0 --zDim=32 --run-only --levels=5 --max-ite=5 --pre-iter=6 --post-iter=6 --bottom-iter=10
# cs_python ./run_gmg_vcycle.py -m=32 -n=32 -k=32 --latestlink out_vcycle_32 --channels=16 \
# --width-west-buf=0 --width-east-buf=0 --zDim=32 --run-only --levels=5 --max-ite=6 --pre-iter=6 --post-iter=6 --bottom-iter=10
# cs_python ./run_gmg_vcycle.py -m=32 -n=32 -k=32 --latestlink out_vcycle_32 --channels=16 \
# --width-west-buf=0 --width-east-buf=0 --zDim=32 --run-only --levels=5 --max-ite=10 --pre-iter=6 --post-iter=6 --bottom-iter=10

echo "============================================"
echo "Done"
echo "============================================"

# echo "Compiling..."
# cslc ./src/layout_gmg_vcycle.csl --arch wse3 --fabric-dims=42,35 --fabric-offsets=4,1 \
# --params=width:32,height:32,MAX_ZDIM:32,LEVELS:5,BLOCK_SIZE:32 -o=out_vcycle \
# --memcpy --channels=15 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000

# cs_python ./run_gmg_vcycle.py -m=32 -n=32 -k=32 --latestlink out_vcycle --channels=15 \
# --width-west-buf=0 --width-east-buf=0 --zDim=32 --run-only --levels=5 --max-ite=100

# ./check_memory_usage.sh out_vcycle 0 0 --summary

# echo "============================================"
# echo "Done"
# echo "============================================"

# echo "Compiling..."
# cslc ./src/layout_gmg_vcycle.csl --arch wse3 --fabric-dims=74,67 --fabric-offsets=4,1 \
# --params=width:64,height:64,MAX_ZDIM:64,LEVELS:6,BLOCK_SIZE:64 -o=out_vcycle \
# --memcpy --channels=15 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000

# cs_python ./run_gmg_vcycle.py -m=64 -n=64 -k=64 --latestlink out_vcycle --channels=15 \
# --width-west-buf=0 --width-east-buf=0 --zDim=64 --run-only --levels=6 --max-ite=100

# ./check_memory_usage.sh out_vcycle 0 0 --summary

# echo "============================================"
# echo "Done"
# echo "============================================"