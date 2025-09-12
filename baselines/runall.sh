#!/bin/bash


SPARSE_MATRIX=matrix.mtx

if [ "$1" = "SINGLE" ]; then
    echo "Running in single GPU mode"
    GPU_MODE="single"
    BINARY=amgx_capi
elif [ "$1" = "MULTIPLE" ]; then
    echo "Running in multiple GPU mode"
    GPU_MODE="multiple"
    BINARY=amgx_mpi_capi
else
    echo "Usage: $0 [SINGLE|MULTIPLE]"
    echo "Please specify SINGLE or MULTIPLE as parameter"
    exit 1
fi


echo "Step 1: Changing directory to AMGX"
cd AMGX
mkdir --p logs/

for configs in ./src/configs/*.json; do
  echo "=== Running with config $configs ==="
  if [ "$GPU_MODE" = "single" ]; then
    ./install/lib/examples/$BINARY -m ./examples/$SPARSE_MATRIX -c $configs > ./logs/output_$(basename $configs .json)_$GPU_MODE.log 2>&1
  elif [ "$GPU_MODE" = "multiple" ]; then
    mpirun ./install/lib/examples/$BINARY -m ./examples/$SPARSE_MATRIX -c $configs > ./logs/output_$(basename $configs .json)_$GPU_MODE.log 2>&1
  fi
  echo "Output written to ./logs/output_$(basename $configs .json)_$GPU_MODE.log"
done


# 2. Multi GPU
# mpirun -n 2 ./install/lib/examples/amgx_mpi_capi -m ../examples/matrix.mtx -c ../src/configs/AMG_CLASSICAL_CG.json
