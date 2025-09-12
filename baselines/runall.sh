#!/bin/bash

echo "Step 1: Changing directory to build"
cd build

for configs in ../src/configs/*.json; do
  echo "=== Running with config $configs ==="
  ../install/lib/examples/amgx_mpi_capi -m ../examples/matrix.mtx -c $configs > ../output_$(basename $configs .json).log 2>&1
  echo "Output written to ../output_$(basename $configs .json).log"
done

# 1. gpu
# ./install/lib/examples/amgx_mpi_capi -m ./examples/matrix.mtx -c ./src/configs/AMG_CLASSICAL_CG.json

# 2. Multi GPU
# mpirun -n 2 ./install/lib/examples/amgx_mpi_capi -m ../examples/matrix.mtx -c ../src/configs/AMG_CLASSICAL_CG.json