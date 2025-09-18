#!/usr/bin/env bash
# gpu_launcher.sh: wrapper to pin each MPI rank to a local GPU
# Usage: mpirun -n 2 ./gpu_launcher.sh /path/to/amgx_mpi_capi --mode dDDI -m /path/to/matrix.mtx -c /path/to/config.json

# determine local rank (OpenMPI, MVAPICH2, Intel MPI)
if [ -n "$OMPI_COMM_WORLD_LOCAL_RANK" ]; then
  local_rank="$OMPI_COMM_WORLD_LOCAL_RANK"
elif [ -n "$MV2_COMM_WORLD_LOCAL_RANK" ]; then
  local_rank="$MV2_COMM_WORLD_LOCAL_RANK"
elif [ -n "$SLURM_LOCALID" ]; then
  local_rank="$SLURM_LOCALID"
else
  # fallback to world rank (may fail if multiple nodes)
  local_rank="$OMPI_COMM_WORLD_RANK"
fi

# pick device id = local_rank (assumes <= #GPUs per node)
export CUDA_VISIBLE_DEVICES="$local_rank"
echo "Rank $OMPI_COMM_WORLD_RANK (local $local_rank) -> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# exec the provided command
exec "$@"

