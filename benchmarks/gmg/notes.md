Constants and parameters

    MPI_ALPHA = -6, MPI_BETA = 1: coefficients for 7-point Poisson stencil.

    JACOBI_COEFF = 1.0/9.0: relaxation weight for Jacobi/Gauss–Seidel.

    Iteration counts: PRE_SMOOTH_ITER, POST_SMOOTH_ITER, BOTTOM_SOLVER_ITER.

CUDA kernels (run on GPU):

    atomicMaxB: atomic maximum for doubles.

    reduction_kernel: computes the max norm of residual.
        Do you know why? 

    initX_kernel: 
        sets the solution x to 0.

    applyOp_kernel: applies the discrete Poisson stencil operator (auto-generated).
        3D-7-pt-stencil()

    smooth_residual_kernel: smoother (Jacobi) and updates residual.
        x' = x + gamma(Ax-b)
        res = b - A*x'

    smooth_kernel: smoother only (Jacobi).
        x = x + gamma(Ax - b)

    restriction_kernel: restricts residual to coarser level (full weighting).
        sum(residual)/8, which 8 poitns (2D space or 3D space?)

    interpolation_incr_kernel: interpolates coarse correction and adds it to fine grid.
        TODO:write later

Host-side functions (run on CPU but launch kernels / coordinate MPI):

    exchangeField: perform MPI halo exchange (GPU-aware).
        Send(), recv() from 3D_7_pt stencils, sends and receives halo regions.

    init_mg_brick: initialize levels and allocate GPU data (same as you saw).

    point_relax: one smoothing iteration (apply operator → Jacobi update).
        // loop (# of smoothing iter times )
            X_local' = smoothing(x_local)

    restriction: launch restriction kernel.

    interpolation_incr: launch interpolation kernel.

    exchange_pointRelax: combine halo exchange with smoothing.

    vcycle_brick: the actual V-cycle driver (pre-smooth → restriction → bottom solve → prolongation → post-smooth).

    initX_brick: zero initialize solution.

    setTimersZero: reset timers.

    maxNorm_brick: compute maximum residual norm via reduction kernel.


Algorithm main : 
1. get parameters from user
2. build the shapes and level hierarchy
3. Allocate and fill values in the data structures
4. Domain decomposition: Partition the problem to map onto grid of WSE
5. Copy data to device
6. <<<v cycle>>>
7. Get results back
8. Get time back and reorganize time formats
9. Print metrics



########################################
@@@@@@@@@@@@@@@
[u1418973@notch371:nvidia_cuda]$ ./cuda -h
Running MPI with cuda

Program options
  -h: show help (this message)
  MPI downsizing:
  -b: MPI downsize to 2-exponential   Domain size, pick either one, in array order contiguous first
  -d: comma separated Int[3], overall domain size(64x64x64)
  -s: comma separated Int[3], per-process domain size(1x1x1)
  Benchmark control:
  -I: number of iterations, default 100(MPI_ITERATION)
  -l: number of levels for the v-cycle, default 0
  -n: maximum number of iterations, default 10
Example usage:
  ./cuda -d 2048,2048,2048

@@@@@@@@@@@@@@@

$ cd examples/gmg/nvidia_cuda && ./cuda
Pagesize 4096; MPI Size 1 * OpenMP threads 32
Domain size of 262144 split among(64^3)
A total of 1 processes 1x1x1
Running with MPI_GPU_AWARE = 1
Warm up runs
Test # 1
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 2
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 3
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 4
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 5
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 6
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 7
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 8
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 9
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 10
max resGPU 1.13030547e-310 iter 1 
========================= 

Running solver 100 times for statistics

Test # 1
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 2
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 3
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 4
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 5
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 6
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 7
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 8
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 9
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 10
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 11
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 12
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 13
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 14
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 15
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 16
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 17
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 18
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 19
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 20
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 21
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 22
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 23
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 24
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 25
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 26
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 27
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 28
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 29
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 30
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 31
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 32
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 33
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 34
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 35
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 36
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 37
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 38
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 39
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 40
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 41
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 42
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 43
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 44
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 45
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 46
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 47
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 48
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 49
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 50
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 51
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 52
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 53
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 54
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 55
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 56
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 57
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 58
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 59
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 60
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 61
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 62
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 63
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 64
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 65
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 66
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 67
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 68
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 69
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 70
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 71
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 72
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 73
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 74
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 75
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 76
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 77
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 78
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 79
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 80
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 81
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 82
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 83
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 84
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 85
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 86
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 87
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 88
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 89
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 90
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 91
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 92
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 93
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 94
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 95
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 96
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 97
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 98
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 99
max resGPU 1.13030547e-310 iter 1 
========================= 
Test # 100
max resGPU 1.13030547e-310 iter 1 
========================= 
Total Time per operation and level [min,avg,max] (stdev)
level 0 smooth+residual [3.73166e-43, 3.73166e-43, 3.73166e-43] (σ: 0)
Total smooth+residual 3.73166e-43
level 0 applyOp [3.73166e-43, 3.73166e-43, 3.73166e-43] (σ: 0)
Total applyOp 3.73166e-43
level 0 restriction [0, 0, 0] (σ: 0)
Total restriction 0
level 0 interpolation+incr [0, 0, 0] (σ: 0)
Total interpolation+incr 0
level 0 exchange [0.00562412, 0.00562412, 0.00562412] (σ: 0)
Total exchange 0.00562412
maxNormRes [0, 0, 0] (σ: 0)
======================== 
Timings per invocation 
level 0 smooth+residual - time per inv: 3.73166e-45
level 0 applyOp - time per inv: 3.73166e-45
level 0 restriction - time per inv: -nan
level 0 interpolation+incr - time per inv: -nan
level 0 exchange - time per inv: 0.000374941
======================== 
Total Time per Level 
level 0 Total time 0.00562412
======================== 
Bricks-GMG Total Time: 0.00562412
Perf 0.0466107 GStencil/s
$ 
#############################