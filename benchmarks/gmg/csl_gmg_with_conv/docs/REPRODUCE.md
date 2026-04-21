# Artifact Evaluation — Reproduction Guide

This document describes how to reproduce the results and figures of the GMG V-cycle solver from the paper.

For a high-level overview, prerequisites, and directory layout, see the top-level [`README.md`](../README.md).

## Environment

Evaluated configuration:

- **Cerebras SDK 1.4.0** (release 2.5.0 — `cerebras-appliance==2.5.0`,
  `cerebras-sdk==2.5.0`; SIF image `sdk-cbcore-202505010205-2-ef181f81.sif`),
  providing `cslc`, `cs_python`, `SdkCompiler`, and `SdkLauncher`
- **Python 3.8.17** (3.8+ required)
- Cerebras CS-3 appliance access for device runs (host-only mode needs no WSE)

### Setup

```bash
# 1. Activate the SDK's Python environment and install SDK packages
source /path/to/sdk_venv/bin/activate
pip install -r /path/to/sdk/req.txt

# 2. Install this artifact's host-side and plotting dependencies
pip install -r requirements.txt
```

`requirements.txt` in the project root pins `numpy==1.24.4`, `matplotlib==3.7.5`,
`scipy==1.10.1`, `pandas==2.0.3`, and (optional) `numba==0.58.1`, all compatible
with the SDK 1.4.0 environment. See the top-level README's **Prerequisites**
section for the per-package breakdown.

## Running the Solver

All commands are issued from the `csl_gmg_with_conv/` directory.

```bash
# Device run (compile + execute on WSE-3)
python compile_and_run_wse3.py --only-device
```

Problem sizes and V-cycle parameters are configured in the `configs` list inside
`compile_and_run_wse3.py`. See the top-level README for the parameter schema.

Each run writes to `out_dir_S{size}x_L{levels}_M{max_ite}_P{pre}_P{post}_B{bottom}/`
containing `response.txt` (timing, convergence, memory) and compiled artifacts.


## Reproducing Paper Figures

After all configured runs have completed:

```bash
cd plots/
bash GENERATEFIGURES.sh
```

This aggregates per-config `response.txt` files into `all_responses_*.txt`
and invokes the individual plotting scripts. Individual figures can also be
generated directly; see the top-level README for the per-script commands.

## CSL Compilation Configuration

Compilation is driven by `SdkCompiler` (from `cerebras.sdk.client`) inside
`compile_and_run_wse3.py`. No manual `cslc` invocation is required — the script
assembles the flag string and submits sources (`./src`) to the appliance. The
effective compile command for a run of size `N` with `L` multigrid levels is:

```
--arch=wse3
--fabric-dims=762,1172
--fabric-offsets=4,1
--params=width:N,height:N,MAX_ZDIM:N,LEVELS:L
--params=BLOCK_SIZE:B
--memcpy
--channels=C
--width-west-buf=0
--width-east-buf=0
-o out_vcycle{N}
--llvm-option=--inline-threshold=256
--llvm-option=--unroll-threshold=256
```

### Flag reference

| Flag | Value | Meaning |
|---|---|---|
| `--arch` | `wse3` | Target Cerebras WSE-3 architecture |
| `--fabric-dims` | `762,1172` | Full WSE-3 fabric size (width × height) |
| `--fabric-offsets` | `4,1` | Origin of the compute rectangle inside the fabric (leaves halo for IO routing) |
| `--params=width,height,MAX_ZDIM` | `N,N,N` | Grid dimensions; `MAX_ZDIM` equals `N` (pencil decomposition, full z per PE) |
| `--params=LEVELS` | `L` | Number of multigrid levels for this run |
| `--params=BLOCK_SIZE` | `B` | Per-PE block size, auto-mapped from `N` via `{4:4, 8:8, 16:16, 32:32, 64:64, 128:128, 256:256, 512:256}` |
| `--memcpy` | — | Enable host ↔ device memcpy runtime (required by `SdkLauncher`) |
| `--channels` | `C` | DMA channels into the compute rectangle. Mapping: `C = N` when `N ≤ 16`, else `C = 16` |
| `--width-west-buf`, `--width-east-buf` | `0` | No east/west memcpy staging buffers (north/south routing only) |
| `-o` | `out_vcycle{N}` | Compiled artifact name |
| `--llvm-option=--inline-threshold` | `256` | LLVM inliner threshold (speed over code size) |
| `--llvm-option=--unroll-threshold` | `256` | LLVM loop-unroll threshold |

The exact command used for each run is recorded at the top of every
`build/out_dir_*/response.txt` under `Compile command:`.

### Artifact caching

`artifact_cache.json` (next to `compile_and_run_wse3.py`) maps each `out_path`
to a previously compiled artifact. On subsequent invocations, cached artifacts
are reused and compilation is skipped. Delete the JSON to force a clean
rebuild.

### SDK requirements

- Cerebras SDK with `cerebras.sdk.client` (`SdkCompiler`, `SdkLauncher`)
- Appliance/cluster access configured (SDK picks up credentials from the user
  environment — no extra flags in this repo)
- `SdkCompiler(disable_version_check=True)` is used so minor SDK version
  mismatches between the client and the appliance do not block compilation

## Expected Output

- `response.txt` per run with V-cycle timings, iteration counts, and final residual
- PNG/PDF figures in `plots/` corresponding to the paper's figures