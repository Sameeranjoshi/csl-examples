"""
Compile and run GMG V-cycle for WSE3
Simple port of commands_vcycle_wse3.sh with flag support
Supports multiple problem sizes with artifact caching
"""

import json
import os
import time
from cerebras.sdk.client import SdkCompiler
from cerebras.sdk.client import SdkLauncher
import glob
import shutil
import logging
import argparse
from cerebras.appliance import logger
# logging.basicConfig(level=logging.INFO)

# Cache file for storing artifact paths
ARTIFACT_CACHE_FILE = "artifact_cache.json"

###############################################################################
# Build commands (similar to shell script)
###############################################################################

def load_artifact_cache():
    """
    Loads the artifact cache from file. Returns a dictionary mapping out_path to list of artifact_paths.
    """
    if os.path.exists(ARTIFACT_CACHE_FILE):
        try:
            with open(ARTIFACT_CACHE_FILE, "r") as f:
                cache = json.load(f)
                # Convert to list format if old format (single artifact_path)
                # Use list() to avoid RuntimeError if cache is modified during iteration
                for key in list(cache.keys()):
                    if isinstance(cache[key], str):
                        cache[key] = [cache[key]]
                return cache
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load artifact cache: {e}")
            return {}
    return {}

def save_artifact_cache(cache):
    """
    Saves the artifact cache to file.
    """
    with open(ARTIFACT_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def get_cached_artifact(out_path):
    """
    Checks if there's a cached artifact for the given out_path.
    Returns the artifact_path if found and file exists, None otherwise.
    """
    cache = load_artifact_cache()
    if out_path in cache:
        # Check all artifact paths for this out_path
        for artifact_path in cache[out_path]:
            if os.path.exists(artifact_path):
                print(f"Found cached artifact for {out_path}: {artifact_path}")
                return artifact_path
        # If none exist, remove from cache
        del cache[out_path]
        save_artifact_cache(cache)
    return None

def add_artifact_to_cache(out_path, artifact_path):
    """
    Adds an artifact_path to the cache for the given out_path.
    """
    cache = load_artifact_cache()
    if out_path not in cache:
        cache[out_path] = []
    if artifact_path not in cache[out_path]:
        cache[out_path].append(artifact_path)
    save_artifact_cache(cache)

def write_run_info(out_path, size, levels, channels, max_ite, pre_iter, post_iter, bottom_iter, Compile_command, Run_command):
    """
    Writes compile and run information into a response.txt file in the specified output directory.
    """
    print(f"Compile command: {Compile_command}")
    print(f"Run command: {Run_command}")
    with open(f"./{out_path}/response.txt", "w") as f:
        f.write(f"########################################################\n")
        f.write(f"Parameters: size={size}, levels={levels}, channels={channels}\n")
        f.write(f"Run parameters: max_ite={max_ite}, pre_iter={pre_iter}, post_iter={post_iter}, bottom_iter={bottom_iter}\n")
        f.write(f"Fabric dims: 762,1172\n")
        f.write(f"Compile command: {Compile_command}\n")
        f.write(f"Run command: {Run_command}\n")
        f.write(f"########################################################\n")

# COMPILE FUNCTION
def compile_app(layout_file, Compile_command, out_path):
    """
    Compiles the app using the SdkCompiler and returns artifact_path and compile_duration.
    First checks cache, and only compiles if no valid cached artifact exists.
    """
    # Check cache first
    cached_artifact = get_cached_artifact(out_path)
    if cached_artifact is not None:
        print(f"Using cached artifact, skipping compilation for {out_path}")
        return cached_artifact, 0.0  # No compile time for cached artifacts
    
    print("Compiling...")
    compile_start = time.time()
    with SdkCompiler(disable_version_check=True) as compiler:
        artifact_path = compiler.compile(
            app_path=".",
            csl_main=layout_file,
            options=Compile_command,
            out_path=out_path,
        )
    compile_end = time.time()
    compile_duration = compile_end - compile_start
    
    # Save to cache
    add_artifact_to_cache(out_path, artifact_path)
    
    return artifact_path, compile_duration

# RUN FUNCTION
def run_on_appliance(artifact_path, out_path, Run_command):
    print(f"Artifact path: {artifact_path}")
    with SdkLauncher(artifact_path, simulator=False, disable_version_check=True) as launcher:

        print("Staging files on appliance ----------------------------------------")
        files_to_stage = [
            "cmd_parser.py",
            "run_gmg_vcycle.py",
            "util.py", 
            "python_gmg/gmgoscar.py",
        ]
        for file_to_stage in files_to_stage:
            launcher.stage(file_to_stage)
        print("Running host code on appliance ----------------------------------------")
        run_start = time.time()
        response = launcher.run(Run_command)
        run_end = time.time()
        print(response)
        print("Copying data from appliance ----------------------------------------")
        with open(f"./{out_path}/response.txt", "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write("Run output:\n")
            f.write(f"{'='*70}\n")
            f.write(response)
        print("Done ----------------------------------------")
        # Time
        run_duration = run_end - run_start

    return run_duration

def process_on_device(size, levels, channels, max_ite, abs_tolerance, pre_iter, post_iter, bottom_iter):
    """
    Process a single problem: compile (if needed) and run.
    Returns compile_duration and run_duration.
    """
    
    out_path = f"out_dir_S{size}x_L{levels}_M{max_ite}_P{pre_iter}_P{post_iter}_B{bottom_iter}"
    os.makedirs(out_path, exist_ok=True)
    layout_file = "./src/layout_gmg_vcycle.csl"
    bsizemap = {
        4: 4,
        8: 8,
        16: 16,
        32: 32,
        64: 64,
        128: 128, # Let's keep a sweet spot of totalsize/4, so 1/4 th size is block size.
        256: 256,
        512: 256,
    }
    BSIZE  = bsizemap[size]

    INLINE_THRESHOLD = 256    # inline always good for speed, # code segment
    artifact_name = f"out_vcycle{size}"

    # prepare commands
    Compile_command = (
        f"--arch=wse3 --fabric-dims=762,1172 "
        f"--fabric-offsets=4,1 "
        f"--params=width:{size},height:{size},MAX_ZDIM:{size},LEVELS:{levels} "
        f"--params=BLOCK_SIZE:{BSIZE} --memcpy --channels={channels} "
        f"--width-west-buf=0 --width-east-buf=0 -o {artifact_name} "
        f"--llvm-option=--inline-threshold={INLINE_THRESHOLD} "
        f"--llvm-option=--unroll-threshold=256"
    )

    Run_command = (
        f"cs_python run_gmg_vcycle.py -m={size} -n={size} -k={size} "
        f"--latestlink {artifact_name} --channels={channels} "
        f"--width-west-buf=0 --width-east-buf=0 --zDim={size} --run-only "
        f"--levels={levels} --max-ite={max_ite} --blockSize={BSIZE} "
        f"--pre-iter={pre_iter} --post-iter={post_iter} --bottom-iter={bottom_iter} "
        f"--tolerance={abs_tolerance} "
        f"--cmaddr %CMADDR%"
    )

    write_run_info(out_path, size, levels, channels, max_ite, pre_iter, post_iter, bottom_iter, Compile_command, Run_command)
    artifact_path, compile_duration = compile_app(layout_file, Compile_command, out_path)
    run_duration = run_on_appliance(artifact_path, out_path, Run_command)
    
    # Print and save timing results for this problem
    print(f"\nCOMPILE TIME: {compile_duration:.3f} seconds")
    print(f"RUN TIME: {run_duration:.3f} seconds")
    with open(f"./{out_path}/response.txt", "a") as response_file:
        response_file.write(f"\nCompile time (s): {compile_duration:.6f}\n")
        response_file.write(f"Run time (s): {run_duration:.6f}\n")
    
    return compile_duration, run_duration


import argparse
# from gmg import SimpleGMG
from python_gmg.gmgoscar import SimpleGMG as SimpleGMGOSCAR
import time


def process_on_host(size, num_levels, verbose, max_iterations, abs_tolerance, pre_iter, post_iter, bottom_iter):
    """Benchmark a single problem size"""
    print(f"\nProcessing on host {size}x{size}x{size} grid with {num_levels} levels...")
    solver = SimpleGMGOSCAR(size, size, size, num_levels, verbose, abs_tolerance, pre_iter, post_iter, bottom_iter)
    start_time = time.time()
    rho_max, iterations = solver.solve_iterative(max_iterations)
    reltol = solver.rel_tolerance
    total_time = time.time() - start_time
    return rho_max, iterations, total_time, reltol

def main():
    # add a cmd line option called --only-host, --only-device, --host-and-device
    parser = argparse.ArgumentParser(description='Geometric Multigrid Solver')
    parser.add_argument('--only-host', action='store_true', default=False, help='Process on host')
    parser.add_argument('--only-device', action='store_true', default=False, help='Process on device')
    parser.add_argument('--host-and-device', action='store_true', default=False, help='Process on host and device')
    args = parser.parse_args()

    # Require at least one of the options, otherwise print help and exit
    if not (args.only_host or args.only_device or args.host_and_device):
        parser.print_help()
        exit(1)
    
    # Test problems (size, levels, max_ite, pre_iter, post_iter, bottom_iter)
    # Format: (size, levels, max_ite, abs_tolerance, pre_iter, post_iter, bottom_iter)
    # NOTE: MANUALLY DELETE FOLDER IF THERE IS SOME CHANGES IN THE SOURCE CODE AS IT WILL SKIP COMPILATION DUE TO CACHING.
    problems = [
        #  (4, 2, 100, 1e-5, 6, 6, 100),   # Tiny problem
        #  (8, 3, 100, 1e-5, 6, 6, 100),   # Tiny problem
        #  (16, 4, 100, 1e-5, 6, 6, 100),   # Small problem
        #  (32, 5, 100, 1e-5, 6, 6, 100),   # Small problem
         (64, 6, 100, 1e-5, 6, 6, 10),   # Medium problem
        #  (128, 7, 100, 1e-5, 6, 6, 100),   # Large problem
        #  (256, 8, 100, 1e-5, 6, 6, 100),   # Very large
        #  (512, 9, 100, 1e-5, 6, 6, 100),   # Very large


        # OSCAR matching problems
    ]
    verbose = False    
    results = []
    
    # Process host runs sequentially (if needed)
    if args.only_host or args.host_and_device:
        for size, levels, max_ite, abs_tolerance, pre_iter, post_iter, bottom_iter in problems:
            try:
                rho_max, iterations, total_time, reltol = process_on_host(size, levels, verbose, max_ite, abs_tolerance, pre_iter, post_iter, bottom_iter)
                results.append((size, size, size, rho_max, abs_tolerance, iterations, total_time, reltol))
            except Exception as e:
                print(f"Failed for size={size}, levels={levels}: {e}")
                results.append((size, size, size, float('inf'), float('inf'), 0, 0, float('inf')))
        
                # Summary
        print("\n" + "=" * 70)
        print("Summary of all runs on HOST")
        print("=" * 70)
        header = (
            f"{'Grid Size':<15} | "
            f"{'|rho|_inf':>11} | "
            f"{'(Rel Tolerance)':>17} | "
            f"{'Absolute Tol.':>14} | "
            f"{'Iters':>7} | "
            f"{'Time (s)':>9} | "
        )
        print(header)
        print("-" * len(header))
        
        for problem_idx, (nx, ny, nz, rho_max, abs_tolerance, iterations, total_time, reltol) in enumerate(results):
            grid_size = f"{nx}x{ny}x{nz}"
            tolerance = reltol
            converged = "Yes" if rho_max < tolerance else "No"
            print(
                f"{grid_size:<15} | "
                f"{rho_max:>11.2e} | "
                f"{tolerance:>17.2e} | "
                f"{abs_tolerance:>14.2e} | "
                f"{iterations:>7} | "
                f"{total_time:>9.4f} | "
                f"{converged:>9}"
            )
    
    # Process device runs
    if args.only_device or args.host_and_device:
        for size, levels, max_ite, abs_tolerance, pre_iter, post_iter, bottom_iter in problems:
            if size > 8:
                channels = 8
            else:
                channels = 4
            try:
                process_on_device(size, levels, channels, max_ite, abs_tolerance, pre_iter, post_iter, bottom_iter)
            except Exception as e:
                print(f"Failed for size={size}, levels={levels}: {e}")


if __name__ == "__main__":
    main()
