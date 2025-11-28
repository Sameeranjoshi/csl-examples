import json
import os
import time
from cerebras.sdk.client import SdkCompiler
from cerebras.sdk.client import SdkLauncher
import glob
import shutil
import json
import logging
from cerebras.appliance import logger
# logging.basicConfig(level=logging.INFO)

###############################################################################
#### Parameters
###############################################################################

import argparse

# Parse command line arguments for script parameters
parser = argparse.ArgumentParser(description="Compile and run GMG V-cycle for WSE3.")
parser.add_argument('--size', type=int, default=8, help='Grid size (default: 8)')
parser.add_argument('--levels', type=int, default=2, help='Number of levels (default: 2)')
parser.add_argument('--channels', type=int, default=5, help='Number of channels (default: 5)')
args = parser.parse_args()
size = args.size
levels = args.levels
channels = args.channels


###############################################################################
# More or less doesn't change
out_path = f"out_dir_{size}x{size}x{size}_L{levels}_C{channels}"
os.makedirs(out_path, exist_ok=True)
###############################################################################
layout_file = "./src/layout_gmg_vcycle.csl"
if size > 256:
    BSIZE = size // 2           # data segment
else:
    BSIZE = size

INLINE_THRESHOLD = 256    # inline always good for speed, # code segment

Compile_command = f"--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 --params=width:{size},height:{size},MAX_ZDIM:{size},LEVELS:{levels} \
    --params=BLOCK_SIZE:{BSIZE} --memcpy --channels={channels} \
    --width-west-buf=0 --width-east-buf=0 -o out_vcycle \
    --llvm-option=--inline-threshold={INLINE_THRESHOLD} --llvm-option=--unroll-threshold=256"

Run_command = f"cs_python run_gmg_vcycle.py -m={size} -n={size} -k={size} --latestlink out_vcycle --channels={channels} \
--width-west-buf=0 --width-east-buf=0 --zDim={size} --run-only --levels={levels} --max-ite=200 --blockSize={BSIZE} --bottom-iter=1 --cmaddr %CMADDR%"


# Run_command1 = f"cs_python run_gmg_vcycle.py -m={size} -n={size} -k={size} --latestlink out_vcycle --channels={channels} \
# --width-west-buf=0 --width-east-buf=0 --zDim={size} --run-only --levels={levels} --max-ite=200 --blockSize={BSIZE} --bottom-iter=50 --cmaddr %CMADDR%"


# Run_command2 = f"cs_python run_gmg_vcycle.py -m={size} -n={size} -k={size} --latestlink out_vcycle --channels={channels} \
# --width-west-buf=0 --width-east-buf=0 --zDim={size} --run-only --levels={levels} --max-ite=200 --blockSize={BSIZE} --bottom-iter=50 --cmaddr %CMADDR%"



###############################################################################
print(f"Compile command: {Compile_command}")
print(f"Run command: {Run_command}")
# write into reponse.txt file
with open(f"./{out_path}/response.txt", "w") as f:
    f.write(f"########################################################\n")
    f.write(f"Parameters: size={size}, levels={levels}, channels={channels}\n")
    f.write(f"Compile command: {Compile_command}\n")
    f.write(f"Run command: {Run_command}\n")
    f.write(f"########################################################\n")
###############################################################################
#### COMPILE
###############################################################################
# Time the compilation process
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

# ###############################################################################
# #### RUNNING
# ###############################################################################
print(f"Artifact path: {artifact_path}")

with SdkLauncher(artifact_path, simulator=False, disable_version_check=True) as launcher:

    # stage files
    files_to_stage = [
        "cmd_parser.py",
        "run_gmg_vcycle.py",
        "util.py", 
        "python_gmg/gmgoscar.py",
    ]
    # stage the entire python_gmg directory to preserve directory structure
    # files_to_stage.append("python_gmg")
    for file_to_stage in files_to_stage:
        launcher.stage(file_to_stage)

    print("Staged files on appliance")
    print("printing inside the appliance ----------------------------------------")
    response = launcher.run(
        "ls",
        # "tar -xvf python_gmg.tar.gz",
    )
    print("Test response: ", response)

    print("Running host code on appliance ----------------------------------------")
    run_start = time.time()
    response = launcher.run(Run_command)
    # response1 = launcher.run(Run_command1)
    # response2 = launcher.run(Run_command2)
    run_end = time.time()
    print(response)
    # print(response1)
    # print(response2)
    print("Run completed on appliance ----------------------------------------")
    print("Copying data from appliance ----------------------------------------")
    # launcher.download_artifact("../sim.log", f"./{out_path}/sim.log")
    # launcher.download_artifact("simfab_traces", f"./{out_path}")  # takes too long to download
    with open(f"./{out_path}/response.txt", "w") as f:
        f.write(response)
        # f.write(response1)
        # f.write(response2)

    # os.rename("python_gmg.tar.gz", f"./{out_path}/python_gmg.tar.gz")
    # os.rename("run_meta.json", f"./{out_path}/run_meta.json")
    # for file in glob.glob("wsjob-*.json"):
    #     shutil.move(file, f"./{out_path}/")

    print("Done ----------------------------------------")
    # Time
    run_duration = run_end - run_start

###############################################################################
#### PRINT AND SAVE TIMING RESULTS
###############################################################################
print(f"\nCOMPILE TIME: {compile_duration:.3f} seconds")
print(f"RUN TIME: {run_duration:.3f} seconds")
with open(f"./{out_path}/response.txt", "a") as response_file:
    response_file.write(f"\nCompile time (s): {compile_duration:.6f}\n")
    response_file.write(f"Run time (s): {run_duration:.6f}\n")
