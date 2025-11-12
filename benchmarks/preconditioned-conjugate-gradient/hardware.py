
#####
import json
import os
import time
from cerebras.sdk.client import SdkCompiler
from cerebras.sdk.client import SdkLauncher

###############################################################################
#### COMPILE
###############################################################################

# Define the output path for artifacts

layout_file = "./src/layout_pcg.csl"
# Compiler arguments
Compile_command = f"--arch wse3 --fabric-dims=300,300 --fabric-offsets=4,1 \
--params=width:200,height:200,MAX_ZDIM:200 --params=BLOCK_SIZE:200 --params=C0_ID:0 \
--params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
--params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000"

Run_command = f"cs_python ./run_pcg.py -m=200 -n=200 -k=200 --latestlink out --channels=1 \
--width-west-buf=0 --width-east-buf=0 --zDim=200 --run-only --max-ite=2 "

out_path = "out_dir"
os.makedirs(out_path, exist_ok=True)

# Instantiate compiler and time compile step
compile_start_time = time.time()
with SdkCompiler(resource_cpu=48000, resource_mem=128<<30, disable_version_check=True) as compiler:
    # Launch compile job
    artifact_path = compiler.compile(
        app_path=".",
        csl_main=layout_file,
        options=Compile_command,
        out_path=out_path,
    )
compile_end_time = time.time()
compile_time_sec = compile_end_time - compile_start_time
print(f"Compile Time: {compile_time_sec:.2f} seconds")

###############################################################################
#### RUNNING
###############################################################################

print(f"Artifact path: {artifact_path}")
# artifact_path contains the path to the compiled artifact.
# It will be transferred and extracted in the appliance.
# The extracted directory will be the working directory.
# Set simulator=False if running on CS system within appliance.

run_start_time = time.time()
with SdkLauncher(artifact_path, simulator=False, disable_version_check=True) as launcher:

    # stage files
    files_to_stage = [
        "cmd_parser.py",
        "run_pcg.py",
        "util.py",
        "pcg.py",
    ]
    # stage the entire python_gmg directory to preserve directory structure
    for file_to_stage in files_to_stage:
        launcher.stage(file_to_stage)
    response = launcher.run(
        "pwd",
        "ls",
    )
    print("Test response: ", response)

    # run now
    run_exec_start = time.time()
    response = launcher.run(Run_command)
    run_exec_end = time.time()
    run_exec_time_sec = run_exec_end - run_exec_start
    print("Host code execution response: ", response)
    print(f"Runtime (host code execution only): {run_exec_time_sec:.2f} seconds")

    # cleanup
    # Fetch files from the appliance
    # launcher.download_artifact("sim.log", f"./{out_path}/out/sim.log")
    # launcher.download_artifact("simfab_traces", f"./{out_path}")  # takes too long to download
    with open(f"./{out_path}/response.txt", "w") as f:
        f.write(response)
    os.rename("run_meta.json", f"./{out_path}/run_meta.json")
    # os.rename("wsjob-*.json", f"./{out_path}/wsjob-*.json")

run_end_time = time.time()
total_runtime_sec = run_end_time - run_start_time
print(f"Total appliance run section time (including staging, setup, etc): {total_runtime_sec:.2f} seconds")