import json
import os
from cerebras.sdk.client import SdkCompiler
from cerebras.sdk.client import SdkLauncher

###############################################################################
#### COMPILE
###############################################################################

# Define the output path for artifacts

layout_file = "./src/layout_gmg_vcycle.csl"
# Compiler arguments
Compile_command = f"--arch=wse3 --fabric-dims=75,69 --fabric-offsets=4,1 --params=width:64,height:64,MAX_ZDIM:64,LEVELS:3 \
    --params=BLOCK_SIZE:64 --params=C0_ID:0 --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 \
    --params=C5_ID:5 --params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 --memcpy --channels=10 \
    --width-west-buf=0 --width-east-buf=0 -o out_vcycle --max-inlined-iterations=1000000"

Run_command = f"cs_python run_gmg_vcycle.py -m=64 -n=64 -k=64 --latestlink out_vcycle --channels=10 \
--width-west-buf=0 --width-east-buf=0 --zDim=64 --run-only --levels=3 --max-ite=1 "
out_path = "out_dir"
os.makedirs(out_path, exist_ok=True)

# Instantiate compiler
with SdkCompiler(resource_cpu=48000, resource_mem=128<<30, disable_version_check=True) as compiler:
    # Launch compile job
    artifact_path = compiler.compile(
        app_path=".",
        csl_main=layout_file,
        options=Compile_command,
        out_path=out_path,
    )

###############################################################################
#### RUNNING
###############################################################################

print(f"Artifact path: {artifact_path}")
# artifact_path contains the path to the compiled artifact.
# It will be transferred and extracted in the appliance.
# The extracted directory will be the working directory.
# Set simulator=False if running on CS system within appliance.
with SdkLauncher(artifact_path, simulator=False, disable_version_check=True) as launcher:

    # stage files
    files_to_stage = [
        "cmd_parser.py",
        "run_gmg_vcycle.py",
        "util.py",
    ]
    # stage the entire python_gmg directory to preserve directory structure
    files_to_stage.append("python_gmg")
    for file_to_stage in files_to_stage:
        launcher.stage(file_to_stage)
    response = launcher.run(
        "pwd",
        "ls",
        # "cat run_gmg_vcycle.py",
        "tar -xvf python_gmg.tar.gz",
    )
    print("Test response: ", response)

    # run now
    response = launcher.run(Run_command)
    print("Host code execution response: ", response)

    # cleanup
    # Fetch files from the appliance
    launcher.download_artifact("sim.log", f"./{out_path}/sim.log")
    # launcher.download_artifact("simfab_traces", f"./{out_path}")  # takes too long to download
    with open(f"./{out_path}/response.txt", "w") as f:
        f.write(response)
    os.rename("python_gmg.tar.gz", f"./{out_path}/python_gmg.tar.gz")
    os.rename("run_meta.json", f"./{out_path}/run_meta.json")
    # os.rename("wsjob-*.json", f"./{out_path}/wsjob-*.json")