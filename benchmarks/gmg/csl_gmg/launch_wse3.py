import json
import os

from cerebras.sdk.client import SdkLauncher

# read the compile artifact_path from the json file
with open("artifact_path.json", "r", encoding="utf8") as f:
    data = json.load(f)
    artifact_path = data["artifact_path"]

print(f"Artifact path: {artifact_path}")
# artifact_path contains the path to the compiled artifact.
# It will be transferred and extracted in the appliance.
# The extracted directory will be the working directory.
# Set simulator=False if running on CS system within appliance.
with SdkLauncher(artifact_path, simulator=True, disable_version_check=True) as launcher:

    # Transfer an additional file to the appliance,
    # then write contents to stdout on appliance
    files_to_stage = [
        "check_memory_usage.sh",
        "clean.sh",
        "cmd_parser.py",
        "commands_vcycle_wse3.sh",
        "run_gmg_vcycle.py",
        "util.py",
    ]
    # Stage the entire python_gmg directory to preserve directory structure
    files_to_stage.append("python_gmg")
    
    # Stage all files
    for file_to_stage in files_to_stage:
        launcher.stage(file_to_stage)
    response = launcher.run(
        "echo \"ABOUT TO RUN IN THE APPLIANCE\"",
        "cat run_gmg_vcycle.py",
        "tar -xvf python_gmg.tar.gz",
        "pwd",
        "ls",
    )
    print("Test response: ", response)

    # Run the original host code as-is on the appliance,
    # using the same cmd as when using the Singularity container
    # print the working directory
    # print pwd
    response = launcher.run("cs_python run_gmg_vcycle.py -m=4 -n=4 -k=4 --latestlink out_vcycle --channels=4 \
--width-west-buf=13 --width-east-buf=13 --zDim=4 --run-only --levels=2 --max-ite=1 \
--pre-iter=3 --post-iter=3 --bottom-iter=10")
    print("Host code execution response: ", response)

    # Fetch files from the appliance
    launcher.download_artifact("sim.log", "./output_dir/sim.log")
