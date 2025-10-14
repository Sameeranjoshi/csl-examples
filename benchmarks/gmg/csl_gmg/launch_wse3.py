import json
import os

from cerebras.sdk.client import SdkLauncher

# read the compile artifact_path from the json file
with open("out_vcycle/artifact.json", "r", encoding="utf8") as f:
    data = json.load(f)
    artifact_path = data["artifact_id"]

# artifact_path contains the path to the compiled artifact.
# It will be transferred and extracted in the appliance.
# The extracted directory will be the working directory.
# Set simulator=False if running on CS system within appliance.
with SdkLauncher(artifact_path, simulator=True) as launcher:

    # Transfer an additional file to the appliance,
    # then write contents to stdout on appliance
    launcher.stage("additional_artifact.txt")
    response = launcher.run(
        "echo \"ABOUT TO RUN IN THE APPLIANCE\"",
        "cat additional_artifact.txt",
    )
    print("Test response: ", response)

    # Run the original host code as-is on the appliance,
    # using the same cmd as when using the Singularity container
    response = launcher.run("cs_python run_gmg_vcycle.py -m=4 -n=4 -k=4 --latestlink out_vcycle --channels=4 \
--width-west-buf=13 --width-east-buf=13 --zDim=4 --run-only --levels=2 --max-ite=1 \
--pre-iter=3 --post-iter=3 --bottom-iter=10 --cmaddr %CMADDR%")
    print("Host code execution response: ", response)

    # Fetch files from the appliance
    launcher.download_artifact("sim.log", "./output_dir/sim.log")
