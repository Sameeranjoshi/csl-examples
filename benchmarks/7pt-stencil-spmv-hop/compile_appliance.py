import json
from cerebras.sdk.client import SdkCompiler


import os
from cerebras.sdk.client import SdkCompiler, SdkLauncher

# params
levels = 2  # Adjust this value to your maximum number of levels desired
m = 4
n = 4
k = 4
zDim = 4
channels = 4
blockSize = 4
# --- Step 1: Compilation ---
print("Compiling job...")

# Define your compilation arguments exactly as they were in your bash script
# Note: We excluded the source file and '-o' from this string as they are passed as separate args
compile_flags = (
    f"--arch wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 "
    f"--params=width:{m},height:{n},MAX_ZDIM:{zDim} --params=BLOCK_SIZE:{blockSize} "
    f"--params=C0_ID:0 --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 "
    f"--params=C4_ID:4 --params=C5_ID:5 --params=C6_ID:6 --params=C7_ID:7 "
    f"--params=C8_ID:8 -o=out "
    f"--memcpy --channels={channels} --width-west-buf=0 --width-east-buf=0"
)

# Use context manager for SdkCompiler
# disable_version_check=True prevents errors if client/server versions differ slightly
with SdkCompiler(disable_version_check=True) as compiler:
    artifact_path = compiler.compile(
        ".",                 # Source directory (root)
        "./src/layout.csl",  # Entry file path relative to Source directory
        compile_flags,       # The flags defined above
        "."                  # Output directory for the artifact
    )
print(f"Compilation successful. Artifact saved to: {artifact_path}")


# --- Step 2: Execution on Hardware ---
print("Launching job on appliance (Real Machine)...")

# Initialize Launcher with the compiled artifact
# simulator=False ensures this runs on real hardware
with SdkLauncher(artifact_path, simulator=False, disable_version_check=True) as launcher:
    
    # Upload your local run.py to the appliance workspace
    launcher.stage("./run.py")
    launcher.stage("./cmd_parser.py")
    launcher.stage("./util.py")
    # Run a loop over levels

    for level_id in range(levels):
        print(f"Running level {level_id}...")
        run_cmd = (
            f"cs_python run.py -m={m} -n={n} -k={k} --level-id={level_id} "
            f"--latestlink out --channels={channels} --width-west-buf=0 "
            f"--width-east-buf=0 --zDim={zDim} --run-only "
            f"--cmaddr %CMADDR%"
        )
        try:
            response = launcher.run(run_cmd)
            print("Execution Output:")
            print(response)
        except Exception as e:
            print(f"WARNING: Run failed at level_id={level_id} with error: {e}")