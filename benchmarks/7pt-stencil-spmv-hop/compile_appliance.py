import json
from cerebras.sdk.client import SdkCompiler


import os
from cerebras.sdk.client import SdkCompiler, SdkLauncher

# --- Step 1: Compilation ---
print("Compiling job...")

# Define your compilation arguments exactly as they were in your bash script
# Note: We excluded the source file and '-o' from this string as they are passed as separate args
compile_flags = (
    f"--arch wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 "
    f"--params=width:64,height:64,MAX_ZDIM:64 --params=BLOCK_SIZE:64 "
    f"--params=C0_ID:0 --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 "
    f"--params=C4_ID:4 --params=C5_ID:5 --params=C6_ID:6 --params=C7_ID:7 "
    f"--params=C8_ID:8 -o=out "
    f"--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0"
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

    # Define the execution command
    # CRITICAL: Added '%CMADDR%' which the appliance replaces with the actual hardware address
    # run_cmd = (
    #     f"cs_python run.py -m=64 -n=64 -k=64 --level-id=0 "
    #     f"--latestlink out --channels=1 --width-west-buf=0 "
    #     f"--width-east-buf=0 --zDim=64 --run-only "
    #     f"--cmaddr %CMADDR%"
    # )

    # # Execute the command on the appliance
    # response = launcher.run(run_cmd)
    
    # print("Execution Output:")
    # print(response)

    # run_cmd1 = (
    #     f"cs_python run.py -m=64 -n=64 -k=64 --level-id=1 "
    #     f"--latestlink out --channels=1 --width-west-buf=0 "
    #     f"--width-east-buf=0 --zDim=64 --run-only "
    #     f"--cmaddr %CMADDR%"
    # )
    # response1 = launcher.run(run_cmd1)
    # print("Execution Output:")
    # print(response1)

    print("Running level 2...")
    run_cmd2 = (
        f"cs_python run.py -m=64 -n=64 -k=64 --level-id=2 "
        f"--latestlink out --channels=1 --width-west-buf=0 "
        f"--width-east-buf=0 --zDim=64 --run-only "
        f"--cmaddr %CMADDR%"
    )
    response2 = launcher.run(run_cmd2)
    print("Execution Output:")
    print(response2)

    # run_cmd3 = (
    #     f"cs_python run.py -m=64 -n=64 -k=64 --level-id=3 "
    #     f"--latestlink out --channels=1 --width-west-buf=0 "
    #     f"--width-east-buf=0 --zDim=64 --run-only "
    #     f"--cmaddr %CMADDR%"
    # )
    # response3 = launcher.run(run_cmd3)
    # print("Execution Output:")
    # print(response3)

    # run_cmd4 = (
    #     f"cs_python run.py -m=64 -n=64 -k=64 --level-id=4 "
    #     f"--latestlink out --channels=1 --width-west-buf=0 "
    #     f"--width-east-buf=0 --zDim=64 --run-only "
    #     f"--cmaddr %CMADDR%"
    # )
    # response4 = launcher.run(run_cmd4)
    # print("Execution Output:")
    # print(response4)

    # run_cmd5 = (
    #     f"cs_python run.py -m=64 -n=64 -k=64 --level-id=5 "
    #     f"--latestlink out --channels=1 --width-west-buf=0 "
    #     f"--width-east-buf=0 --zDim=64 --run-only "
    #     f"--cmaddr %CMADDR%"
    # )
    # response5 = launcher.run(run_cmd5)
    # print("Execution Output:")
    # print(response5)

    # run_cmd6 = (
    #     f"cs_python run.py -m=64 -n=64 -k=64 --level-id=6 "
    #     f"--latestlink out --channels=1 --width-west-buf=0 "
    #     f"--width-east-buf=0 --zDim=64 --run-only "
    #     f"--cmaddr %CMADDR%"
    # )
    # response6 = launcher.run(run_cmd6)
    # print("Execution Output:")
    # print(response6)