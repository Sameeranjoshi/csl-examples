import sys
import json
import time
from cerebras.sdk.client import SdkCompiler
import os

# Define the output path for artifacts
# makedir if not exists
# out_path = "compile_out"
layout_file = "./src/layout_gmg_vcycle.csl"

# Compiler arguments
ARGS=f"--arch=wse3 --fabric-dims=37,15 --fabric-offsets=17,1 --params=width:4,height:4,MAX_ZDIM:4,LEVELS:2 \
    --params=BLOCK_SIZE:4 --params=C0_ID:0 --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 \
    --params=C5_ID:5 --params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 --memcpy --channels=4 \
    --width-west-buf=13 --width-east-buf=13 -o out_vcycle --max-inlined-iterations=1000000"


print("Start compiling: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)

# Instantiate compiler
with SdkCompiler(resource_cpu=48000, resource_mem=64<<30, disable_version_check=True) as compiler:
    # Launch compile job
    artifact_path = compiler.compile(
        app_path=".",
        csl_main=layout_file,
        options=ARGS,
        out_path=".",
    )

    print(f"Compilation successful! Artifact: {artifact_path}")

    with open(f"artifact_path.json", "w", encoding="utf-8") as f:
        json.dump({"artifact_path": artifact_path,}, f)

print("End compiling: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)
