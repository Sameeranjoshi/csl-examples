import sys
import json
import time
from cerebras.sdk.client import SdkCompiler
import os


out_path = "out_vcycle"
layout_file = "./src/layout_gmg_vcycle.csl"
ARGS=f"--arch=wse3 --fabric-dims=37,15 --fabric-offsets=17,1 --params=width:4,height:4,MAX_ZDIM:4,LEVELS:2 \
    --params=BLOCK_SIZE:4 --params=C0_ID:0 --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 \
    --params=C5_ID:5 --params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 --memcpy --channels=4 \
    --width-west-buf=13 --width-east-buf=13 --max-inlined-iterations=1000000"


print("Start compiling: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)
# Instantiate compiler
with SdkCompiler(resource_cpu=48000, resource_mem=64<<30, disable_version_check=True) as compiler:
    # Launch compile job
    artifact_id = compiler.compile(
        app_path=".",
        csl_main=layout_file,
        options=ARGS,
        out_path=out_path,
    )

    # Write the artifact_id to a JSON file
    with open(f"{out_path}/artifact.json", "w", encoding="utf-8") as f:
        json.dump({"artifact_id": artifact_id,}, f)

print("End compiling: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)
