import sys
import json
import time
from cerebras.sdk.client import SdkCompiler


# cs_python ./run_gmg.py -m=8 -n=8 -k=8 --latestlink out --channels=1 \
# --width-west-buf=0 --width-east-buf=0 --zDim=8 --run-only --levels=3

out_path = "out"
layout_file = "./src/layout_gmg.csl"
ARGS=f"--arch=wse3 --fabric-dims=16,10 --fabric-offsets=4,1 --params=width:8,height:8,MAX_ZDIM:8 --params=BLOCK_SIZE:8 --params=C0_ID:0 --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 --params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000"

# Instantiate compiler
# 48 cores
# 64GB memory
with SdkCompiler(resource_cpu=48000, resource_mem=64<<30) as compiler:
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