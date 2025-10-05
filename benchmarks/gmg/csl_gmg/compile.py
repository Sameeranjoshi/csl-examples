import sys
import json
import time
from cerebras.sdk.client import SdkCompiler
import os


# ABS_BENCHMARKS_ROOT = "/cloud_anl_homedir/sameeranj-cloud/AMGCerebras/benchmarks"
# print CSL_IMPORT_PATH ENV variable

out_path = "out"
layout_file = "src/layout_gmg.csl"
ARGS = (
    f"--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 "
    f"--params=width:8,height:8,MAX_ZDIM:8 --params=BLOCK_SIZE:8 --params=C0_ID:0 --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 --params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 "
    f"--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0"
    # f"--import-path '{IMPORT_PATHS}'"
)

# Instantiate compiler
# 16 cores
# 16GB memory = 16 << 30 = 16 *10^30(GB)
# disable_version_check=True is added because of some internal bug, it converts the version into warning.
with SdkCompiler(resource_cpu=16000, resource_mem=16<<30, disable_version_check=True) as compiler:
    # Launch compile job
    artifact_id = compiler.compile(
        app_path="src",
        csl_main=layout_file,
        options=ARGS,
        out_path=out_path,
    )

    # Write the artifact_id to a JSON file
    with open(f"{out_path}/artifact.json", "w", encoding="utf-8") as f:
        json.dump({"artifact_id": artifact_id,}, f)

print("End compiling: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)