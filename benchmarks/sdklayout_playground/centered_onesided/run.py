#!/usr/bin/env cs_python
# build_layouts.py
#
# Creates two compiled layouts side-by-side:
#  - centered layout: 100x100 base, + centered 50x50, + centered 25x25
#  - one-sided layout: 100x100 base, + 50x50 at (0,0), + 25x25 at (0,0)
#
# Try:
#   cs_python build_layouts.py --arch wse3
#   (then run the produced out_centered.* or out_onesided.* with SdkRuntime)

import argparse
from cerebras.geometry.geometry import IntVector, IntRectangle
from cerebras.sdk.runtime.sdkruntimepybind import (
    Color,
    Route,
    RoutingPosition,
    SdkLayout,
    SdkTarget,
    SdkRuntime,
    SimfabConfig,
    get_platform,
)

def make_platform(arch: str):
    cfg = SimfabConfig(dump_core=True)
    tgt = SdkTarget.WSE3 if arch == "wse3" else SdkTarget.WSE2
    return get_platform(None, cfg, tgt)

def rect(x, y, w, h):
    return IntRectangle(IntVector(x, y), IntVector(w, h))

def set_layer(layout_region, r: IntRectangle, value: int, iters: int):
    # Turn on the kernel (enabled=true) for a rectangular sub-region,
    # set common compute params (same for all layers).
    layout_region.set_param_range(r, "enabled", 1)  # Make sure this is bool, not int
    layout_region.set_param_range(r, "value",   value)
    layout_region.set_param_range(r, "iters",   iters)

def compile_and_demo(artifacts_prefix, platform):
    # Optional quick smoke-run, then stop. You can comment this out
    # if you only want the compile artifacts.
    rt = SdkRuntime(artifacts_prefix, platform, memcpy_required=False)
    rt.load(); rt.run(); 
    rt.stop()
    #     # snippet after rt.run()
    # result = rt.read_symbol(0, 0, 'acc', dtype='uint16')
    # print(f"PE(0,0) acc = {result}")

def build_centered(platform, iters=128, value=7):
    layout = SdkLayout(platform)

    # One region 100x100; we'll "carve" layers by param ranges.
    R = layout.create_code_region("./layer_kernel.csl", "centered_region", 10, 10)

    # Start by defaulting *all* PEs to disabled=false.
    R.set_param_all("enabled", 1)  # Ensure this is bool, not int
    R.set_param_all("value",   value)
    R.set_param_all("iters",   iters)

    # Layer1 = whole 100x100
    set_layer(R, rect(0, 0, 10, 10), value, iters)

    # Layer2 = centered 50x50 -> top-left (25,25)
    set_layer(R, rect(5, 5, 10, 10), value, iters)

    # Place anywhere on wafer (no ports/links needed here). For visibility:
    R.place(10, 10)

    return layout.compile(out_prefix="out")  # creates out_centered.*

def build_onesided(platform, iters=128, value=7):
    layout = SdkLayout(platform)

    R = layout.create_code_region("./layer_kernel.csl", "onesided_region", 20, 20)

    R.set_param_all("enabled", 1)  # Ensure this is bool, not int
    R.set_param_all("value",   value)
    R.set_param_all("iters",   iters)

    # Layer1 = whole 100x100
    set_layer(R, rect(0, 0, 10, 10), value, iters)

    # Layer2 = 50x50 anchored at top-left (0,0)
    set_layer(R, rect(0, 0, 5, 5), value, iters)

    # # Layer3 = 25x25 anchored at top-left (0,0)
    # set_layer(R, rect(0, 0, 2, 2), value, iters)

    # R.place(20, 10)
    return layout.compile(out_prefix="out")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["wse2", "wse3"], default="wse3")
    p.add_argument("--iters", type=int, default=128)
    p.add_argument("--value", type=int, default=7)
    args = p.parse_args()

    platform = make_platform(args.arch)

    # art_c = build_centered(platform, args.iters, args.value)
    art_o = build_onesided(platform, args.iters, args.value)

    # quick smoke runs (optional)
    # compile_and_demo(art_c, platform)
    compile_and_demo(art_o, platform)

