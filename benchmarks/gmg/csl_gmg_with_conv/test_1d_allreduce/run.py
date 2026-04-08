#!/usr/bin/env python3
"""
1D allreduce timing test driver.

Runs rowReduceReverse at levels 0..NUM_LEVELS-1 on a W×2 grid,
measures per-level cycle count, prints results.

Usage: cs_python run.py --width=16 --levels=4 --n=4 --latestlink out_test16
"""
import argparse
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime, MemcpyDataType, MemcpyOrder
)


def oned_to_hwl_colmajor(width, height, num_words, flat, dtype):
    """Convert 1D col-major buffer to (H, W, L) array."""
    return flat.reshape(num_words, width, height).transpose(2, 1, 0).astype(dtype)


def hwl_2_oned_colmajor(height, width, num_words, hwl, dtype):
    return hwl.transpose(2, 1, 0).ravel().astype(dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, required=True, help="PE grid width")
    p.add_argument("--height", type=int, default=2)
    p.add_argument("--levels", type=int, required=True, help="Number of levels to test")
    p.add_argument("--n", type=int, default=4, help="Pencil/z length")
    p.add_argument("--latestlink", type=str, required=True)
    p.add_argument("--cmaddr", type=str, default="")
    args = p.parse_args()

    kwargs = {}
    if args.cmaddr:
        kwargs["cmaddr"] = args.cmaddr

    sim = SdkRuntime(args.latestlink, **kwargs)

    sym_data = sim.get_id("data")
    sym_results = sim.get_id("results")
    sym_timing = sim.get_id("timing")

    sim.load()
    sim.run()

    # Enable timer
    sim.launch("f_enable_timer", nonblock=False)

    # Run test: iterates internally through all levels
    sim.launch("f_run_test", np.int16(args.n), nonblock=False)

    # Collect timing data: NUM_LEVELS × 2 u32 words per PE
    n_words = args.levels * 2
    timing_buf = np.zeros(args.height * args.width * n_words, dtype=np.uint32)
    sim.memcpy_d2h(
        timing_buf, sym_timing, 0, 0, args.width, args.height, n_words,
        streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.COL_MAJOR, nonblock=False
    )
    timing_hwl = oned_to_hwl_colmajor(args.width, args.height, n_words, timing_buf, np.uint32)

    # Collect results (reduced values) per level
    results_buf = np.zeros(args.height * args.width * args.levels, dtype=np.float32)
    sim.memcpy_d2h(
        results_buf, sym_results, 0, 0, args.width, args.height, args.levels,
        streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.COL_MAJOR, nonblock=False
    )
    results_hwl = oned_to_hwl_colmajor(args.width, args.height, args.levels, results_buf, np.float32)

    sim.stop()

    # Parse timing for PE(0,0) — this is the window[0] receiver at every level
    # Each timing entry is 3 u16 words representing 48-bit cycle count
    pe00_timing = timing_hwl[0, 0, :]  # n_words = levels * 3
    print(f"\n{'='*70}")
    print(f"1D Row Reduce Reverse Timing (width={args.width}, n={args.n})")
    print(f"{'='*70}")
    print(f"{'Level':>6} {'stride':>8} {'window':>8} {'active':>8} {'cycles':>10} {'time(us)':>12} {'result[0]':>12}")
    print(f"{'-'*70}")

    for lv in range(args.levels):
        lo = int(pe00_timing[lv * 2])       # low 32 bits (packs w0 | w1<<16)
        hi = int(pe00_timing[lv * 2 + 1])   # high: just w2
        cycles = lo | (hi << 32)
        time_us = cycles / 850.0  # WSE clock at 850 MHz (or 875 — both used)
        stride = 2 ** lv
        window = 2 ** (lv + 1)
        active = 2  # by construction
        result = results_hwl[0, 0, lv]
        print(f"{lv:>6} {stride:>8} {window:>8} {active:>8} {cycles:>10} {time_us:>12.3f} {result:>12.1f}")

    print(f"{'='*70}\n")

    # Paper formula: cost = (z+1)*w + (z+c)*w  (comm + add)
    print("Paper formula comparison (cost per direction, c=2 assumed):")
    print(f"{'Level':>6} {'formula':>12} {'measured':>12} {'ratio':>8}")
    for lv in range(args.levels):
        z = args.n
        w = 2 ** (lv + 1)
        formula = (z + 1) * w + (z + 2) * w
        lo = int(pe00_timing[lv * 2])
        hi = int(pe00_timing[lv * 2 + 1])
        measured = lo | (hi << 32)
        ratio = measured / formula if formula > 0 else 0
        print(f"{lv:>6} {formula:>12} {measured:>12} {ratio:>8.2f}")


if __name__ == "__main__":
    main()
