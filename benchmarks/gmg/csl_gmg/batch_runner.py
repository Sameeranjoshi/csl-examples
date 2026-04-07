"""
Batch runner: fires one config group at a time, monitors jobs, cancels stale ones (>20min).
Usage: nohup python batch_runner.py &
"""
import os
import re
import subprocess
import sys
import time
import threading

# Ensure sdk_venv is active
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compile_and_run_wse3 import (
    process_on_device, get_out_path, run_check_memory_for_outputs, BUILD_DIR
)

MAX_JOB_DURATION_MINUTES = 20

# ── Config batches: (label, [(size, levels, max_ite, abs_tol, pre, post, bottom), ...]) ──
BATCHES = [
    ("6/6/100", [
        (4, 2, 100, 1e-2, 6, 6, 100),
        (8, 3, 100, 1e-2, 6, 6, 100),
        (16, 4, 100, 1e-2, 6, 6, 100),
        (32, 5, 100, 1e-2, 6, 6, 100),
        (64, 6, 100, 1e-2, 6, 6, 100),
        (128, 7, 100, 1e-2, 6, 6, 100),
        (256, 8, 100, 1e-2, 6, 6, 100),
        (512, 9, 100, 1e-2, 6, 6, 100),
    ]),
    ("4/4/100", [
        (4, 2, 100, 1e-2, 4, 4, 100),
        (8, 3, 100, 1e-2, 4, 4, 100),
        (16, 4, 100, 1e-2, 4, 4, 100),
        (32, 5, 100, 1e-2, 4, 4, 100),
        (64, 6, 100, 1e-2, 4, 4, 100),
        (128, 7, 100, 1e-2, 4, 4, 100),
        (256, 8, 100, 1e-2, 4, 4, 100),
        (512, 9, 100, 1e-2, 4, 4, 100),
    ]),
    ("4/4/6", [
        (4, 2, 100, 1e-2, 4, 4, 6),
        (8, 3, 100, 1e-2, 4, 4, 6),
        (16, 4, 100, 1e-2, 4, 4, 6),
        (32, 5, 100, 1e-2, 4, 4, 6),
        (64, 6, 100, 1e-2, 4, 4, 6),
        (128, 7, 100, 1e-2, 4, 4, 6),
        (256, 8, 100, 1e-2, 4, 4, 6),
        (512, 9, 100, 1e-2, 4, 4, 6),
    ]),
    ("6/6/6", [
        (4, 2, 100, 1e-2, 6, 6, 6),
        (8, 3, 100, 1e-2, 6, 6, 6),
        (16, 4, 100, 1e-2, 6, 6, 6),
        (32, 5, 100, 1e-2, 6, 6, 6),
        (64, 6, 100, 1e-2, 6, 6, 6),
        (128, 7, 100, 1e-2, 6, 6, 6),
        (256, 8, 100, 1e-2, 6, 6, 6),
        (512, 9, 100, 1e-2, 6, 6, 6),
    ]),
    ("6/6/6 shallow", [
        (16, 2, 100, 1e-2, 6, 6, 6),
        (32, 3, 100, 1e-2, 6, 6, 6),
        (64, 4, 100, 1e-2, 6, 6, 6),
        (128, 5, 100, 1e-2, 6, 6, 6),
        (256, 6, 100, 1e-2, 6, 6, 6),
        (512, 7, 100, 1e-2, 6, 6, 6),
    ]),
]


def get_running_jobs():
    """Get list of (job_id, status, duration_minutes) from csctl."""
    try:
        result = subprocess.run(
            ["csctl", "get", "jobs"],
            capture_output=True, text=True, timeout=30
        )
        jobs = []
        for line in result.stdout.strip().split("\n")[1:]:  # skip header
            # Only monitor our own jobs
            if "bricklib_dataflow" not in line and "bricklib-dataflow" not in line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            job_id = parts[1]  # NAME column (wsjob-...)
            # Find PHASE column — look for known phase values
            phase = ""
            dur_str = "0s"
            for i, p in enumerate(parts):
                if p in ("RUNNING", "QUEUED", "COMPLETED", "FAILED", "CANCELLED", "STARTING"):
                    phase = p
                if re.match(r"^\d+[hms]", p) and "wsjob" not in p:
                    dur_str = p
            minutes = parse_duration_minutes(dur_str)
            if phase:
                jobs.append((job_id, phase, minutes))
        return jobs
    except Exception as e:
        print(f"  [monitor] csctl failed: {e}")
        return []


def parse_duration_minutes(dur_str):
    """Parse duration like '5m30s', '1h2m3s', '45s' to minutes."""
    hours = re.search(r"(\d+)h", dur_str)
    mins = re.search(r"(\d+)m", dur_str)
    secs = re.search(r"(\d+)s", dur_str)
    total = 0
    if hours:
        total += int(hours.group(1)) * 60
    if mins:
        total += int(mins.group(1))
    if secs:
        total += int(secs.group(1)) / 60.0
    return total


def cancel_job(job_id):
    """Cancel a job via csctl."""
    try:
        result = subprocess.run(
            ["csctl", "cancel", "job", job_id],
            capture_output=True, text=True, timeout=30
        )
        print(f"  [monitor] Cancelled {job_id}: {result.stdout.strip()}")
    except Exception as e:
        print(f"  [monitor] Failed to cancel {job_id}: {e}")


def wait_for_batch(out_paths, label):
    """Wait until all out_paths have 'Run output:' in response.txt, monitoring for stale jobs."""
    print(f"  [{label}] Waiting for {len(out_paths)} jobs to complete...")
    start = time.time()
    poll_interval = 30  # seconds

    while True:
        time.sleep(poll_interval)
        elapsed = (time.time() - start) / 60

        # Check completion: does each response.txt have run output?
        done = 0
        for p in out_paths:
            resp = os.path.join(p, "response.txt")
            if os.path.exists(resp):
                try:
                    with open(resp) as f:
                        content = f.read()
                    if "Run output:" in content or "Run time (s):" in content:
                        done += 1
                except:
                    pass

        print(f"  [{label}] {done}/{len(out_paths)} done ({elapsed:.1f} min elapsed)")

        if done >= len(out_paths):
            print(f"  [{label}] All jobs completed!")
            return True

        # Monitor for stale jobs (> MAX_JOB_DURATION_MINUTES)
        jobs = get_running_jobs()
        for job_id, status, dur_min in jobs:
            if status.lower() in ("running", "starting") and dur_min > MAX_JOB_DURATION_MINUTES:
                print(f"  [{label}] Job {job_id} running {dur_min:.1f} min > {MAX_JOB_DURATION_MINUTES} min limit — cancelling")
                cancel_job(job_id)

        # Safety: if we've been waiting > 60 min total for this batch, bail
        if elapsed > 60:
            print(f"  [{label}] Batch timeout after 60 min. {done}/{len(out_paths)} completed.")
            return False


def run_batch(label, problems):
    """Fire all problems in a batch and wait for completion."""
    print(f"\n{'='*70}")
    print(f"BATCH: {label} ({len(problems)} problems)")
    print(f"{'='*70}")

    out_paths = []
    for size, levels, max_ite, abs_tol, pre, post, bottom in problems:
        out_path = get_out_path(size, levels, max_ite, pre, post, bottom)
        out_paths.append(out_path)

        # Skip if already has run output
        resp = os.path.join(out_path, "response.txt")
        if os.path.exists(resp):
            with open(resp) as f:
                content = f.read()
            if "Run output:" in content:
                print(f"  {size}^3: already has results, skipping")
                continue

        channels = size if size <= 16 else 16
        try:
            process_on_device(size, levels, channels, max_ite, abs_tol, pre, post, bottom)
        except Exception as e:
            print(f"  {size}^3: FAILED to submit: {e}")

    # Wait for all to finish
    success = wait_for_batch(out_paths, label)

    # Run memory check on this batch
    print(f"  [{label}] Running memory checks...")
    try:
        run_check_memory_for_outputs(out_paths=out_paths)
    except Exception as e:
        print(f"  [{label}] Memory check failed: {e}")

    # Report results
    print(f"\n  [{label}] Results summary:")
    for out_path in out_paths:
        resp = os.path.join(out_path, "response.txt")
        basename = os.path.basename(out_path)
        if os.path.exists(resp):
            with open(resp) as f:
                content = f.read()
            conv = re.search(r"Converged: (Yes|No)", content)
            iters = re.search(r"Device iterations\s*:\s*(\d+)", content)
            rho = re.search(r"Device final \|rho\|_inf\s*:\s*([\d.eE+-]+)", content)
            avg = re.search(r"Avg V-cycle time \(no conv\)\s*:\s*([\d.]+)us", content)
            conv_s = conv.group(1) if conv else "?"
            iter_s = iters.group(1) if iters else "?"
            rho_s = rho.group(1) if rho else "?"
            avg_s = avg.group(1) if avg else "?"
            print(f"    {basename}: conv={conv_s} iters={iter_s} rho={rho_s} avg_vcycle={avg_s}us")
        else:
            print(f"    {basename}: NO RESPONSE")

    return success


def main():
    print(f"Batch runner started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Max job duration: {MAX_JOB_DURATION_MINUTES} min")
    print(f"{len(BATCHES)} config batches, {sum(len(b[1]) for b in BATCHES)} total problems")

    for i, (label, problems) in enumerate(BATCHES):
        print(f"\n[{i+1}/{len(BATCHES)}] Starting batch: {label}")
        run_batch(label, problems)

    print(f"\n{'='*70}")
    print(f"All batches done at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
