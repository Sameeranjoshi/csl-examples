import re
import pandas as pd

def parse_gmg_output_final(text: str):
    # --- Extract metadata ---
    meta_pattern = re.compile(
        r"Grid size:\s*([\dxX]+).*?"
        r"Levels:\s*(\d+).*?"
        r"Initial residual at level 0:\s*([\d.eE+-]+).*?"
        r"Residual\s*=\s*([\d.eE+-]+).*?"
        r"Time\s*=\s*([\d.eE+-]+)\s*s.*?"   # host time
        r"Device rho_up:\s*([\d.eE+-]+).*?"
        r"Total V-cycle time:.*?\(\s*([\d.]+)\s*us\)",
        re.DOTALL
    )
    meta_match = meta_pattern.search(text)
    metadata = {}
    if meta_match:
        metadata = {
            "Grid size": meta_match.group(1),
            "Levels": int(meta_match.group(2)),
            "Initial residual": float(meta_match.group(3)),
            "Host residual": float(meta_match.group(4)),
            "Host time (s)": float(meta_match.group(5)),
            "Device residual": float(meta_match.group(6)),
            "Device total time (us)": float(meta_match.group(7))
        }

    # --- Extract per-level operator timings ---
    level_pattern = re.compile(
        r"Level\s+(\d+):\s*"
        r".*?smooth\s*:\s*\[\s*\d+\s*cycles,\s*([\d.]+)\s*us\].*?"
        r"residual:\s*\[\s*\d+\s*cycles,\s*([\d.]+)\s*us\].*?"
        r"restriction:\s*\[\s*\d+\s*cycles,\s*([\d.]+)\s*us\].*?"
        r"interpolation:\s*\[\s*\d+\s*cycles,\s*([\d.]+)\s*us\].*?"
        r"total:\s*\[\s*\d+\s*cycles,\s*([\d.]+)\s*us\]",
        re.DOTALL
    )

    records = []
    levels = []
    for m in level_pattern.finditer(text):
        lvl, smooth, residual, restriction, interp, total = m.groups()
        levels.append(int(lvl))
        records.append({
            "smooth": float(smooth),
            "residual": float(residual),
            "restriction": float(restriction),
            "interpolation": float(interp),
            "total": float(total),
        })

    # --- Build DataFrame (operators as rows, levels as columns) ---
    df = pd.DataFrame(records, index=[f"Level {i}" for i in levels]).T
    df.columns = [f"Level {i}" for i in levels]

    # --- Compute total time per operator (sum over levels) ---
    df["Total (us)"] = df.sum(axis=1)

    # --- Convert host time to µs ---
    host_time_us = metadata.get("Host time (s)", 0.0) * 1e6
    device_time_us = metadata.get("Device total time (us)", 0.0)

    # --- Add Host Time and Speedup only in last row ---
    host_col = [""] * len(df)
    host_col[-1] = f"{host_time_us:.3f}"

    speedup_col = [""] * len(df)
    if host_time_us > 0 and device_time_us > 0:
        speedup = host_time_us / device_time_us
        speedup_col[-1] = f"{speedup:.2f}×"

    df["Host Time (us)"] = host_col
    df["Speedup"] = speedup_col

    # --- Add operator column for readability ---
    df.insert(0, "Operator", df.index)
    df.reset_index(drop=True, inplace=True)

    # --- Clean header string ---
    header_str = (
        f"Grid: {metadata.get('Grid size','?')}  —  "
        f"Levels={metadata.get('Levels','?')}  —  "
        f"Init Resid={metadata.get('Initial residual','?'):.3e}  —  "
        f"Host Resid={metadata.get('Host residual','?'):.3e}  —  "
        f"Device Resid={metadata.get('Device residual','?'):.3e}"
    )

    return metadata, header_str, df


# --- Example usage ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <gmg_output_file.txt>")
        sys.exit(1)
    infile = sys.argv[1]
    with open(infile) as f:
        text = f.read()

    meta, header, df = parse_gmg_output_final(text)

    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    print(df.to_string(index=False))

    # Save to Excel
    with pd.ExcelWriter("gmg_clean_table.xlsx") as writer:
        pd.DataFrame([[header]]).to_excel(writer, header=False, index=False, startrow=0)
        df.to_excel(writer, index=False, startrow=2)
    print("\n✅ Clean summary with speedup saved to gmg_clean_table.xlsx")
