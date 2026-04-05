# Why f_recv Time Goes Up at Coarser Levels (and Send Stays Flat)

## Observed Data

From the 7-pt Stencil table (us[cycles] per level):

| level | spmv_send | spmv_recv |
|-------|-----------|-----------|
| 0     | 186.9     | 130.0     |
| 1     | 138.7     | 92.5      |
| 2     | 120.1     | 89.1      |
| 3     | 51.1      | **93.5**  ← recv goes up |
| 4     | 49.1      | **124.8** |
| 5     | 48.8      | **222.1** |
| 6     | 22.5      | **191.4** |

- **Send** decreases (or stays flat) as level increases: smaller grid ⇒ less data to send.
- **Recv** decreases initially then **increases sharply** after a trigger level (around 2–3).

## 1. Why Recv Goes Up: More “Forwarding” PEs and Single Microthread Serialization

### Hop-based layout (factor = 2^level)

- `factor = 2^level` and `setHopBasedParams(hops, factor, width, height)` set:
  - **Active PEs:** `(px % factor == 0) and (py % factor == 0)` ⇒ 1/factor² of PEs do real work.
  - **Inactive “void” PEs:** the rest. Among these:
    - **inactive_horizontal:** void PEs in interior columns with no active PE in that column (`px % factor != 0`, not first/last column).
    - **inactive_vertical:** void PEs in interior rows with no active PE in that row.

So as **level increases**, **factor** grows ⇒ **fewer active PEs**, **more void PEs**, and **more PEs with inactive_horizontal or inactive_vertical** (entire columns/rows with no active PE).

### Two recv paths in `f_recv`

1. **Active path (receive into buffer)**  
   - `load_to_dsr(dest_dsr_recv, mem_*_buf_dsd);`  
   - `load_to_dsr(src1_dsr_recv, fab_recv_*_wdsd, .{.async=true, .activate = f_recv});`  
   - Continuation: **f_recv** (default microthread = queue ID of first fabric operand).

2. **Forward path (inactive_horizontal / inactive_vertical)**  
   - “Forward” data: e.g. west→east via `fab_trans_east_wdsd`.  
   - `load_to_dsr(dest_dsr_recv, fab_trans_*_wdsd, .{.async=true, .activate = f_actv_send_recv, .ut_id = in_ut});`  
   - `load_to_dsr(src1_dsr_recv, fab_recv_*_wdsd, .{.async=true});`  
   - Continuation: **f_actv_send_recv** on **microthread in_ut** (`ut_id = 1`).

So:
- **Active PEs** use the default microthread (per queue).
- **Forwarding PEs** all use the **same** microthread: **in_ut** (ut_id = 1).

### Microthread semantics (Cerebras SDK)

- From [Microthreads WSE-3](https://sdk.cerebras.net/csl/language/microthreads_wse3):
  - `.ut_id` selects which microthread runs the **completion** of the async DSD op.
  - **“Two operations cannot concurrently use the same microthread.”**
  - If `.ut_id` is not set, the microthread ID defaults to the queue ID of the first fabric operand (so different queues → different microthreads).

So:
- **Active recv:** continuations spread over queue-based microthreads ⇒ can run in parallel.
- **Forward recv:** all continuations go to **one** microthread (in_ut = 1) ⇒ **serialized**.

At **coarser levels**:
- More PEs are void and use the **forward** path.
- More of them schedule **f_actv_send_recv** on **in_ut**.
- That single microthread processes a **long chain** of forward completions.
- The **critical path** for “recv done” becomes **length of this chain** (control/scheduling) more than raw data movement ⇒ **recv time goes up** even though the grid is smaller.

Send does not show this because when `inactive_horizontal`/`inactive_vertical` the send side **skips** the load_to_dsr (comment: “delay the SEND at f_recv”); it does not schedule a shared microthread for send. So send time stays dominated by data volume, which decreases with level.

## 2. Is Control Code Hogging?

Yes, in the sense that:

- **More distance (coarser level)** ⇒ **more inactive horizontal/vertical PEs**.
- More of them take the **forward** path and **queue on in_ut**.
- The **control** cost (running f_actv_send_recv one after another on microthread 1) grows and dominates.

So the “control code” (continuation scheduling on a single microthread) is effectively hogging the recv critical path at coarser levels.

## 3. f_recv Micro-Benchmark (Implemented)

Time in `f_recv` is split into:

- **recv_into_buf:** active path (receive into mem_*_buf, activate f_recv; default microthread per queue).
- **recv_forward:** forward path (load from trans, activate f_actv_send_recv with **in_ut**).

Run the benchmark and check the "f_recv Micro-Benchmark" table: **recv_forward** should grow with level while **recv_into_buf** stays flat or shrinks, confirming in_ut serialization.

## 4. Mitigation Ideas

- Use **multiple microthreads** for the forward path (e.g. different `ut_id` per direction or per “lane”) so forward completions are not all serialized on in_ut.
- Restructure so forwarding does not go through a single shared microthread.
