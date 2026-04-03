# Read-After-Write (RAW) Hazard in Hop-Based Stencil Communication

## Executive Summary

A **Read-After-Write (RAW) hazard** exists in the 7-point stencil PE (`pe.csl`) when inactive (forwarding) PEs participate in hop-based communication. The hazard arises because **SEND** and **RECV** tasks run **concurrently** while forwarding PEs must **read** from receive buffers that **RECV** is still **writing**. PE N can attempt to forward data before it has arrived from PE N-1, causing incorrect or stale data to propagate.

---

## 1. Root Cause

### 1.1 Concurrent SEND and RECV Activation

In `f_comm()` (lines 745–747), both tasks are activated **simultaneously**:

```csl
@activate(SEND);
@activate(RECV);
```

The same pattern appears in `compute_and_next_send_recv()` when starting the next hop (lines 504–505):

```csl
@activate(SEND);
@activate(RECV);
```

### 1.2 Buffer Dependencies for Forwarding PEs

For **inactive (forwarding) PEs**, the SEND task uses receive buffers as source data:

| SEND Direction | Buffer Used | Written By RECV |
|----------------|-------------|-----------------|
| SEND EAST      | `mem_west_buf_dsd` (west_buf)  | RECV WEST  |
| SEND WEST      | `mem_east_buf_dsd` (east_buf)  | RECV EAST  |
| SEND NORTH     | `mem_south_buf_dsd` (south_buf)| RECV SOUTH |
| SEND SOUTH     | `mem_north_buf_dsd` (north_buf)| RECV NORTH |

The forwarding logic in `f_send()`:

```csl
// SEND EAST - inactive PE forwards from west_buf
} else {
    @load_to_dsr(src1_dsr_send, mem_west_buf_dsd);   // READ west_buf
}
@mov32(dest_dsr_send, src1_dsr_send, ...);

// SEND WEST - inactive PE forwards from east_buf
} else {
    @load_to_dsr(src1_dsr_send, mem_east_buf_dsd);   // READ east_buf
}
// ... similar for NORTH (south_buf) and SOUTH (north_buf)
```

The corresponding RECV logic writes into those same buffers:

```csl
// RECV WEST - writes west_buf
@load_to_dsr(dest_dsr_recv, mem_west_buf_dsd);
@load_to_dsr(src1_dsr_recv, fab_recv_west_wdsd, .{.async=true, .activate = f_recv});
@mov32(dest_dsr_recv, src1_dsr_recv, .{.async=true});   // WRITE west_buf
```

### 1.3 The RAW Hazard

Because SEND and RECV start at the same time:

1. **RECV WEST** begins an async fabric read into `west_buf`.
2. **SEND EAST** (for a forwarding PE) immediately tries to read `west_buf` and send it east.

There is no ordering between these operations. SEND EAST can run its `@load_to_dsr(src1_dsr_send, mem_west_buf_dsd)` and `@mov32` **before** RECV WEST’s `@mov32(dest_dsr_recv, src1_dsr_recv)` completes.

**Result:** SEND reads `west_buf` before it has been written → **RAW hazard**.

---

## 2. Effect: PE N Waits for PE N-1

Data flows west-to-east as:

```
PE 0 → PE 1 → PE 2 → PE 3 → ...
```

For a multi-hop setup:

- **PE 1** (forwarder) must receive from **PE 0** before it can send to **PE 2**.
- **PE 2** (forwarder) must receive from **PE 1** before it can send to **PE 3**.
- And so on.

This forms a chain:

1. PE 0 sends to PE 1.
2. PE 1 must **receive** from PE 0 (write `west_buf`) before it can **send** (read `west_buf`) to PE 2.
3. PE 2 must **receive** from PE 1 before it can **send** to PE 3.

Because SEND and RECV are concurrent, PE 1’s SEND can start before its RECV finishes. PE 2 then receives incomplete or stale data, and the dependency chain breaks. In effect, PE N cannot correctly forward data until PE N-1 has both sent and PE N has received it; the current design does not enforce this order for forwarding PEs.

---

## 3. Why Active PEs Are Unaffected

For **active PEs**, SEND uses `mem_center_buf_dsd` (local data), which is set up in `spmv()` **before** `COMM` is activated:

```csl
mem_center_buf_dsd = @set_dsd_base_addr(mem_center_buf_dsd, x);  // in spmv()
// ...
@activate(COMM);
```

So active PEs:

- Do **not** read from receive buffers in SEND.
- Have no dependency between RECV and SEND for the same block.

The RAW hazard only affects **inactive (forwarding) PEs**.

---

## 4. Why “All RECV Before All SEND” Fails: Serialization and Deadlock

The naive fix—run **all** RECV before **any** SEND for forwarding PEs—breaks the bidirectional pipeline and causes **deadlock**.

### 4.1 East–West Bidirectional Dependencies

Each PE participates in **both** directions on the same axis:

- **PE1 RECV EAST** receives from **PE2 SEND WEST**
- **PE2 RECV WEST** receives from **PE1 SEND EAST**

With “all RECV before all SEND”:

1. PE1 runs RECV first. PE1 RECV WEST (from PE0) can complete.
2. PE1 RECV EAST (from PE2) needs PE2 to have sent west. But PE2 also runs RECV first, and PE2 RECV WEST needs PE1 SEND EAST.
3. So PE1 must SEND EAST before PE2 can finish RECV, but PE1 will not SEND until PE1 RECV completes, and PE1 RECV EAST is blocked on PE2 SEND.

Result: **deadlock**.

### 4.2 Serialization Along the Chain

Even if we only consider the eastward flow (PE0 → PE1 → PE2 → …):

- PE1 cannot SEND until PE1 RECV is fully done.
- PE2 cannot RECV WEST until PE1 SEND EAST.
- PE2 cannot SEND until PE2 RECV is fully done.
- And so on.

The chain becomes strictly serial: PE0 sends → PE1 receives all → PE1 sends → PE2 receives all → PE2 sends → … We lose **pipeline overlap** between neighboring PEs and serialize communication.

### 4.3 Conclusion

**SEND and RECV must be allowed to overlap** for the bidirectional pattern to avoid deadlock and keep pipeline parallelism. The fix must enforce ordering **per direction**, not across all directions.

---

## 5. The Correct Fix: Per-Direction Interleaving

Instead of “all RECV before all SEND,” enforce **each SEND only after its corresponding RECV**:

| SEND that reads | Must wait for |
|-----------------|---------------|
| SEND EAST  (reads west_buf)  | RECV WEST  |
| SEND WEST  (reads east_buf)  | RECV EAST  |
| SEND NORTH (reads south_buf) | RECV SOUTH |
| SEND SOUTH (reads north_buf) | RECV NORTH |

### 5.1 Interleaved Order for Forwarding PEs

Run RECV and SEND in dependency order:

```
RECV WEST  → SEND EAST  → RECV EAST  → SEND WEST  →
RECV SOUTH → SEND NORTH → RECV NORTH → SEND SOUTH
```

This preserves:

1. **Pipeline:** PE1 can RECV WEST from PE0 while PE0 SEND EAST runs. As soon as RECV WEST completes, PE1 can SEND EAST to PE2. PE2 can RECV WEST from PE1 without waiting for PE1’s full RECV.
2. **No deadlock:** Each direction is ordered only by its own RECV→SEND pair; east and west (and north/south) remain independent.
3. **RAW safety:** Each SEND reads only from a buffer that its corresponding RECV has already written.

### 5.2 Implementation Options

Implementing this requires restructuring the SEND/RECV state machines to interleave per direction. Possible approaches:

- **Option A:** Merge SEND and RECV into one task that alternates RECV→SEND per direction for forwarding PEs.
- **Option B:** Add per-direction synchronization (e.g., RECV completion callbacks that trigger only the next SEND in that direction).
- **Option C:** Use a single state machine with states: `RECV_WEST → SEND_EAST → RECV_EAST → SEND_WEST → RECV_SOUTH → SEND_NORTH → RECV_NORTH → SEND_SOUTH` for forwarding PEs, while keeping the current parallel SEND/RECV for active PEs.

---

## 6. Fabric-to-Fabric Forwarding (Recommended Fix)

A cleaner approach used in `modified_csl_lib_hops/pe.csl` is **fabric-to-fabric forwarding**. Instead of RECV→buffer→SEND, forwarding PEs perform a **direct fabric-to-fabric copy**: read from fabric input, write to fabric output, with **no intermediate buffer**. This removes the RAW hazard entirely.

### 6.1 How It Works

For a forwarding PE (e.g., east-west):

- **Current (buggy):** RECV WEST writes `west_buf`, SEND EAST reads `west_buf` → RAW hazard.
- **Fabric-to-fabric:** In RECV WEST, do `mov32(fab_trans_east, fab_recv_west)` — copy directly from input queue to output queue. No buffer.

The `f_actv_send_recv()` task is used as the **async completion callback** for the fabric-to-fabric `mov32`. When that copy completes, it reactivates SEND and RECV so both state machines can continue.

### 6.2 Why There Is No RAW Hazard

- **Active PEs:** SEND from `mem_center_buf_dsd` (local data). RECV into buffers. No overlap.
- **Forwarding PEs:** RECV does `fab_recv_* → fab_trans_*` (fabric-to-fabric). SEND does nothing for that direction (skipped). No buffer read/write conflict → **no RAW hazard**.

### 6.3 Fix for `modified_csl_lib/stencil_3d_7pts/wse3/pe.csl`

The current `modified_csl_lib` pe.csl uses `is_active_pe` and `can_communicate_*`. Adapt the hops approach as follows.

#### Step 1: Add `f_actv_send_recv` task and bind it

```csl
param ACTV_SEND_RECV: local_task_id = @get_local_task_id(13);  // or another free task ID

task f_actv_send_recv() void {
    @activate(SEND);
    @activate(RECV);
}

comptime {
    @bind_local_task(f_send, SEND);
    @bind_local_task(f_recv, RECV);
    @bind_local_task(f_comm, COMM);
    @bind_local_task(f_actv_send_recv, ACTV_SEND_RECV);
}
```

#### Step 2: In `f_recv()` — use fabric-to-fabric for forwarding PEs

For each direction, when `!is_active_pe` and the PE can forward (can send in the opposite direction), do fabric-to-fabric instead of recv-into-buffer:

```csl
// RECV WEST
if (can_communicate_west) {
    if (!is_active_pe and can_communicate_east) {
        // Forward: fabric_recv_west -> fabric_send_east (no buffer)
        @load_to_dsr(dest_dsr_recv, fab_trans_east_wdsd, .{.async=true, .activate = f_actv_send_recv});
        @load_to_dsr(src1_dsr_recv, fab_recv_west_wdsd, .{.async=true});
    } else {
        @load_to_dsr(dest_dsr_recv, mem_west_buf_dsd);
        @load_to_dsr(src1_dsr_recv, fab_recv_west_wdsd, .{.async=true, .activate = f_recv});
    }
    @mov32(dest_dsr_recv, src1_dsr_recv, .{.async=true});
} else {
    @activate(RECV);
}
// ... similar for RECV EAST (fab_recv_east -> fab_trans_west), RECV SOUTH, RECV NORTH
```

#### Step 3: In `f_send()` — skip send for forwarding PEs

When `!is_active_pe`, the RECV path already performs the send via fabric-to-fabric. SEND should do nothing and just advance the state:

```csl
// SEND EAST
if (can_communicate_east) {
    if (is_active_pe) {
        @load_to_dsr(dest_dsr_send, fab_trans_east_wdsd, .{.async=true, .activate = f_send, .ut_id = out_ut});
        @load_to_dsr(src1_dsr_send, mem_center_buf_dsd);
        @mov32(dest_dsr_send, src1_dsr_send, .{.async=true, .ut_id = out_ut});
    }
    // else: forwarding PE — RECV already did fabric-to-fabric, skip
} else {
    @activate(SEND);
}
send_state = SEND_STATE_WEST;
// ... similar for SEND WEST, SEND NORTH, SEND SOUTH
```

**Note:** When SEND is skipped for forwarding PEs, it advances the state and returns without scheduling async work. The RECV path's fabric-to-fabric `mov32` uses `f_actv_send_recv` as its completion callback, which reactivates both SEND and RECV. SEND then runs again and continues from the next state.

#### Step 4: Mapping to `modified_csl_lib` conditions

`modified_csl_lib` uses `can_communicate_west`, `can_communicate_east`, etc. Forward east when: `!is_active_pe and can_communicate_west and can_communicate_east`. Forward west when: `!is_active_pe and can_communicate_east and can_communicate_west`. So the conditions are symmetric — use fabric-to-fabric whenever `!is_active_pe` and both directions on that axis can communicate.

### 6.4 Summary

| Aspect            | Current (Buggy)             | Fabric-to-Fabric Fix                     |
|-------------------|-----------------------------|------------------------------------------|
| Forwarding path   | RECV→buffer, SEND←buffer    | RECV: fab_in→fab_out (no buffer)         |
| SEND for forwarders | Reads from recv buffers   | Skipped (RECV does the send)             |
| f_actv_send_recv  | N/A                         | Callback to reactivate SEND and RECV     |
| RAW hazard        | Yes                         | No                                       |
| Pipeline          | Broken by hazard            | Preserved                                |

---

## 7. Summary

| Aspect             | Current (Buggy)                     | Naive Fix (Flawed)               | Fabric-to-Fabric Fix (Recommended)    |
|--------------------|-------------------------------------|----------------------------------|---------------------------------------|
| SEND/RECV ordering | Concurrent (RAW hazard)             | All RECV before all SEND         | RECV does fabric→fabric; SEND skipped for forwarders |
| Pipeline           | Overlaps but has RAW                | **Serialized + deadlock**        | Preserves overlap                     |
| Implementation     | Two parallel tasks                  | Would block east-west flow       | Add f_actv_send_recv; RECV fabric-to-fabric for !is_active_pe |
| RAW hazard         | Yes                                 | Would fix but causes deadlock    | No (no buffer in forwarding path)     |

**Takeaway:** The recommended fix is **fabric-to-fabric forwarding**: for forwarding PEs (`!is_active_pe`), RECV performs a direct fabric input → fabric output copy instead of buffer read/write, and SEND is skipped for that direction. `f_actv_send_recv` serves as the async completion callback to reactivate SEND and RECV. This removes the RAW hazard without serialization or deadlock.

---

## 8. Applied Fix: First Solution (Alternating RECV–SEND Order)

The **first solution** — per-direction ordering without fabric-to-fabric or microthread changes — has been applied to `modified_csl_lib/stencil_3d_7pts/wse3/pe.csl`.

### Idea

RAW occurs because SEND and RECV run in parallel while SEND reads buffers RECV is writing. The fix is to **serialize per direction**: run RECV, then SEND for each direction pair. No buffer is read until the corresponding RECV has finished writing it.

### Changes Made

1. **Alternating activation:** RECV and SEND run in strict order:
   - RECV WEST → SEND EAST → RECV EAST → SEND WEST → RECV SOUTH → SEND NORTH → RECV NORTH → SEND SOUTH

2. **`f_comm()`** — Start only RECV:
   ```csl
   @activate(RECV);
   ```

3. **`f_recv()`** — Each RECV completion activates SEND:
   ```csl
   .activate = f_send
   ```

4. **`f_send()`** — Each SEND completion activates RECV:
   ```csl
   .activate = f_recv
   ```

5. **`f_actv_both`** — When SEND SOUTH completes, both SEND and RECV are activated so both reach their DONE state and increment `count_send_recv`.

6. **Buffer-based forwarding unchanged** — Inactive PEs still send from recv buffers (west_buf, east_buf, etc.), but ordering guarantees that each RECV has completed before the corresponding SEND runs.

### Why This Works

Each SEND that uses a recv buffer runs only after the matching RECV has finished. The execution order enforces RECV-before-SEND per direction, so there is no RAW hazard.

---

## 9. Reference: Affected Code Regions

**Target file:** `modified_csl_lib/stencil_3d_7pts/wse3/pe.csl` (fix applied)

- `f_comm()`: ~lines 721–748
- `f_send()`: ~lines 528–593 (forwarding uses west_buf, east_buf, south_buf, north_buf)
- `f_recv()`: ~lines 603–651
- `compute_and_next_send_recv()`: ~lines 494–518
