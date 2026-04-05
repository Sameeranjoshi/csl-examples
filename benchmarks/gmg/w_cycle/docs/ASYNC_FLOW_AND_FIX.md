# Async Flow in Base Stencil and Fix for Hop-Based Stall

## 1. How Async Works in the Base csl-libs Stencil

The base stencil (no hop forwarding) uses **two concurrent tasks**: `f_send` and `f_recv`. Both are activated by `f_comm` and run in parallel.

### 1.1 Activation Flow

```
spmv() → @activate(COMM)
f_comm() → @activate(SEND)
        → @activate(RECV)
```

Both SEND and RECV start at the same time. Each advances through its own state machine.

### 1.2 Async Completion Semantics

When an operation uses `.async=true` and `.activate = f_send`:

```csl
@load_to_dsr(dest_dsr_send, fab_trans_east_wdsd, .{.async=true, .activate = f_send, .ut_id = out_ut});
@load_to_dsr(src1_dsr_send, mem_center_buf_dsd);
@mov32(dest_dsr_send, src1_dsr_send, .{.async=true, .ut_id = out_ut});
```

- The `@mov32` initiates a DMA/fabric transfer.
- When the transfer **completes**, the runtime invokes `f_send` again (via `@activate(SEND)`).
- The task returns immediately; it does **not** block on the transfer.

### 1.3 State Advancement

Each task advances its state **before** returning:

```csl
// In f_send, SEND_STATE_EAST:
send_state = SEND_STATE_WEST;  // advance state
// then either async mov32 (completion will re-invoke f_send) or @activate(SEND)
```

When re-invoked, the task runs from the top with the **new** state.

### 1.4 Completion Coordination

Both SEND and RECV must finish a full cycle (EAST, WEST, NORTH, SOUTH). Each increments `count_send_recv` when it hits DONE. When `count_send_recv >= 2`, `compute_and_next_send_recv` runs:

- Laplacian on x-y
- Advance block pointers
- `@activate(COMM)` for next block

The base code has **no RAW hazard** because active PEs always SEND from `mem_center_buf_dsd` (local data), never from receive buffers.

---

## 2. Why the Stall-Based Fix Fails

The stall-based approach adds per-direction flags (`recv_done_west`, etc.) and makes SEND wait:

```csl
if (!is_active_pe and !recv_done_west) {
    return;  // STALL - no @activate, no continuation
}
```

When SEND returns (stalls), it schedules **nothing**. RECV is supposed to set the flag and call `@activate(SEND)` when it completes that direction.

### 2.1 Potential Stall Causes

1. **Async completion timing**: The POST-ACTION (set flag, activate SEND) runs when RECV is re-invoked after its async completes. If completion semantics differ from expectations (e.g., completion fires before data is fully written), ordering can break.

2. **"Nothing to receive" paths**: When `can_communicate_west` is false, RECV does `@activate(RECV)` and never activates SEND. For an inactive PE at a boundary, SEND would never run. (In typical layouts, boundary PEs are active, so this may not trigger.)

3. **Hop restart**: When starting the next hop, `reset_recv_flags()` clears all flags. SEND is activated. For inactive PEs, SEND immediately stalls (flags false). RECV must complete and set flags. If RECV's async never completes (e.g., upstream deadlock), the whole chain stalls.

4. **Bidirectional dependencies**: East-west and north-south form chains. A single PE waiting indefinitely can stall the entire grid.

---

## 3. The Correct Fix: Single-Task RECV-then-SEND

Per RAW_HAZARD_REPORT Section 5, the correct fix is **per-direction interleaving**:

```
RECV WEST  → SEND EAST  → RECV EAST  → SEND WEST  →
RECV SOUTH → SEND NORTH → RECV NORTH → SEND SOUTH
```

By running RECV and SEND in this order within a **single task**, we:

1. **Eliminate RAW hazard**: We never read a recv buffer until we have just written it.
2. **Eliminate stall/deadlock**: No flag coordination; one linear state machine.
3. **Preserve pipeline**: Neighboring PEs still overlap (PE1 RECV WEST overlaps with PE0 SEND EAST).

### 3.1 Trade-off

We lose SEND/RECV overlap *within* a single PE. The base design lets SEND and RECV run in parallel. The unified design serializes them per direction. For correctness with hop forwarding, this is the reliable approach. Performance can be tuned later (e.g., fabric-to-fabric for forwarders).

---

## 4. Implementation: Fabric-to-Fabric (Current)

The implementation uses **fabric-to-fabric forwarding** ([Pipeline 1](https://sdk.cerebras.net/csl/code-examples/tutorial-pipeline-01-basic)): direct fabin→fabout redirect with no buffer.

### 4.1 How It Works

- **Active PEs:** RECV into buffers; SEND from `mem_center_buf_dsd`. Standard flow.
- **Forwarding PEs:** RECV does `mov32(fab_trans_*, fab_recv_*)` — copy fabric input → fabric output. No buffer. SEND is skipped for that direction.

### 4.2 Why No RAW Hazard

Forwarding PEs never touch recv buffers. Data flows fabin→fabout. `f_actv_send_recv` reactivates SEND and RECV when the fabric-to-fabric `mov32` completes.

### 4.3 Flow

SEND and RECV run in parallel. For forwarders, RECV does fabin→fabout; SEND advances state only. Both hit DONE and call `compute_and_next_send_recv` for hop/block progression.
