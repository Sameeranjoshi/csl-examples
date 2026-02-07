# Async DSD Operations and Microthread IDs (ut_id)

This note summarizes the Cerebras SDK rules for **asynchronous DSD operations** and how we apply them in the stencil `pe.csl` (WSE-3) so that send and recv (and forwarding) do not serialize.

**Primary reference:** [Data Structure Descriptors — Asynchronous DSD Operations](https://sdk.cerebras.net/csl/language/dsds#asynchronous-dsd-operations) and [Microthread IDs (WSE-3)](https://sdk.cerebras.net/csl/language/microthreads_wse3).

**Cerebras Microthread IDs rules we follow:**
- `.ut_id` can be set on the DSD operation or on `@load_to_dsr` for explicit DSR operands; value must be **comptime-known**.
- When **multiple** operands have `.ut_id`, hardware picks: **destination > first source > second source**. With a mix of DSR and DSD, DSD uses the operation’s `.ut_id`.
- When `.ut_id` is **not** specified, microthread = queue ID of the **first fabric operand** (same ordering: destination > first source > second source).
- So for `@mov32(dest_dsr, src_dsr, ...)` with both DSRs from `@load_to_dsr`, the **destination’s** `load_to_dsr` `.ut_id` wins. We set **in_ut** on both loads in RECV forward so the op runs on in_ut; we set **out_ut** on SEND so SEND runs on out_ut.

---

## 1. When is an operation asynchronous?

From the SDK:

- A DSD operation is **asynchronous** if:
  1. At least one operand has a **fabric DSD** (`fabin_dsd` or `fabout_dsd`) and the **`.async = true`** configuration is used, or
  2. At least one **DSR** operand was loaded with **`.async = true`** in `@load_to_dsr`.

- Operations with fabric operands run on **microthreads**. Each async operation consumes a **microthread** (and a **queue**). The programmer must ensure **no two concurrent async operations share the same queue or microthread**.

---

## 2. Queues: input vs output

- **Fabric input** (`fabin_dsd`): must specify **`input_queue`** (data coming *into* the PE).
- **Fabric output** (`fabout_dsd`): must specify **`output_queue`** (data leaving the PE).

IQs and OQs are hardware buffers; each fabric DSD is bound to one of them.

---

## 3. Microthread ID (WSE-2 vs WSE-3)

### WSE-2 (implicit)

Microthread ID is **implicitly** determined by a queue involved in the operation:

1. If a **`fabout_dsd`** operand is used → microthread ID = **output_queue** identifier.
2. Otherwise → microthread ID = **input_queue** identifier of the first **`fabin_dsd`** operand.

So: **send (fabout) → output-queue microthread; receive (fabin) → input-queue microthread.**

### WSE-3 (explicit)

On WSE-3 you can **explicitly** set the microthread ID via **`.ut_id`** in the DSD operation or in **`@load_to_dsr`** (for explicit DSR operands). If **`.ut_id`** is not set, the microthread ID defaults to the **queue ID of the first fabric operand** (destination > first source > second source).

---

## 4. Rule we follow in `pe.csl`

We use **explicit** microthread IDs on WSE-3:

| Operand type | DSD kind   | Queue      | Microthread we use |
|-------------|------------|------------|---------------------|
| **Send / write to fabric** | `fabout_dsd` | output_queue | **`out_ut`** |
| **Recv / read from fabric** | `fabin_dsd`  | input_queue  | **`in_ut`** (or default) |

- **`out_ut`** = microthread for the **output** queue (e.g. `@get_ut_id(output_ut_id)`).
- **`in_ut`** = microthread for the **input** queue (e.g. `@get_ut_id(1)`).

So:

- Any **`@load_to_dsr(..., fab_trans_*_wdsd, ...)`** (fabout = send) → use **`.ut_id = out_ut`**.
- Any **`@load_to_dsr(..., fab_recv_*_wdsd, ...)`** (fabin = recv) → use **`.ut_id = in_ut`** only when we need to pin the recv microthread; otherwise omit or use default (input queue).

---

## 5. Why using `in_ut` on a send caused serialization

In the **forward** path we **send** data (write to `fab_trans_*_wdsd`, a **fabout**). If we incorrectly set **`.ut_id = in_ut`** on that operation:

- The “forward send” runs on the **input** microthread (`in_ut`).
- Real **receives** (from `fab_recv_*_wdsd`) also use the input queue / `in_ut`.

So both “forward send” and “receive” were scheduled on the **same** microthread → **serialized** and slower.

**Fix:** Use **`.ut_id = out_ut`** for every `@load_to_dsr(..., fab_trans_*_wdsd, ...)` in the forward path (and in `f_send`). Then:

- Sends (including forwarding) use **out_ut** (output queue).
- Receives use **in_ut** (input queue).
- They can run **concurrently** and no longer serialize.

## 5b. Why using `out_ut` in RECV forward caused a hang (mentor fix)

SEND and RECV run in **parallel** (different microthreads). Within each task, the four directions run **sequentially** on one microthread. If RECV forward uses **`.ut_id = out_ut`**, then when RECV gets ahead of SEND (e.g. RECV finishes E and starts S→N forward while SEND is still doing W), **both** RECV forward and SEND try to use **out_ut** at once → two concurrent async ops on the same microthread → **hang**.

**Mentor's fix (no ut_id):** In the **forward** path in `f_recv`, **do not** set `.ut_id = out_ut`. Use only `.async=true, .activate = f_actv_send_recv`. That avoids the hang. On WSE-3 the default microthread when omitting ut_id can still be the output queue (destination DSR is fabout), so SEND and RECV forward may still serialize.

**Parallelism fix (in_ut):** Explicitly set **`.ut_id = in_ut`** for RECV forward. Then RECV (receive + forward) runs on the **input** microthread and SEND keeps **out_ut**. SEND and RECV use different microthreads → true parallelism, no hang (RECV never uses out_ut).

---

## 6. Where we apply this in `pe.csl`

- **`f_send()`**: All `@load_to_dsr(dest_dsr_send, fab_trans_*_wdsd, ...)` use **`.ut_id = out_ut`** (SEND owns out_ut).
- **`f_recv()` — forward path**: `@load_to_dsr(dest_dsr_recv, fab_trans_*_wdsd, ...)` use **no `.ut_id`** so the microthread defaults and does not conflict with SEND's out_ut (WEST→EAST, EAST→WEST, SOUTH→NORTH, NORTH→SOUTH).
- **`f_recv()` — receiver path**: `@load_to_dsr(src1_dsr_recv, fab_recv_*_wdsd, ...)` use **`.activate = f_recv`**; microthread is the input side (in_ut by default).

Result: SEND uses out_ut; RECV (receive + forward) uses default / input microthread. They stay in parallel and do not share a microthread.
