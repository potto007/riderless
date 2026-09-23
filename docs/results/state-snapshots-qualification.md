# State snapshots: split18-30-v1 qualification, 2026-09-22

Profile `split18-30-v1` ([ADR 0006](../decisions/0006-split-execution-for-state-snapshots.md)),
worker `build/snapshot-worker`, llama.cpp v0.4.1 plus
`gemma4-layer-range.patch`, Gemma 4 26B-A4B UD-Q4_K_XL, RTX 5090 (sm_120),
context 2048, batch/ubatch 256, CUDA fusion and graphs on. Harness:
`scripts/unridden/qualify_snapshots.py` over the frozen
`unridden/examples/api-v1-cases.json` corpus (12 cases, 36 questions) plus one
synthetic 1,415-token case past the 1,024-token sliding window. Tolerances
fixed in the script before the run.

## Gates

| Gate | Pass | Fail | Notes |
| --- | --- | --- | --- |
| 1 capture integrity | 13 | 0 | `[N, 2816]` F32 H18/H30, K/V per range, shared lower blob |
| 2 execution audit | 77 | 0 | create N/N, create-18 N/0, promote 0/N, branch q/q block tokens; zero generated tokens |
| 3 restore identity | 51 | 0 | resident vs host restore vs disk restore in a fresh process: all bit-identical, including past the SWA window |
| 4 promotion identity | 38 | 0 | S18 promoted then asked equals paired S30 asked, bit-identical |
| 5 stock vs split | 0 | 38 | fails on residual relative L2 only; see below |
| 6 branch isolation | 39 | 0 | A,B,A and reversed order equal; parent blobs unchanged; unaffected by failed requests |
| 7 rejections | 28 | 0 | block-18 readout, prefix mismatch, budget overflow, corrupt blob |
| informational: branch vs stock | 38 | 0 | prefix+suffix chunking against the stock single pass |

The profile therefore stays **experimental**: gate 5 is predeclared and failed.

### Restore past the sliding window (found and fixed during qualification)

Stock `llama_state_seq_get_data` drops SWA-masked cells. A restored cache then
held fewer cells than the live one, the attention reduction ran over a
different cell count, and a 1,707-token prefix gave label logits
`[14.55, 13.17]` after restore against `[14.68, 12.71]` resident. The patch
adds `LLAMA_STATE_SEQ_FLAGS_KEEP_SWA_MASKED`; the worker saves with it and
all three paths are now bit-identical. The cost is K/V bytes growing linearly
past the window (table below).

### Gate 5 in detail

Same tokens and chunking, stock uninterrupted 30-block graph versus the split
graph, both on the GPU:

| Measure | Tolerance | Median | Max |
| --- | --- | --- | --- |
| label-probability delta | 1e-3 | ~1e-8 | 1.8e-7 |
| H18 last-row relative L2 | 1e-3 | 1e-4 | 8.4e-3 |
| H30 last-row relative L2 | 1e-3 | 1.2e-2 | 2.9e-2 |

Decisions and probabilities agree far inside tolerance; the raw residuals do
not. Diagnosis on one case (124 tokens, all rows):

- **On CPU the split and stock graphs are bit-identical** at every row of H18
  and H30. The patch's layer split, K/V ownership and residual hand-off are
  exact.
- On GPU, H30 rows 0-2 are identical and later rows differ by 2-27%, including
  rows whose H18 input is bit-identical, so the difference arises inside the
  upper blocks' GPU kernels (a different graph shape selects different
  kernels/fusions), not in the hand-off.
- The GPU stock graph is itself 4-7% (median relative L2) from the CPU result
  on the same residuals; the GPU split graph is the same distance. The 1e-3
  residual tolerance is below the kernel-path noise this quantized MoE model
  shows between any two GPU graph shapes, the stock one included.

Per the design the tolerance is not relaxed. Moving past experimental needs a
separately reviewed validation protocol, for example one that bounds residual
drift relative to the stock graph's own GPU-vs-CPU drift and keeps the
decision and probability criteria.

Separately, chunk boundaries matter on this model at long prefixes: at 1,707
tokens a snapshot branch (prefix chunks, then suffix) and the stock single
pass over the whole prompt gave `[14.68, 12.71]` versus `[15.23, 13.44]`. The
same regime-dependence is recorded for v1 in [prefix reuse](prefix-reuse.md).
Snapshot reuse is reproducible against itself, not bit-identical to a
from-scratch v1 answer.

## Cost (gate 8)

`scripts/unridden/bench_snapshots.py`, 20 repeats, p50 / p95 ms, one 56-token
two-option question per branch. "stock full prompt" is the same question
answered from scratch through the stock graph (what no snapshot costs).

| Prefix | create 18+30 | create 18 | promote | branch resident | branch host | branch disk | stock full prompt | save | load |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 141 | 40.8 / 42.8 | 26.0 / 27.7 | 22.3 / 23.9 | 23.7 / 35.2 | 29.5 / 35.8 | 25.2 / 27.1 | 28.6 / 34.6 | 148 / 175 | 83 / 88 |
| 518 | 140.2 / 150.0 | 81.8 / 88.1 | 67.3 / 74.6 | 24.6 / 33.8 | 36.7 / 45.2 | 32.0 / 36.9 | 102.4 / 112.1 | 332 / 409 | 325 / 338 |
| 1011 | 241.3 / 254.9 | 144.8 / 149.2 | 114.4 / 122.7 | 24.5 / 29.6 | 45.2 / 51.4 | 38.3 / 42.2 | 178.8 / 197.4 | 714 / 812 | 625 / 639 |
| 1765 | 432.8 / 442.0 | 251.8 / 258.8 | 193.2 / 201.1 | 25.1 / 27.8 | 56.8 / 60.9 | 56.1 / 61.6 | 296.9 / 310.1 | 1118 / 1256 | 1116 / 1240 |

Serialized sizes per snapshot pair:

| Prefix | lower K/V | upper K/V | H18 | H30 |
| --- | --- | --- | --- | --- |
| 141 | 18.2 MiB | 12.1 MiB | 1.5 MiB | 1.5 MiB |
| 518 | 66.8 MiB | 44.5 MiB | 5.6 MiB | 5.6 MiB |
| 1011 | 130.3 MiB | 86.9 MiB | 10.9 MiB | 10.9 MiB |
| 1765 | 227.6 MiB | 151.7 MiB | 19.0 MiB | 19.0 MiB |

Readings:

- A warm branch costs ~25 ms whatever the prefix; a host restore adds 5-32 ms.
  At 1,765 prefix tokens that is 11.8x (resident) and 5.2x (host) cheaper
  than answering from scratch. At 141 tokens there is no saving.
- Creating the pair costs ~1.4x one stock pass (the two captures plus two
  state serializations); it pays back from the second question on long
  prefixes.
- Promotion costs ~65% of a stock pass, consistent with 12 of 30 blocks plus
  one state save.
- Durable save/load is dominated by fsync and by the bundled single-threaded
  SHA-256 over hundreds of MB; it is the slowest operation and the obvious
  next optimization.
- Peak GPU memory was not separately instrumented; the worker ran with
  :8080 unloaded and ~27 GB free.
