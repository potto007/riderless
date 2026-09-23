# State snapshots: split18-30-v1 qualification, 2026-09-22

Profile `split18-30-v1` ([ADR 0006](../decisions/0006-split-execution-for-state-snapshots.md)),
worker `build/snapshot-worker`, llama.cpp v0.4.1 plus
`gemma4-layer-range.patch` (sha256 `3e802b5a...`), Gemma 4 26B-A4B
UD-Q4_K_XL, RTX 5090 (sm_120), context 2048, batch/ubatch 256, CUDA fusion and
graphs on. Harness: `scripts/riderless/qualify_snapshots.py` over the frozen
`riderless/examples/api-v1-cases.json` corpus (12 cases, 36 questions),
tolerances fixed in the script before the run. Whole run: 29 s.

## Result

| Gate | Pass | Fail | Notes |
| --- | --- | --- | --- |
| 1 capture integrity | 12 | 0 | `[N, 2816]` F32 H18/H30, K/V per range, shared lower blob |
| 2 execution audit | 72 | 0 | create N/N, create-18 N/0, promote 0/N, branch q/q block tokens; zero generated tokens |
| 3 restore identity | 48 | 0 | resident vs host restore vs disk restore in a fresh process: all 48 bit-identical |
| 4 promotion identity | 36 | 0 | S18 promoted then asked equals paired S30 asked |
| 5 stock vs split | 0 | 36 | fails on residual relative L2 only; see below |
| 6 branch isolation | 36 | 0 | A,B,A and reversed order equal; parent blobs unchanged; unaffected by failed requests |
| 7 rejections | 26 | 0 | block-18 readout, prefix mismatch, budget overflow, corrupt blob |
| informational: branch vs stock | 36 | 0 | prefix+suffix chunking against the stock single pass, max prob delta 2.7e-6 |

The profile therefore stays **experimental**: gate 5 is predeclared and failed.

## Gate 5 in detail

Same tokens and chunking, stock uninterrupted 30-block graph versus the split
graph, both on the GPU:

| Measure | Tolerance | Median | Max |
| --- | --- | --- | --- |
| label-probability delta | 1e-3 | ~1e-8 | 1.8e-7 |
| H18 last-row relative L2 | 1e-3 | 9.7e-5 | 8.4e-3 |
| H30 last-row relative L2 | 1e-3 | 1.2e-2 | 2.9e-2 |

Decisions and probabilities agree far inside tolerance; the raw residuals do
not. Diagnosis on one case (124 tokens, all rows):

- **On CPU the split and stock graphs are bit-identical** at every row of H18
  and H30. The patch's layer split, K/V ownership and residual hand-off are
  exact.
- On GPU, H30 rows 0-2 are identical and later rows differ by 2-27%, including
  rows whose H18 input is bit-identical, so the difference arises inside the
  upper blocks' GPU kernels (graph shape changes kernel/fusion choices), not in
  the hand-off.
- The GPU stock graph is itself 4-7% (median relative L2) from the CPU
  result on the same residuals; the GPU split graph is the same distance. The
  1e-3 residual tolerance is below the kernel-path noise this quantized MoE
  model shows between any two GPU graph shapes, the stock one included.

The earlier early-exit experiment failed a similar gate for the same class of
reason. Per the design, the tolerance is not relaxed. Moving the profile past
experimental needs a separately reviewed validation protocol, for example one
that bounds residual drift relative to the stock graph's own GPU-vs-CPU
drift and keeps the decision and probability criteria.

## Not yet measured

Gate 8 (cost: creation, save, host/disk restore, promotion, warm branching,
p50/p95, bytes, peak memory) and gate 7 (task quality on the use-case suites
through the new compiler) are separate runs. Byte sizes observed in gate 1 for
a 60-token prefix: lower K/V 8.1 MB, upper K/V 5.4 MB, each residual matrix
0.68 MB.
