# Unridden State Snapshots: What They Are, Why They Matter, and How They Let Researchers Steer a Model's Thinking

Briefing document, written to be loaded as a source in a Gemini notebook
(NotebookLM). It is self-contained: every term is defined here, and every
number is quoted from the project's own recorded results. Where a capability
is designed but not yet built, or proposed but not yet tested, the text says
so explicitly.

Status as of 2026-09-23, Unridden v0.3.0.

---

## 1. One-paragraph summary

Unridden is a local decision API. It answers structured questions (Choice,
Score, and true/false "Noul" questions) by reading the logits of a language
model (Gemma 4 26B-A4B, 30 transformer blocks) at the final prompt position,
and it never generates a token. The **state snapshot** feature lets a caller
freeze the model's internal computation over a piece of context at two depths:
**after block 18** (partway through the model) and **after block 30** (the full
depth). A frozen snapshot is immutable and can be branched any number of times:
each branch appends a new question or prompt, runs only the new tokens, and
returns a typed answer or a new child snapshot, while the parent stays exactly
as it was. The snapshot also exposes the model's raw hidden vectors for
inspection. This turns a one-shot classifier into a platform for reusable,
branchable, inspectable model state.

---

## 2. Background terms

- **Block (layer).** Gemma 4 26B-A4B processes every token through 30 stacked
  transformer blocks. Unridden reads its decision after the last one.
- **Residual stream / hidden state.** The vector each token carries between
  blocks. For this model it has 2,816 numbers per token. `H18` is the residual
  entering block 19; `H30` is the residual after block 30, before the final
  normalization and the vocabulary head.
- **K/V cache.** Each attention layer stores a key and a value vector for every
  token already processed, so later tokens can attend to earlier ones without
  recomputing them. This per-layer memory is what a normal "prefix cache" keeps.
- **Prefix cache.** A common inference-server optimization: when two prompts
  start with the same tokens, keep the K/V cache from the first and reuse it for
  the second.
- **Zero generated tokens.** Unridden never samples. Every answer is a softmax
  over the logits of the option labels (`A`, `B`, `C`, ...) at one position.

---

## 3. What a snapshot actually contains

A snapshot is not a text summary and not a single embedding. It is a complete,
resumable computation checkpoint.

| Component | Snapshot at block 18 (S18) | Snapshot at block 30 (S30) |
| --- | --- | --- |
| Exact token IDs, positions, template cursor | Yes | Yes |
| Attention K/V for blocks 1-18 | Yes | Yes (shared with its S18 parent) |
| Attention K/V for blocks 19-30 | Deliberately absent, marked uncomputed | Yes |
| `H18` for every token, shape `[N, 2816]` | Yes | Kept by reference via parent |
| `H30` for every token, shape `[N, 2816]` | Not available | Yes |
| Final-position normalized vector and readout | Not available | Yes |
| Model, runtime, patch and profile fingerprints | Yes | Yes |

To make a half-depth checkpoint resumable, the runtime runs the model as **two
contexts over one loaded copy of the weights**: a *lower* context that executes
and owns blocks 1-18, and an *upper* context that owns blocks 19-30 plus the
final norm and head. The upper context takes the lower context's residual
vectors as its input. This is implemented as a small recorded patch to
llama.cpp (`gemma4-layer-range.patch`), not a fork.

Snapshots can live in GPU memory (resident), in host RAM, or on disk
(fsynced, checksummed, atomically published, re-validated before use), with
TTLs, leases and a byte budget with least-recently-used eviction.

---

## 4. How this differs from simply using a prefix cache

A prefix cache and a snapshot both avoid recomputing shared context. That is
where the resemblance ends.

| Property | Ordinary prefix cache | Unridden state snapshot |
| --- | --- | --- |
| **What is saved** | K/V for every layer of a token prefix | K/V per layer range, **plus** the full residual matrices `H18` and `H30`, plus fingerprints and template metadata |
| **Depth** | Always full depth: a token is either fully processed or not | Can stop at block 18 and finish blocks 19-30 later ("promotion") without repeating blocks 1-18 |
| **Identity** | Implicit: a cache hit happens if tokens happen to match | Explicit: an opaque, immutable snapshot ID that callers name and branch from |
| **Lifetime** | Opportunistic; evicted whenever the server needs memory; Unridden v1 keeps nothing between requests | Deliberate: TTLs, leases, durable disk persistence, survives a process restart |
| **Branching guarantees** | Best-effort; siblings can perturb shared state | Every branch starts from the same immutable parent and trims back afterward; tested with A,B,A orderings, reversed orderings, and after failed requests: parent bytes unchanged, answers equal |
| **Reproducibility** | A cache hit can compute differently from a miss (see SWA note below) | Resident, host-RAM and disk-in-a-fresh-process restores were bit-identical across 51 comparisons, including past the sliding window |
| **Inspectability** | Opaque internal buffers | Hidden vectors `H18`, `H30` and the last normalized vector are exported through a bounded API |
| **Accounting** | "Cached tokens" count, at best | Reports block-token work per range (lower vs upper), restore source, restored bytes, and restore/promotion/inference time separately |
| **Safety checks** | Token match | Exact-token splice check at a compiler-controlled boundary (a mismatch is an explicit `snapshot_prefix_mismatch` error), plus model/runtime/profile hash checks and blob checksums |
| **Follow-up semantics** | None: it is just a cache | Two defined relationships: continue from a neutral *context* boundary, or follow up on a *readout* snapshot taken at an actual answer position |

### The sliding-window trap, found during qualification

Gemma uses sliding-window attention in some layers. Stock llama.cpp sequence
serialization drops cells outside the window. After a restore, the cache held
fewer cells than the live one and the attention reduction ran over a
different cell count. On a 1,707-token prefix the label logits changed from
`[14.68, 12.71]` (live) to `[14.55, 13.17]` (restored). A naive "save the
prefix cache to disk" design would silently give different answers after a
restart. The snapshot patch adds a flag that keeps the masked cells; with it,
all three restore paths are bit-identical. The price is that K/V bytes grow
linearly past the window.

### The within-request prefix reuse Unridden already had

Unridden v1 already reuses a shared prefix *within one request*: the system
instruction and state are prefilled once and each further question prefills
only its own remainder. That is a prefix cache in the narrow sense, and nothing
survives the request. Snapshots extend reuse across requests, across processes,
across partial depth, and into inspection.

---

## 5. Measured benefits

All figures: one RTX 5090, Gemma 4 26B-A4B UD-Q4_K_XL, 2,048-token context,
profile `split18-30-v1`, 20 repeats, p50 milliseconds, one 56-token two-option
question per branch.

| Prefix length (tokens) | Answer from scratch | Warm branch (resident) | Branch from host RAM | Branch from disk | Create both snapshots | Promote S18 to S30 |
| --- | --- | --- | --- | --- | --- | --- |
| 141 | 28.6 | 23.7 | 29.5 | 25.2 | 40.8 | 22.3 |
| 518 | 102.4 | 24.6 | 36.7 | 32.0 | 140.2 | 67.3 |
| 1,011 | 178.8 | 24.5 | 45.2 | 38.3 | 241.3 | 114.4 |
| 1,765 | 296.9 | 25.1 | 56.8 | 56.1 | 432.8 | 193.2 |

What this shows:

- **Constant-cost questions.** A warm branch costs about 25 ms regardless of
  how long the context is. At 1,765 prefix tokens that is **11.8x** cheaper
  than answering from scratch (resident) and **5.2x** cheaper from host RAM.
- **Fast payback.** Creating the snapshot pair costs about 1.4x one ordinary
  pass. On long contexts it pays back from the second question.
- **Cheap deferred depth.** Promotion (finishing blocks 19-30 on a block-18
  snapshot) costs about 65% of a full pass, consistent with 12 of 30 blocks.
- **No win on short context.** At 141 tokens there is no saving. The feature
  is for long, reused context.
- **Storage is real.** A 1,765-token pair takes about 228 MiB of lower K/V,
  152 MiB of upper K/V and 19 MiB each for `H18` and `H30`. Durable save and
  load (about 1.1 s at that size) are dominated by fsync and single-threaded
  SHA-256, and are the obvious next optimization.

### Correctness gates

| Gate | Result |
| --- | --- |
| Capture integrity (shapes, coverage, byte round trip) | 13 / 13 pass |
| Execution audit (S18 runs only blocks 1-18; promotion only 19-30; zero generated tokens) | 77 / 77 pass |
| Restore identity (resident vs host vs disk in a fresh process) | 51 / 51 bit-identical |
| Promotion identity (S18 promoted then asked equals paired S30 asked) | 38 / 38 bit-identical |
| Branch isolation (A,B,A; reversed order; after failures) | 39 / 39 pass |
| Rejections (bad readout depth, prefix mismatch, budget overflow, corrupt blob) | 28 / 28 pass |
| Stock graph vs split graph | **0 / 38 pass** |

The last gate is the honest caveat. Decisions and label probabilities agree
with the stock single-context graph far inside tolerance (median probability
delta about 1e-8, max 1.8e-7). The raw residual vectors do not meet the
predeclared 1e-3 relative-L2 tolerance on GPU (H30 median 1.2e-2). On CPU the
split and stock graphs are bit-identical at every row, so the split itself is
exact; the GPU difference comes from different kernel and fusion choices for a
different graph shape. The stock GPU graph is itself 4-7% from the CPU result.
The tolerance was not relaxed after the fact, so the profile is labeled
**experimental** and the `/v2` routes are off by default (enable with
`serve --snapshots`).

A second caveat: snapshot answers are reproducible against themselves, not
bit-identical to a from-scratch v1 answer, because chunk boundaries change
numerics on this quantized mixture-of-experts model at long prefixes.

---

## 6. How snapshots enable researchers to steer thinking

"Thinking" here means the model's internal state over a context, not a
written chain of thought. Unridden generates no text. The design documents
state plainly that a snapshot is **not** an interpretable transcript of
thought. What snapshots provide is a controllable, repeatable substrate on
which to run steering experiments. Four levels, from built today to proposed:

### 6.1 Prompt steering by branching (built today)

`POST /v2/state-evaluations` takes a snapshot and a plain prompt such as
*"Consider whether the refund was already issued and focus on the timeline."*
It runs only the new tokens, returns exported vectors and top logits, and with
`save_result_snapshot: true` returns a **child snapshot**. A researcher can
then ask the same typed questions of the parent and of each steered child.

This makes steering a controlled experiment:

- **Same parent, different nudges.** Fork one context into "focus on timeline",
  "focus on customer intent" and "assume fraud" children and compare the answer
  distributions for identical questions. Branch isolation guarantees siblings
  cannot contaminate each other.
- **Exact counterfactuals.** Because the parent is immutable and restores are
  bit-identical, any change in the answer is attributable to the steering text,
  not to cache state, ordering or reload noise.
- **Steering trees.** Children can be steered again, building a tree of
  framings, each node a named snapshot with its parent recorded.
- **Cheap sweeps.** Each branch costs about 25 ms warm, so hundreds of
  framings over a long document are affordable where re-reading the document
  each time would not be.
- **Full distributions, not strings.** Every answer is a probability
  distribution, so steering effects are measured as probability shifts, not as
  differences in generated prose.

### 6.2 Observing where a decision forms (built today)

Snapshots export `H18` and `H30` for every token (in bounded row ranges) and
the final normalized vector. Researchers can:

- Train linear probes on `H18` vs `H30` to ask which information is already
  present halfway through the model and which only appears in the upper blocks.
- Compare a parent's vectors with a steered child's to see which positions and
  which depth a steering prompt actually changes.
- Relate vector movement to the change in the typed answer distribution from
  the same branch.

### 6.3 Intervening at the halfway point (enabled by the architecture, partly built)

The block-18 snapshot is a paused computation: the upper half of the model has
not yet run on the context. The runtime already accepts residual vectors as
input to the upper context. This creates a clean seam for mid-depth
interventions:

- **Early readout.** The earlier proof of concept read a trained head at block
  18 and answered in a median 17 ms against 28 ms on a three-class task (114 of
  120 correct). The snapshot design reserves a slot for registered early heads.
  Today a block-18 readout is refused because no head has been qualified for
  the new prompt profile.
- **Activation steering.** A modified `H18` could be promoted through blocks
  19-30 and its answer compared with the unmodified promotion. The design
  requires control-vector and adapter settings to be part of a snapshot's
  fingerprint so steered and unsteered snapshots cannot be confused; the
  current build records model, runtime, patch and profile hashes, and has no
  steering adapter. The API does not currently accept caller-supplied
  residuals; this is a research extension, not a shipped feature.
- **Depth attribution.** Because promotion of S18 is bit-identical to the
  paired S30, any difference after an intervention at block 18 is caused by the
  intervention.

### 6.4 Latent conditioning (proposed, not built)

The design describes an explicit `latent_conditioning` mode: a trained adapter
that turns a saved `H18` or `H30` matrix into a small set of memory vectors a
new prompt can attend to, without the original text or caches. This would let
researchers test whether the latent state alone carries enough to answer new
questions. It has a predeclared evaluation plan (withhold source text, compare
against mismatched snapshots and question-only baselines, split by scenario)
and no promised quality or speed gain.

---

## 7. Other potential benefits

- **Many questions over one long document.** Legal, support, policy or
  medical records can be read once and questioned many times at near-constant
  cost.
- **Multi-turn decision flows.** A readout snapshot at an actual answer
  position supports follow-up questions that include the earlier question in
  history, while a context snapshot supports replacing the question entirely.
- **Deferred compute.** Take cheap S18 snapshots of many documents; promote only
  the ones that need full-depth questions. Promotions are memoized.
- **Warm starts across restarts.** Disk snapshots are re-indexed on startup and
  restore bit-identically in a new process, so an expensive context survives a
  service restart.
- **Auditability.** Each snapshot records exact token IDs, prompt hash, model
  and runtime hashes and its parent. Responses report the real work performed
  per layer range, so claimed savings are measurable rather than asserted.
- **Evaluation harness.** Researchers can freeze a benchmark's contexts once
  and rerun many question variants or label orders against identical state,
  isolating prompt effects from context-processing noise.
- **Dataset building for probes and adapters.** Snapshots produce paired
  (context state, question, answer distribution) records needed to train probes,
  early-exit heads, or the proposed latent adapter.
- **Privacy-aware storage.** Snapshots follow the original input's ownership,
  store owner-only on disk, and expire by TTL, because caches and activations
  retain the contents of the input.

---

## 8. Limitations to keep in mind

- Experimental: the stock-vs-split residual gate failed on GPU; decisions and
  probabilities pass.
- One model profile only: text-only Gemma 4 26B-A4B, 30 blocks, 2,048-token
  context. The worker rejects other architectures.
- No benefit for short contexts (about 141 tokens or less).
- Storage grows linearly with context, and durable save/load is slow today.
- Confidences are uncalibrated conditional probabilities over the supplied
  options.
- Snapshots are runtime-private; they cannot be transferred across model
  versions, quantizations or runtime builds.
- Hidden-vector export is for research. The project does not claim these
  vectors are human-readable or that they reveal the model's "reasoning".
- One request at a time.

---

## 9. API at a glance

| Route | Purpose |
| --- | --- |
| `POST /v2/snapshots` | Create S18 and/or S30 from context, a decision, or a prompt |
| `POST /v2/decisions` | Ask typed Choice / Score / Noul questions from a snapshot (S18 is promoted automatically and memoized) |
| `POST /v2/state-evaluations` | Append a plain prompt; export vectors and top logits; optionally save a child snapshot |
| `GET /v2/snapshots/{id}` | Metadata |
| `GET /v2/snapshots/{id}/vectors` | Bounded rows of `H18`, `H30` or the last normalized vector |
| `DELETE /v2/snapshots/{id}` | Drop a snapshot and free its bytes |
| `GET /v2/models` | Available snapshot profiles and limits |

Example: create both checkpoints over one context.

```json
POST /v2/snapshots
{
  "model": "local-gemma-unridden-v1",
  "input": {"kind": "context",
            "state": {"message": "The merchant issued a refund last week; it has not arrived."}},
  "checkpoints": [18, 30],
  "persistence": "disk",
  "ttl_seconds": 3600
}
```

Example: steer, then keep the steered state as a child.

```json
POST /v2/state-evaluations
{
  "snapshot": {"id": "snap_30_a", "relationship": "followup"},
  "prompt": "Consider whether the refund was already issued and focus on the timeline.",
  "readout": {"completed_blocks": 30,
              "export": ["last_residual", "last_normalized", "top_logits"]},
  "save_result_snapshot": true
}
```

---

## 10. Suggested questions for this notebook

- Why can't an ordinary prefix cache resume computation from block 18?
- What does the failed stock-vs-split gate mean in practice for decisions?
- How would you design an experiment comparing three steering prompts on the
  same contract?
- What would a probe on `H18` tell you that a probe on `H30` would not?
- When is creating a snapshot not worth its cost?
- What must be true before latent conditioning could be advertised as a
  capability?

## Sources in the repository

- `docs/superpowers/specs/2026-09-22-state-snapshots-design.md`: design
- `docs/superpowers/specs/2026-09-22-state-snapshots-protocol.md`: worker protocol
- `docs/decisions/0006-split-execution-for-state-snapshots.md`: architecture decision
- `docs/results/state-snapshots-qualification.md`: gates, latency and sizes
- `docs/results/prefix-reuse.md`: v1 within-request prefix reuse
- `docs/whitepaper.md`: early-exit proof of concept and determinism findings
