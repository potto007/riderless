# Reusable model-state snapshots at blocks 18 and 30

Status: proposed design, 2026-09-22. No runtime implementation or new GPU results.
Repository baseline: Unridden `7cefca20b625d35c5196081bd5a932a9d56c7bb5`.

## Intended result

Save a model's state after block 18 and after all 30 blocks. A caller can later
submit either snapshot together with a new prompt or a Unridden question and
receive a new state or a typed decision. The original snapshot remains reusable
for independent branches. Decision calls generate zero answer tokens.

The default interpretation is continuation from remembered context. A snapshot
also exposes its hidden vectors for inspection. Explicitly conditioning a new
input on those vectors is a separate extension described below; ordinary cache
continuation does not make that claim. The user has not yet selected between
these interpretations, so this proposal preserves that distinction in the API.

Success means that persisted and restored computation agrees with an
uninterrupted run of the same execution profile, questions remain independent,
and the accounting demonstrates which prior computation was reused. Latency and
storage are measured; the earlier 17 ms PoC timing is not a forecast here.

## Recommendation and alternatives

| Approach | What it enables | Tradeoff |
| --- | --- | --- |
| Complete computation snapshot, recommended | Continue the same model from a saved prefix; finish a partial pass; branch new questions | Saves per-layer attention state and, at block 18, every token's boundary activation. Requires a runtime extension for partial-depth continuation. |
| Hidden vectors plus a learned adapter | Give a separate prompt direct access to a latent representation, potentially across tasks | Needs training and held-out evaluation. No established equivalence to original context. |
| Re-render and process original text | Reference behavior with minimal runtime work | Repeats the computation; use as the comparison baseline. |

Do not feed a block-18 or block-30 vector into the ordinary input-embedding slot.
Matching the embedding width does not make its representation compatible with
token embeddings. Do not describe a three-class probe as an arbitrary-question
reader. The existing trained early head remains confined to its validated task.

## Verified starting points

- `unridden/api/compiler.py` renders a question from state, instructions, and
  ordered choices. `unridden/api/native/worker.cpp` already reuses an exact
  token prefix within a request, and clears state between requests.
- The worker's shared-prefix code excludes a boundary token that could merge
  with a suffix. Snapshot compilation must preserve this protection.
- The earlier local-ai probe exports one 2,816-value raw residual after block 18
  at the final prompt position. It does not export a resumable prefix. Its
  fallback runs the original prompt in a separate full context.
- The local Unridden runtime manifest pins its tested build to llama.cpp
  `v0.4.1`, commit `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`. The original probe
  patch targets the older `afeebe103bd99cda8f5dfaefcabadf890db7fda7` revision and
  cannot be applied as though it were already part of Unridden.
- The installed `llama.h` exposes sequence state save/restore. Its
  `PARTIAL_ONLY` flag concerns SWA/recurrent memory, not a selected block range.
  `ON_DEVICE` state is not a durable byte payload and a later save for the same
  sequence invalidates previous such states. Neither flag is an immutable
  snapshot format.
- Gemma's current graph exposes `h_nextn` after final normalization. The earlier
  early-exit patch used the same output API for a raw intermediate residual.
  The new schema must identify the representation explicitly.

These are source observations, not evidence that the proposed split execution
or snapshot format has passed runtime tests.

## What a snapshot contains

Let `P` be the exact processed token prefix, `N` its length, `d` the hidden width,
and `H18[P]` the residual matrix after block 18, before block 19.

| Component | Snapshot at block 18 | Snapshot at block 30 |
| --- | --- | --- |
| Token IDs, absolute positions, sequence metadata, template cursor | Required | Required |
| Attention K/V for blocks 1-18 | Required | Required |
| Attention K/V for blocks 19-30 | Absent and explicitly marked uncomputed | Required |
| Raw `H18[P]`, shape `[N,d]` | Required to finish the prefix without repeating blocks 1-18 | Retained by reference when derived from an 18 snapshot; optional otherwise |
| Raw `H30[P]`, before final normalization | Unavailable | Captured for latent inspection/conditioning |
| Last-position post-final-norm vector and readout | Unavailable unless an early head supplies its own distinct readout | Stored with representation and position metadata |
| Runtime, model, adapter, tensor-layout, attention and compiler fingerprints | Required | Required |

The raw residual is captured after all operations of the block, including its
output scale and any active control vector. All control-vector and LoRA settings
belong in the fingerprint. The baseline snapshot profile has no steering adapter.

At block 18, capture every processed position in every prefill chunk. The
current last-position export is insufficient. At block 30, disable the graph's
output-row pruning when capturing the full residual matrix. Compute vocabulary
logits only for requested readout positions; do not allocate an `N x vocabulary`
logit matrix just to save a snapshot.

Use lossless tensor storage in the initial implementation. Current residuals
are expected to be F32; record and verify the actual dtype and layout. A full
matrix at `d=2816` costs 11 KiB per token: 22 MiB at 2,048 tokens, or 352 MiB at
32,768 tokens. Saving both residual matrices doubles those figures. KV state is
additional and usually larger; obtain its actual serialized size from the
runtime. Saving state does not save or unload the model weights.

The last-position vector is a useful readout. The whole matrix and caches make
this a computation checkpoint. Their shapes and purposes stay distinct.

## Two places to take a snapshot

### Reusable context boundary

Compile supplied information into a fixed prefix, before any particular
question. This is the preferred anchor for independent questions about the
same information. The prefix uses a new neutral, versioned compiler profile so
both a plain prompt and a finite decision can follow it. This changes the prompt
from Unridden v1 and requires its own quality comparison.

The compiler owns the chat template and creates a verified token splice point.
It stores the exact prefix tokens, the unresolved boundary text/token suffix,
and its rendering cursor. On each branch, render the complete logical prompt
and verify that the frozen token prefix still matches exactly. A mismatch is an
explicit `snapshot_prefix_mismatch`, not permission to patch token IDs in place.
No arbitrary caller-provided special tokens or system-message replacement.

### Existing decision or prompt readout

Also permit capture at the actual answer position of a prepared Unridden
question, or at the final position of a supplied prompt. This preserves the
question-dependent state the user asked to snapshot.

A follow-up to that snapshot includes the old question and any existing answer
prefix in its history. The trusted template compiler closes the unfinished
assistant stub without inserting a selected label, then starts the follow-up
turn. Those structural tokens count as input. The snapshot records that no
answer was generated. Only template profiles with a tested continuation rule
may offer this operation.

Replacing the old question is a different operation: branch from the reusable
context parent. Return that effective parent ID explicitly. Do not silently
substitute a context snapshot for a requested readout snapshot. A readout-only
snapshot with no context parent cannot offer `replace_question`.

## Runtime design

Add an experimental snapshot worker sharing one loaded model with fixed graph
profiles. Keep the existing `/v1/decisions` behavior as the comparison baseline.
The final snapshot runtime uses two layer ranges:

1. **Lower context:** blocks 1-18, receiving ordinary input tokens, storing lower
   K/V, and optionally returning the raw boundary matrix.
2. **Upper context:** blocks 19-30, receiving boundary activations at their
   original positions, storing upper K/V, and applying final normalization and
   the vocabulary head when a readout is requested.

Both contexts share model weights. Each owns its own sequence-position
bookkeeping. This avoids asking one ordinary cache to represent tokens that are
simultaneously processed and unprocessed at different layers. Layer IDs remain
their original global IDs, preserving per-layer RoPE, attention and MoE settings.

This requires new native support for a context's block range and memory-layer
ownership, boundary-tensor input, boundary-tensor output, and durable state for
that profile. The current public state API alone does not provide it. Reuse its
sequence serialization only after the allocator and layer map accurately own
the requested range; otherwise add a dedicated serializer. Never serialize
uncomputed upper-layer buffers from a truncated ordinary context.

Graph identity must include start/end block, input representation, output mask,
capture mode, attention settings and layout. Use fixed contexts and graph-cache
keys, not an environment-variable switch on a reused context. Synchronize before
export, restore or mode transitions. The full reference context is used by the
qualification harness and need not remain allocated in normal operation.

For the first model profile require text-only Gemma 4 26B-A4B, 30 blocks, causal
attention and full retention of SWA cache (`swa_full`). At startup verify the
actual GGUF's hidden width, shared-KV dependencies and per-layer embedding
inputs. The previous assessment found no shared-KV layers or embedding side
inputs for its checkpoint; the new worker must recheck this, not infer it from
the model name. Reject an unsupported dependency across the split. Supporting
another architecture requires enumerating and preserving its additional state.

### Creating the pair

```text
prefix P -> blocks 1-18 -> freeze S18: lower KV + H18[P]
                              |
                              +-> blocks 19-30 -> freeze S30: lower KV + upper KV
                                                   + H30[P] + final readout
```

If only S18 is requested, upper blocks must not run. When both are requested,
S30 records S18 as its parent and its graph profile. The pair comes from the
same lower computation. Publish the requested set atomically only after every
checkpoint is complete. A failure while constructing S30 exposes neither ID and
releases temporary state. A later, separate promotion call can fail without
invalidating its already published S18 parent.

### A new question from S30

Restore or branch both caches. Compile the new question into suffix `Q` at
positions `N..N+len(Q)-1`. Process only Q through the lower context, then through
the upper context attending to the saved prefix's upper K/V. Read the new last
position's label logits and apply the existing Choice/Score/Noul mapping.

Each sibling question starts from the same immutable parent and never sees
another sibling's suffix. Normal query calls discard their branch when finished;
an explicit `save_result_snapshot` creates a child snapshot.

### A new question from S18

For a registered early head, process Q through the lower context and read that
head at the new answer position. The old banking head is not available for
arbitrary question schemas or for a changed prompt profile without validation.

For a generic question, first finish P through blocks 19-30 using the stored
`H18[P]`. This creates the missing upper prefix cache and a derived S30. Then
process Q as above. Memoize the derived S30 by parent ID and execution profile
so subsequent questions do not repeat promotion. The parent remains unchanged.
The first call pays promotion cost; later calls can reuse it.

For an early-head call that falls back, retain `H18[Q]` until routing finishes.
Promote the parent prefix if needed, then run those suffix activations through
the upper context. This can remove the old PoC's second lower-layer pass. It is
a proposed optimization requiring verification and does not establish that
fallback improves accuracy.

Promotion processes prefix chunks in causal order and verifies upper cache
positions after every chunk. It never treats a single final-position residual
as all prefix activations. It never skips the first 18 blocks for new tokens.

### What continuation actually uses

New tokens use the saved per-layer keys and values. `H18[P]` is directly consumed
when completing the unfinished upper computation. Once the full caches exist,
normal continuation does not directly inject the exported `H30[P]` matrix into
new tokens. No blocks remain after block 30 to resume on that matrix alone.
Re-scoring the old position through its existing output head can reuse H30,
but a new semantic question requires suffix computation or a trained adapter.

Consequently, a generic full-depth question from S18, after promotion, should
agree with the same question from its paired S30. This compares two ways of
reusing computation, not two levels of decision capability. To study what the
18-block representation can answer on its own, keep the readout at block 18
with a qualified head, or test the latent-conditioning adapter on H18 without
access to upper caches. The API reports which operation actually occurred.

## Proposed API

Use versioned `/v2` routes for this extension. Existing v1 callers remain on the
current contract. Snapshot IDs are opaque, immutable references owned by the
caller/service namespace; clients do not submit paths or tensor arrays in JSON.

Create both checkpoints:

```json
POST /v2/snapshots
{
  "model": "local-gemma-unridden-v1",
  "input": {
    "kind": "context",
    "state": {"message": "The merchant issued a refund last week; it has not arrived."}
  },
  "checkpoints": [18, 30],
  "persistence": "disk",
  "ttl_seconds": 3600
}
```

`input.kind` is a tagged union: `context` carries state; `decision` carries state
and one existing Unridden question; `prompt` carries validated messages.
Only context input supports the neutral reusable-prefix contract. Decisions and
prompts produce literal readout snapshots with the follow-up semantics above.

```json
{
  "snapshots": [
    {"id": "snap_18_a", "completed_blocks": 18, "boundary": "context", "capabilities": ["continue", "promote", "inspect"]},
    {"id": "snap_30_a", "completed_blocks": 30, "boundary": "context", "parent": "snap_18_a", "capabilities": ["continue", "inspect"]}
  ],
  "usage": {"generated_tokens": 0}
}
```

Ask independent typed questions; using `snap_18_a` causes explicit promotion:

```json
POST /v2/decisions
{
  "model": "local-gemma-unridden-v1",
  "snapshot": {"id": "snap_18_a", "relationship": "followup"},
  "questions": {
    "status": {
      "type": "choice",
      "instructions": "What should the support team investigate?",
      "criteria": {"new_request": "A refund has not been issued", "missing_refund": "A refund was issued but has not arrived"}
    }
  },
  "readout": {"completed_blocks": 30},
  "save_result_snapshot": false
}
```

Return the familiar answers plus `snapshot_usage`: requested/effective parent,
promotion performed or reused, profile, suffix length, per-range processed
tokens, reused prefix tokens, restored bytes, and restore/promotion/inference
times. Block-token work is reported separately from logical input tokens so
finishing 12 blocks on N tokens is not misreported as zero work or as a complete
30-block prefill. Each early result names its exact registered head and
calibration version. Full label probabilities retain current conditional-score
semantics, without claiming calibrated correctness.

Plain prompts are accepted without forcing a Choice schema:

```json
POST /v2/state-evaluations
{
  "snapshot": {"id": "snap_30_a", "relationship": "followup"},
  "prompt": "Consider whether the refund was already issued and focus on the timeline.",
  "readout": {"completed_blocks": 30, "export": ["last_residual", "last_normalized", "top_logits"]},
  "save_result_snapshot": true
}
```

This returns a child snapshot ID and bounded diagnostics/artifact references.
It generates zero tokens and makes no claim that the exported vector is a
human-readable answer. Callers wanting a decision supply a Unridden question;
text generation would be a separate explicit output mode outside this design.

Add metadata lookup, expiry/delete, and bounded vector artifact retrieval.
Expose available snapshot profiles through `/v2/models`. A request above the
combined prefix/suffix context limit is rejected before inference; never drop
prefix tokens or shift positions to make it fit.

## Storage, restoration and branching

Persist a bounded manifest plus typed tensor blobs and sequence-state blobs.
The manifest records schema version, model/tokenizer hashes, exact runtime and
patch hashes, execution profile, block coverage, layer map, dtype, RoPE/window
settings, positions, token IDs, template version/cursor, tensor checksums, parent
IDs, owner, expiry, and sizes. Model weights are referenced by fingerprint.

Store native sequence blobs as runtime-private artifacts, not a universal model
format. Load only into an exact compatible profile in the initial version.
Cross-version, cross-quantization and cross-model transfer need a separately
qualified conversion. Save final vectors explicitly; do not assume sequence
state serialization includes output buffers. Do not store raw GPU pointers.

Publish using a temporary directory, flush/checksums, then an atomic rename.
Reopen and validate a snapshot before declaring it durable. Bounds-check all
lengths and tensor dimensions before calling a native loader. Initial API
supports service-created snapshots only; external binary imports are excluded.
For `persistence: "disk"`, publish before a successful API response and rebuild
the snapshot index on service startup. The store manifest must include the
messages and answer prefix needed to compile future branches, as well as their
original owner and expiry. The snapshot directory and manifest are owner-only
because they contain the supplied state as readable text. Store manifest schema
2 adds these fields; schema 1 disk artifacts cannot be resumed by the API.

GPU-resident parents use sequence branching only within a compatible context;
`llama_memory_seq_cp` is a candidate, subject to the split-context tests. Pin
parent cells and prevent eviction, position shifts or SWA recycling while a
branch references them. Start with serialized requests, as Unridden already
does. Clear only child suffix state on completion or cancellation. Failed native
restore invalidates its scratch context before another request can use it.

Evict inactive snapshots to host/disk under explicit byte budgets and TTLs;
active leases keep their required blobs alive. Shared blobs remain until their
last parent/child reference expires. Snapshot authorization follows the original
input's ownership because caches and activations can retain its contents. The
store must not serialize another sequence's data from a shared allocation.

Storage and copying may dominate short requests. Report resident and disk-restore
latency separately. Keep the first profile at the existing 2,048-token context;
qualify larger contexts separately. Preserve multi-GiB GPU headroom and load the
weights once. Do not assume two contexts cost the same as one or claim savings
from the number of skipped blocks alone.

## Direct vector conditioning extension

If the intended input is specifically the saved hidden representation, introduce
an explicit `usage_mode: "latent_conditioning"` and a versioned trained adapter.
Never implement this mode by silently restoring caches or replaying source text.

One candidate consumes the full selected boundary matrix through a learned
resampler, producing K memory vectors, followed by a learned projection to a
model-compatible prefix embedding space. A layer/normalization tag distinguishes
H18 from H30; initially train and qualify separate adapters. New prompt tokens
attend to that adapted latent prefix. The base model can stay frozen. A direct
cross-attention adapter is an alternative with a larger runtime change.

Training pairs contain a saved state, a new question, and independent target
answers or teacher distributions. Split by source scenario so questions about
one source cannot leak across train/test. Start with the full matrix; compare a
last-vector-only ablation instead of assuming one position retains all relevant
facts. Saving K memory vectors is compression and may lose information.

At evaluation, withhold source text and original KV from this mode. Compare
against fresh full-context inference, cache continuation, question-only input,
and shuffled/mismatched snapshots. Rotate label assignments, test unseen
questions, and measure probability quality and retained facts. This establishes
whether the result uses the state and whether it is useful. It is a separate
research milestone with no promised quality or speed gain.

Neither continuation nor latent conditioning is evidence that a snapshot is an
interpretable thought transcript. Both can support operational decisions if
their readouts pass the task's evaluation.

## Verification and release criteria

Before any speed claim, freeze model/runtime/profile and test vectors. Keep CUDA
fusion enabled in the intended profile. Graph splitting and extra outputs can
change fusion, so compare instrumented and uninstrumented execution separately.

1. **Capture integrity:** validate token coverage, position order, `[N,d]`
   dimensions, exact tensor-byte round trip, per-layer cache coverage and no
   uninitialized layers. Compare exported residuals with audit-only callbacks.
   Stage tags distinguish pre-final-norm residuals and post-norm head inputs.
2. **Execution audit:** S18 creation executes only blocks 1-18; promotion only
   19-30 on old prefix tokens; new suffix tokens traverse their requested blocks.
   Every decision and state-evaluation record has zero generated tokens.
3. **Restore identity:** compare an uninterrupted split-profile run with an
   immediate snapshot branch, disk restore in a fresh process, and restoration
   after eviction. Use identical tokenization, chunk sizes and positions. Require
   identical class choices, max label-probability delta <= 1e-5, and relative L2
   <= 1e-5 for corresponding residuals; retain exact tensor-byte equality for
   serialization. A violation is a failed gate, not a reason to relax it later.
4. **Stock-versus-split qualification:** compare the new profile with the frozen
   ordinary 30-block baseline. Predeclare relative L2 <= 1e-3 and max probability
   delta <= 1e-3, with equal decisions on the frozen qualification corpus. The
   previous early-exit experiment failed a similar gate because of fusion, so
   passing is uncertain. If it fails, keep the profile experimental and report
   drift; broader deployment requires a separately reviewed validation protocol.
5. **Branch isolation:** ask A, B, A; reverse question order; change adversarial
   sibling content at fixed token length; repeat after a cancelled/failed branch.
   Parent checksums and same-profile outputs must remain stable. Test both 18
   and 30 parents and promotion reuse. Reject wrong model, runtime, template,
   block coverage, corrupt blobs and token-budget overflow.
6. **Template semantics:** compare context branches and readout follow-ups with
   freshly rendered, identical logical histories. Cover Unicode/token-boundary
   merges, assistant stubs, multi-chunk prefill, exact context limits and empty
   or unsupported suffixes. A replacement-question test must prove it used the
   context parent and did not retain the old question.
7. **Task quality:** run the existing Unridden suites plus a new frozen set of
   follow-up questions and ambiguous cases. Keep original-v1, new-compiler and
   split-runtime effects separate. Do not reuse the PoC's temperature or 0.71
   routing gate without qualifying the new graph/compiler path.
8. **Cost:** report creation, durable save, host/disk restore, first promotion,
   warm branching, p50/p95, serialized bytes, peak host/GPU memory and actual
   block-token work. Compare one and many queries per parent at several allowed
   prefix lengths. Include failed and cancelled operations in resource tests.

For latency experiments disable audit callbacks and retain the real boundary
outputs needed by the feature. Use paired runs and document graph profile and
chunk shape. Corpus construction and tolerances are fixed before outcomes are
seen. No production service restart is required for a standalone harness.

## Delivery sequence and component boundaries

1. Build a full-depth durable snapshot prototype using native sequence state
   APIs, with exact prompt-splice validation, immutable branches and a fresh-run
   comparison. Tag it `whole30-v1`; do not mistake it for completed 18 support.
2. Add fixed lower/upper graph profiles, all-position boundary capture, layer
   memory ownership and identity tests in an isolated llama.cpp patch. Tag it
   `split18-30-v1`; profiles are not silently interchangeable.
3. Add S18 persistence and promotion, paired S18/S30 capture, prompt/readout
   boundary handling, request isolation, then the versioned Unridden endpoints.
4. Qualify task quality and cost. Only then consider an early-head cascade that
   continues retained suffix activations instead of rerunning them.
5. If direct vector input is required, run the adapter experiment as a distinct
   milestone and advertise the capability only after its evaluation passes.

Keep concerns separate: a Python snapshot schema/compiler, a store and lease
manager, a native range-execution adapter, and a Unridden response mapper.
Proposed locations are `unridden/api/snapshots/` and
`unridden/api/native/snapshots/`, plus a recorded runtime patch. Extend the
worker command protocol with explicit create/promote/evaluate/inspect/drop
operations. Large tensors stay in the native/store boundary, not the existing
4 MiB JSONL response stream. The API passes artifact IDs and bounded metadata.

This document is the design deliverable. Implementation and measured feasibility
of the split runtime remain subsequent work.

## Sources

- Unridden baseline: `unridden/api/compiler.py`,
  `unridden/api/native/worker.cpp`, `unridden/api/README.md`.
- Local runtime inspected: `build/llama-base-v0.4.1/build.json`,
  `headers/include/llama.h:869-937`, `headers/src/llama-kv-cache.cpp:2236`,
  and `headers/src/models/gemma4.cpp` below that build directory.
- [Pinned Gemma graph](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/src/models/gemma4.cpp)
  identifies residual, normalization, output pruning and head boundaries.
- [llama.cpp state API](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/include/llama.h)
  provides existing sequence serialization. Its presence does not establish
  partial-depth continuation.
- [Transformer cache explanation](https://huggingface.co/docs/transformers/cache_explanation)
  describes the per-layer context required when processing appended tokens.
- Earlier local-ai evidence:
  the separate local-ai repository's `docs/research/activation-probe-early-exit-results.md`,
  `activation-probe-poc-results.md`, and
  `intermediate-state-control-assessment.md` in the same directory; runtime
  experiment at `systemone/probes/early_build.py` in that repository.
