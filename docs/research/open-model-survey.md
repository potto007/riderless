# Survey of open decision-readout projects

Eight public projects that answer typed questions by reading logits, or by
reading a trained decision head, were read against riderless in September 2026.
Each was cloned read-only, at depth 1, and read as source; nothing was installed
and nothing was run. This document records what each one is, which of their
ideas are worth taking, and what riderless already does that none of them do.

Everything a candidate's own README asserts about its accuracy, its latency, or
any hosted service is that project's claim. Claims are labelled as claims here
and none of them are repeated as fact. No measurement of any hosted service
appears in this document.

Licences below are read from each clone's `LICENSE` file. Weight licences are
not verified here and must be checked on the model cards before any comparison
is published.

## The candidates

**jaredpalmer/kev** (Apache-2.0). A family of small trained decision models
(LoRA adapters plus a learned pointer head on Qwen3.5 bases) served behind a
FastAPI endpoint, with the bulk of the repository given over to data generation,
frozen evaluation suites, and metrics. Its readout is a pointer head over option
spans rather than a vocabulary label, so it has no letter alphabet and no
26-option ceiling (`kev/kev/model.py:132-145`).

**NandhaKishorM/laya** (Apache-2.0). A pip-installable library, not a service,
that answers the same three primitives with a purpose-trained bidirectional
encoder of 322M to 421M parameters and a marker-position readout
(`laya/laya/common.py:49-136`). It ships fitted temperatures, an entropy-based
confidence, and a benchmark harness under `research/`.

**ikermoel/open-alternative-jev** (Apache-2.0, import name `so1`). About 330
lines of library that read option-letter probabilities out of any open-weights
ChatML model through Transformers or vLLM, with no server and no provenance.
Its value is the benchmark methodology: a padded-shape control, an order
rotation control, cluster bootstrap intervals, and a published retraction of its
own earlier speed claim.

**razorback16/openjev** (Apache-2.0). A Jev-shaped FastAPI shim in front of a
vLLM server running a discrete-diffusion Gemma variant, reading every question
of a request from one shared denoise canvas (`openjev/openjev/engine.py:220-274`).
Structurally the opposite of riderless: one prompt and one canvas per request,
with a re-read triggered by the maximum entropy across the whole group.

**Rizzo-AI-Academy/rizzo-flow** (Apache-2.0). The closest sibling: a local
zero-generation decision API with single-letter labels, a final-position logit
readout, a shared state prefix, and a Jev-compatible route, serving 1.7B and 4B
GGUF models through a hand-written ctypes binding to a prebuilt llama.cpp. It
adds abstention slots, a numeric primitive, offline calibration bound to a
configuration fingerprint, and a perturbation harness.

**bnsd55/jevmlx** (MIT). A large Apple-Silicon/MLX library (roughly 23k lines)
that compiles a schema of boolean, enum, and multi-select fields into one
batched prefill, with calibration, abstention, prior correction, cross-field
constraints, an eval pipeline, and its own Jev-shaped endpoint. Its native path
is hard-gated to macOS arm64, so it cannot run on this hardware.

**rupeshpoojary9/poorjev** (MIT). A small library that reproduces the three
primitives on a commodity zero-shot NLI encoder, scoring each option as an
independent entailment query and renormalising across the option set. Its stated
purpose is calibration, and its metrics module is the cleanest small reference
for ECE, risk-coverage, and AURC.

**yunhai-dev/laya2typesafeapi** (no licence file). A 162-line FastAPI wrapper
that exposes the laya library over HTTP with optional bearer auth, containing no
decision logic of its own. It is listed for completeness; its interest is
entirely in the laya wheel it pins.

## Ranked adoption list

Ranked by value to riderless against effort and risk. Every item below is either
pure post-processing over data riderless already records, harness work, or an
opt-in mode. Nothing here weakens per-question isolation, the zero-generation
contract, or provenance on the default path.

### 1. Fit a temperature offline and publish a calibrated confidence beside the raw one

Sources: `kev/kev/metrics.py:190` (`fit_temperature`, log-spaced 81-point grid
from 0.25 to 4, minimising NLL, and refusing rows that were already calibrated),
`poorjev/src/poorjev/calibration.py:57` and `:88` (`cross_val_calibrate`, fitting
on k-1 folds and reporting ECE only on the held-out fold),
`open-alternative-jev/so1/calibration.py:55` (`cross_fit_temperature`),
`laya/laya/common.py:219` and `:228-232` (per (question type, option count)
bucket, with every shipped value clamped into [0.5, 5.0] and a warning naming
each rejected bucket), `rizzo-flow/src/rizzo_flow/calibration.py:16-23` (the
artifact carries a configuration fingerprint and a literal status of
`fitted_requires_held_out_validation`).

Riderless's first documented limitation is that confidence is uncalibrated and
saturated (`README.md:220-224`), and the whitepaper names a temperature fit as
the first thing to do next (`docs/whitepaper.md:331-334`). The raw label logits
of every frozen run are already on disk, because `riderless/api/mapping.py:110`
writes `raw_label_logits` into diagnostics. So the fit costs no inference at all.

Four details are worth copying rather than reinventing: bucket by question type
and option count rather than fitting one global scalar; fit and evaluate on
disjoint partitions; clamp and warn rather than accept a fitted value below 1;
and bind the fitted table to the configuration that produced it, the way
riderless already binds model and runtime hashes (`riderless/api/mapping.py:46-50`).

Two riderless-specific hazards the candidates do not state. A positive
temperature cannot move a Choice argmax, but Score is
`sum(i * p_i)` (`riderless/api/mapping.py:99`) and Noul is `P(true)`
(`riderless/api/mapping.py:105`), so a temperature does move two of the three
answer types. And fitting on the same seven suites that produced the published
accuracy would be circular. Effort medium. Risk none if it stays a reported
artifact; if it is ever applied at the API it needs its own
`confidence_formula` value beside `max_probability_v1`
(`riderless/api/schema.py:142`).

### 2. Calibration and selective-prediction metrics in the suite scorer

Sources: `poorjev/src/poorjev/metrics.py:43,54,80,95,117`
(`brier_toplabel`, `reliability_bins`, `ece`, `risk_coverage`, `aurc`, all pure
stdlib), `kev/kev/metrics.py:103,135,142` (`coverage_at_error`,
`risk_coverage_curve`, `area_under_risk_coverage`, walking the confidence order
in whole tie groups), `jevmlx/jevmlx/evalmetrics.py:184,211,578`
(`correctness_auroc`, `ece_equal_mass`, `risk_coverage_curve`),
`laya/research/scripts/build_benchmark_nb.py:247` (`aurc` plus accuracy at 50%
and 80% coverage), `rizzo-flow/src/rizzo_flow/evaluation.py:86-92,126-128`
(scalar ECE, gold-label NLL, multi-class Brier in the summary block).

Riderless already computes the inputs and stops one step short: the scorer
builds a ten-bin reliability table at
`scripts/riderless/run_usecase_suites.py:440`, a majority baseline at `:462`, and
a Brier for one answer type at `:507`. Adding scalar ECE, gold-label NLL,
top-label Brier, AURC, and coverage at an error budget turns the prose caveat at
`README.md:220-224` into numbers, and is the only way to show whether a fitted
temperature helped.

The tie-group handling in `kev/kev/metrics.py:103` matters specifically here,
because riderless's confidences are saturated near 1.0 and a naive index cut
would split a large tie at 0.999 arbitrarily. Equal-mass binning
(`jevmlx/jevmlx/evalmetrics.py:211`) is the same defence on the reliability
table. Effort small. Risk none: scoring-side only, no change to the worker, the
compiler, or the response shape.

### 3. Option-order permutation suite: flip rate and probability movement

Sources: `laya/research/scripts/build_benchmark_nb.py:638-676` (permute each
choice question's options with a fixed seed, map the argmax back through the
permutation, report the flip fraction per suite), `kev/kev/suite.py:135-165`
(`contrast_cases` bakes a `permuted` variant into the frozen suite with a
`parent_id` back-pointer, and the scorer realigns the permuted row's
probabilities to the parent's option order before computing max delta and flip
rate), `jevmlx/jevmlx/evalmetrics.py:466,530` (`perturbation_flip_rate`,
`mean_tvd_across_permutations`), `rizzo-flow/scripts/semif_compare.py:149-198`
(`stability`: per-variant balanced accuracy, argmax flip count with the flipping
row ids, and mean and max per-option probability movement).

This is the measurement riderless's own design makes most load-bearing and has
never taken. Options are assigned to letters in caller order
(`riderless/api/compiler.py:107,114-119`), so permuting the options re-binds
every option to a different letter, and both a positional prior and a letter
prior land straight on the answer. Riderless proves sibling isolation at exactly
0.0 (`docs/architecture.md:117-120`) and publishes about 1% movement under a
batch-shape change (`README.md:229-232`), while the sensitivity that sits between
those two is untested. The harness already has everything needed: the suites are
JSON, option ids round-trip through `probabilities`, and
`scripts/riderless/compare_observations.py` already diffs two runs.

Copy kev's realignment step and rizzo-flow's assertion that the semantic option
set is unchanged, or the comparison silently measures the wrong thing. Effort
small. Risk none to the product; the exposure is that the number may be
unflattering, which is the reason to measure it rather than not to.

### 4. Publish margin and normalised entropy beside the saturated confidence

Sources: `jevmlx/jevmlx/api.py:224,458` (`probability_margin`, top-1 minus
top-2, a first-class field on every scalar result, with one `reason` field for
abstention rather than a parallel boolean, `:425`),
`laya/laya/common.py:210-216` (`confidence_from_probs`, `1 - H(p)/log k`),
`rizzo-flow/src/rizzo_flow/decisions.py:97,116-117` (entropy in nats plus a
concentration normalised by `log n` and clamped to [0,1]),
`openjev/openjev/engine.py:401-405` (the same entropy form).

Riderless returns `confidence = max(probabilities)`
(`riderless/api/mapping.py:88`), which saturates and is not comparable across
option counts: 0.6 over two options and 0.6 over twenty mean different things.
Both alternatives are deterministic pure functions of the probability vector
riderless already computes, so they cost nothing at inference and can be added
as named additive fields or diagnostics without redefining `confidence`, which
is pinned as `max_probability_v1` at `riderless/api/schema.py:142`.

Pair this with item 2: AURC over the same suites says which of max probability,
margin, and normalised entropy actually ranks errors best, without choosing a
threshold. Effort small. Risk none computationally; the only cost is a versioned
wire addition. The presentational risk is real: a second number called
confidence would be read as calibrated, so these must be named for what they
are.

### 5. Cluster bootstrap intervals over correlated cases, with a population guard

Sources:
`open-alternative-jev/benchmarks/scripts/benchmark_v2.py:77-85,101-107`
(`cluster_bootstrap` resamples whole groups, not rows, 2000 resamples at a fixed
seed, and every mode comparison reports a paired accuracy delta with its 95%
interval alongside answer disagreement and mean and max absolute probability
difference), `kev/kev/metrics.py:209` (`paired_bootstrap`, which refuses to
compare two runs unless they cover an identical set of examples with identical
option order and labels, and recomputes non-additive statistics inside every
resample), `jevmlx/jevmlx/evalmetrics.py:643` (`cluster_bootstrap_ci`, with
sibling functions whose docstrings say when a Wilson interval is not the right
instrument).

Riderless publishes paired comparisons whose significance is unstated: 0.928
against 0.925 across the label revision (`README.md:264-268`), the v0.4.1
revalidation, the batched-mode comparison, and the worker-build comparison.
`docs/results/usecase-suites.md` already identifies the clustering, which is the
hard part, and `docs/whitepaper.md:306-307` falls back to a rule of thumb. One
case can carry many correlated questions, so resampling at the case level rather
than the question level is the correct unit and turns that caveat into a
computed interval. Effort small to medium. Risk none: offline statistics over
recorded outputs. Copy kev's refusal to compare non-identical populations, which
is what stops an accidental apples-to-oranges claim.

### 6. A padded-shape control that separates kernel numerics from batch effects

Source: `open-alternative-jev/benchmarks/scripts/benchmark_v2.py:7` (the mode
docstring: "same as A, right-padded to a fixed length: isolates shape/kernel
noise from interference"), `:199` (pad length rounded up to a multiple of 64),
`:240` (the padded length is used only in that mode), with the flip accounting at
`benchmarks/scripts/analyze_run.py`.

This is the control missing from riderless's most hedged number. `README.md:229-232`
records that about 1% of borderline answers move when the prefill batch shape
changes, and `docs/whitepaper.md:313-314` concedes the isolation guarantee is per
configuration. Today that movement is one undifferentiated figure. The riderless
version is a sequential run at the batched mode's batch and ubatch shape but one
question per request: the difference against the ordinary sequential run is pure
kernel numerics, and whatever batched mode moves beyond that is attributable to
the batch itself. It serves the whitepaper's second stated next step directly
(`docs/whitepaper.md:332-333`).

The same source contributes two cheap companions. Regress elapsed time on padded
token count per mode before claiming any speed win
(`open-alternative-jev/benchmarks/scripts/analyze_run.py:69-74` reports a fixed
seconds-per-call intercept, ms per padded token, a Pearson r, and padding waste);
that is the audit that would replace riderless's extrapolated no-reuse figure
(`README.md:270-273`) with a fitted line. And interleave comparison arms with a
per-group seed rather than running them in blocks
(`benchmarks/scripts/benchmark_v2.py:260-269`), because on this machine a
throughput decay over tens of minutes would otherwise be attributed wholesale to
whichever arm ran second. Interleaving must alternate processes or requests, not
mix configurations inside one measurement.

Effort medium. Risk none: it is a measurement configuration built from flags the
worker already takes.

### 7. Near-tie rescore at the canonical single-question shape, inside batched mode

Source: `jevmlx/jevmlx/engine.py:1006` (`finalize_scalar_evidence`, the one place
where rescore, prior, temperature, and margins are applied), `:1014` and
`:1035-1054` (the band is `INSTABILITY_BAND` widened by a measured drift envelope
rather than a bare constant), `:856` (`INSTABILITY_BAND = 5e-2`),
`jevmlx/jevmlx/driftenv.py` (why the constant band was not enough, and the rule
that the parity gate stays fixed while only the rescore trigger widens).

Riderless's batched mode deliberately gives up sibling independence
(`docs/decisions/0004-optional-batched-question-evaluation.md`,
`docs/architecture.md:126-133`). A near-tie rescore reclaims most of what that
costs: when a question's top two label logits sit within a band, re-evaluate
that one question alone on the sequential path, which is the shape riderless
already proves at exactly 0.0 delta, and return that result. The other questions
keep the batched speedup.

This strengthens the opt-in mode rather than weakening a guarantee, and it uses
a code path that already exists in the worker. Two costs: per-request latency
becomes data-dependent, and the band constant becomes part of the configuration,
so it has to travel in diagnostics and in provenance or two builds would
silently disagree. Each answer must name which shape produced it, the way
`evaluation_mode` and `batch_sequences` already do
(`docs/architecture.md:178-185`). Effort medium.

### 8. Skip rather than break when validating the label alphabet

Source: `openjev/openjev/engine.py:84-97` (`_single_token_labels`: probe each
candidate against a carrier string, keep it only if it adds exactly one unseen
token, and continue past a failure or a duplicate instead of stopping).

Riderless's `validate_alphabet` (`riderless/api/native/worker.cpp:303-334`) runs
a stricter probe, against the real chat template and the real answer prefix, but
it breaks out of the loop on the first tokenise failure or the first duplicate
token id rather than skipping that candidate. On a tokenizer where one letter
behaves differently after the answer prefix, the validated alphabet silently
truncates to everything before it and the option capacity collapses, even though
the remaining letters are fine. Skipping keeps the alphabet maximal and is
guarantee-neutral: the alphabet is still validated at startup and re-checked per
prompt (`riderless/api/native/worker.cpp:292-299`).

The capacity extension openjev also does, appending lowercase and two-letter
candidates past Z, is a separate and much larger question: it would change the
options block for any question over 26 options and is a new configuration rather
than a bug fix, so every frozen number would need re-measuring. Effort small for
the skip; the extension is not proposed here.

## Considered and not proposed now

- **Nonce-fenced or tag-fenced state block.** `jevmlx/jevmlx/engine.py:2387-2408`
  derives a fence tag from the sha256 of the context itself so no interior line
  can close the block; `rizzo-flow/src/rizzo_flow/prompts.py:47-55` wraps the
  state in `<evidence>` tags and falls back to JSON when the state itself spells
  the closing tag, with a system line saying the fenced region is data and never
  instructions (`prompts.py:14-22`). Riderless builds its prompt from bare
  markers (`riderless/api/compiler.py:108-120`), and its control-token rejection
  (`riderless/api/native/worker.cpp:170-177`) by design permits ordinary markup,
  so a state containing the literal lines `OPTIONS:` and `A: something` renders
  as if it were structure. The fix is small, but it changes the compiled prompt,
  which bumps `PROMPT_VERSION`, invalidates every prompt hash, and requires
  re-measuring all seven suites plus the reuse and batched results. It should
  ride along with another prompt change rather than be spent alone.
- **An "unknowable" partition scored on confidence only.**
  `kev/kev/transfer_v9.py:83-111` drops the one sentence whose removal makes the
  rule evaluator return undetermined and keeps the intact sibling as a paired
  control; `kev/kev/metrics.py:167` scores those rows on confidence alone, and
  `kev/kev/benchmark.py:83` excludes them from every accuracy number in code
  rather than by convention. Riderless's hierarchical suite already pairs
  determined and underdetermined twins, so the design idea exists in one suite;
  making it a cross-suite partition with the exclusion rule enforced in the
  scorer would give a number for the failure mode the confidence caveat only
  warns about.
- **Blind confident-failure review with confidence-matched controls.**
  `kev/scripts/calibration_audit.py:59` pairs every wrong answer above 0.9
  confidence with a randomly drawn correct answer from the same task and option
  count, shuffles them, and writes the blind cases separately from the key, with
  a recorded status that model and teacher disagreement is not a label
  correction (`:89`). Riderless's suites are hand-authored, so some share of its
  wrong answers may be gold-label disputes; this is the only bias-resistant way
  to find out, and the rule that a blind review never edits a frozen label has to
  be enforced or the suites stop being frozen.
- **Committed golden prompt vectors, and a CI switch that fails on skipped
  model tests.** jevmlx commits the rendered prompt, its token-id hash, and the
  plan hash per tokenizer revision and diffs them in CI
  (`jevmlx/PROMPT_PROTOCOL.md:1-18,124-146`); open-alternative-jev turns a
  model-load skip into a hard failure under an environment flag
  (`open-alternative-jev/tests/conftest.py:10-18`). Riderless hashes the prompt
  per request but its contract tests compare compiled message structures rather
  than bytes rendered through the model's chat template at a pinned revision.
  The scheduled half of that idea, a weekly unpinned revalidation, is not
  proposable here: it would mean building llama.cpp master and loading a 26B
  model unattended on a shared card.
- **Grouped batching instead of an all-or-nothing fallback.**
  `openjev/openjev/engine.py:200-213` splits an over-budget question list into
  the fewest contiguous groups that fit. Riderless answers the entire request
  sequentially when the batched cell count exceeds the batched context
  (`riderless/api/native/worker.cpp:719-729`). Grouping would recover most of the
  speedup, and the per-question `evaluation_mode` and `batch_sequences` fields
  already make a grouped response self-describing. It adds a new source of
  variance inside a mode that has already given up sibling independence, so it
  belongs after item 7, not before it.
- **A one-option question answered without reading the model.**
  `openjev/openjev/engine.py:117-131,354-360` resolves a single-option choice to
  probability 1.0, never sends it, and merges it back preserving caller order.
  Riderless rejects it at the wire (`riderless/api/schema.py:60,73`). Doing this
  properly means marking the answer as not model-derived, because it has no
  prompt, no prompt hash, and no logits; doing it sloppily means a response where
  some answers silently carry weaker provenance than others, which is worse than
  the 422.
- **Self-reported device memory as a delta from the pre-load baseline.**
  `rizzo-flow/src/rizzo_flow/backend_llama.py:161-171` records the lowest free
  device memory seen against an idle baseline captured before the load, and says
  in the report that other processes are counted too. Riderless currently
  footnotes its 22.5 GiB card peak with a 4.83 GiB unrelated baseline
  (`README.md:114-118`); this is that subtraction done by the process itself.

## Rejected

- **Escaping caller control tokens instead of rejecting them**
  (`kev/kev/model.py:20-26` rewrites `<|name|>` to a lookalike). Riderless
  returns 422 `unsupported_content` instead. Escaping means the text the model
  saw is not the text the caller sent, and two different inputs could compile to
  the same prompt hash. Rejecting is louder and cannot produce a silently
  different answer. Recorded so it is not re-proposed.
- **Server-side derived facts injected into the state** (`kev/kev/api.py:66-91`
  appends computed date relationships). Riderless compiles exactly the state the
  caller sent and hashes the result; injecting derived sentences makes the
  service a silent author of prompt content even when opt-in. This is a
  client-side recipe.
- **Cross-question conditioning.** openjev writes earlier chunks' answers into
  later prompts (`openjev/openjev/engine.py:363-385`); jevmlx maximises a joint
  assignment under constraints and re-scores children under a `Given:` header
  (`jevmlx/jevmlx/engine.py:1729-1832,1855-2196`). Both are direct answers to
  riderless's documented trap that a question cannot see its siblings
  (`README.md:242-244`), and both destroy the guarantee riderless can prove and
  they cannot. If it is ever wanted, it belongs in a caller-side second request
  with the first answer placed in the state, where it stays visible in the
  provenance.
- **Cross-request caches.** jevmlx caches a neutral-context prior in a
  process-wide LRU (`jevmlx/jevmlx/engine.py:1304-1368`), kev caches a state
  prefix across
  callers (`kev/kev/serve.py:43-56`), openjev caches prefill on its MLX path.
  Riderless's isolation rule 5 (`docs/architecture.md:111-113`) says nothing
  survives into the next request. That rule is worth more than the latency.
- **Top-k logprob readouts with an invented floor for missing labels**
  (`open-alternative-jev/so1/backends/vllm.py:54-64`;
  `openjev/openjev/engine.py:392` substitutes a floor with no signal to the
  caller). Riderless owns the child and reads exact logits at validated label
  token ids, and already instruments coverage through `allowed_label_mass` and
  `full_vocabulary_argmax`. Adopting either would be a regression.
- **Replacing the label alphabet with a trained head or a trie.**
  `laya/laya/common.py:114-117` scores marker positions through a trained scalar
  scorer; `jevmlx/jevmlx/trie.py:125-183` scores multi-token labels as
  constrained paths through a trie. Both remove the 26-option cap, which is a
  real cost. Both also replace the single clearest invariant riderless has, that
  an answer is one read of the final-position logits at validated single-token
  ids, and the trained-head version introduces weights riderless does not own
  and cannot hash the way it hashes the GGUF.
- **Heuristic script or language gating before inference**
  (`laya/laya/lang.py`). Riderless serves one model and has nothing to route
  between; `allowed_label_mass` is already a better instrument for
  out-of-distribution input, because it is measured from the model rather than
  guessed from Unicode ranges.

## What riderless does that none of them do

- **Provenance that is enforced, not merely recorded.** The install is hash
  verified file by file before the child starts
  (`riderless/api/native_backend.py:102-149`), and every response is re-checked
  against the handshake for the model hash, the runtime hash, the label token
  mapping, question ids and order, and the non-generation invariants
  (`riderless/api/mapping.py:46-84,131-132`). Among the eight, rizzo-flow hashes
  its GGUF at load and jevmlx records hashes in reports, but no candidate refuses
  a response on a provenance mismatch.
- **Isolation proven with a negative control.** 72 comparisons at exactly 0.0
  delta, plus a deliberately broken worker with the context clear removed that
  fails the same harness (`docs/architecture.md:117-120`, `README.md:260-263`).
  Several candidates are isolated by construction; none ships evidence that its
  harness can detect a broken implementation.
- **The non-generation contract checked at the protocol boundary.**
  `riderless/api/mapping.py:51-54` rejects any result with a non-zero generated
  token count, callbacks enabled, or an execution mode other than full. Others
  report a constant zero that nothing cross-checks.
- **Exact, unrounded probabilities from a full-vocabulary read.** kev rounds to
  two decimals before the response leaves the server
  (`kev/kev/api.py:132`), laya rounds to four (`laya/laya/agent.py:342`), and two
  candidates fabricate a floor for labels outside a top-k. Riderless returns the
  softmax as computed and reports how much full-vocabulary mass the option set
  captured.
- **A service failure model.** A documented error table with retryable flags,
  native stderr never copied into an HTTP body, single-flight with 429, no
  partial batch success, and an owned child reaped on any ambiguous transport
  failure followed by 529 rather than a silent restart
  (`README.md:194-214`, `riderless/api/native_backend.py:352-379`).
- **Caller text cannot forge a turn boundary.** A 422 rather than a silent
  mutation (`riderless/api/native/worker.cpp:170-177`). Two candidates silently
  rewrite the caller's text instead, and one of those rewrites a token that is
  structurally load-bearing for its own readout.
- **A budget error instead of silent truncation.** Riderless returns 422
  `budget_error` when a compiled request exceeds the context; laya shrinks the
  per-option budget, clips options and instructions, and truncates the state to
  whatever room is left without telling the caller
  (`laya/laya/common.py:68-86`).
- **Determinism claimed at the right granularity, with the configuration axes
  named and the exception quantified** (`README.md:227-232`).

## Reading these candidates as benchmark arms

Not in scope for this document, and not proposed as work, but worth recording
from the reviews: laya and poorjev run on CPU in well under 2 GiB and would not
touch the shared card at all, which makes them the cheapest calibration
reference points against the seven suites; rizzo-flow is the most runnable
GPU-resident sibling because it fetches its own prebuilt llama.cpp and its 4B
weights are small; kev needs its own virtualenv with an incompatible torch and a
roughly 18 GiB bf16 model; jevmlx's native path cannot run on this hardware at
all; and openjev requires a vLLM built from an unmerged pull request at a
utilisation that would conflict with production serving on this box. Any such
run needs a remote backend in `scripts/riderless/run_usecase_suites.py`, which
today takes worker, manifest, and model path only, and every arm would have to
record what it dropped, because the candidates differ in context budget, option
budget, probability rounding, and the definition of confidence.
