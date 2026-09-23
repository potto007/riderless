# Open decision models on the same suites

Three of the open projects in
[../research/open-model-survey.md](../research/open-model-survey.md) were run
against unridden's own seven use-case suites on 2026-09-21. Two ran as
shipped. The third, `so1`, could not load unridden's model through its own
backends, so it was given a llama.cpp backend and run on the exact weights,
engine and kernels unridden uses, which isolates its prompt and readout design
from the model. This page reports what they scored, on what hardware, and what
each number does and does not mean.

Nothing about any hosted service is measured, quoted, or implied here. Where a
candidate's own documentation asserts something that was used as an input to
this work, it is labelled as that project's claim.

## What was compared

| Row | System | Weights | Where it ran |
| --- | --- | --- | --- |
| unridden | this repository, sequential worker, llama.cpp `v0.4.1` | Gemma 4 26B-A4B UD-Q4_K_XL, model sha256 `ef728c8e...36e30c` | RTX 5090, one owned llama.cpp child |
| Kev-9B | `jaredpalmer/kev`, its own `python -m kev.serve` | `jaredpalmer/kev-9b` @ `2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c` (rank-16 LoRA plus pointer head) on `Qwen/Qwen3.5-9B-Base` @ `68c46c4b3498877f3ef123c856ecfde50c39f404` | RTX 5090, HTTP on localhost, `KEV_DTYPE=bf16 KEV_MERGE=0` |
| Laya | `convaiinnovations/laya`, subfolder `typed-decisions`, library call | HF revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, `model.safetensors` sha256 `4fa56de7...7a24e`, 842 MB ModernBERT-large encoder | CPU only (torch 2.9.1+cpu), GPU never touched |
| so1 separate, so1 packed | `ikermoel/open-alternative-jev` prompt builder and decider, with a llama.cpp backend written for this page (`unridden/api/native/so1_probe.cpp`) | the same GGUF as the unridden row, same sha256, same llama.cpp `v0.4.1` runtime | RTX 5090, one owned helper process, in-process calls |

All three competitors are Apache-2.0 in their code. Weight licences were not
verified and must be checked on the model cards before anything here is reused
commercially.

### Method

- **Same suites, same bytes.** All seven suite files were hashed before the
  first request of every run and the seven sha256 values are identical across
  the three runs and the reference run
  (`outputs/*/run.json`, `suites[].sha256`).
- **Same scorer.** Each driver imports `load_suites`, `score_answer`,
  `score_ranking`, `build_summary` and `write_json` from
  `scripts/unridden/run_usecase_suites.py` rather than reimplementing them.
  Choice is exact match on the argmax option, Noul is thresholded at 0.5, Score
  grades the most probable level with the suite's own tolerance, and rankings
  are scored by top-1 hit and pairwise concordance.
- **No tuning for anyone.** Each system was given the suite requests through
  its own documented entry point with its own defaults. No prompt, threshold,
  or temperature was fitted for any of the three.
- **One confidence definition.** The scorer needs a single confidence field, so
  the answers handed to it carry unridden's `max_probability_v1` recomputed
  from each system's own distribution. Each competitor's native confidence
  statistic is recorded separately and is reported separately below.
- **Reference run.** `outputs/usecase-suites-v041` (unridden sequential,
  llama.cpp `v0.4.1`), the run already documented in
  [llama-v0.4.1-revalidation.md](llama-v0.4.1-revalidation.md). It was replayed
  from its stored observations rather than re-executed on the day, so the runs
  are not contemporaneous.

### Recompute, and one mismatch

Every number on this page was recomputed from the `observations.jsonl` rows of
each run, with gold labels and tolerances read from the suite files, by a
script written for this page and not shared with the benchmark drivers. The
unridden and Laya figures reproduced exactly, to every digit reported here.

Kev did not, by exactly two questions. Kev rounds every probability to two
decimals in its API response, and on
`guardrail-policy-gates/injection-direct-user-turn/route` and
`guardrail-policy-gates/legal-tenancy-buried-question/route` the rounded vector
ties at the top. Kev's own returned `choice` field breaks both ties correctly
from its unrounded internals; a recompute from the published distribution with
a first-option-wins tie-break gets both wrong. The tables below use the strict
recompute. Taking Kev's returned answers instead raises its overall accuracy
from 0.8981 to 0.8997, Choice from 0.8650 to 0.8700, the guardrail suite from
0.8417 to 0.8465, easy from 0.9389 to 0.9427 and hard from 0.8601 to 0.8624,
moves the one-sided split from 76/34 to 74/34, and moves its confidence AUROC
from 0.8905 to 0.8877 because two correctness labels flip. Its other
confidence-conditioned figures move by under 0.005 for the same reason, its
Brier scores do not move at all, and no conclusion on this page turns on which
convention is used.

## Accuracy

The trivial baseline answers the most common gold label for that question id,
the definition used in [usecase-suites.md](usecase-suites.md) and the
whitepaper. It is a strong baseline: a weaker one that answers a single most
common label per question type scores 0.4277 overall.

| Suite | Questions | Trivial baseline | unridden | Kev-9B | Laya | so1 separate | so1 packed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate span and date extraction | 194 | 0.753 | 0.918 | 0.851 | 0.469 | **0.928** | 0.851 |
| claim support verification | 96 | 0.677 | 0.958 | 0.917 | 0.573 | 0.969 | **0.979** |
| entity alignment and record matching | 157 | 0.790 | **1.000** | 0.994 | 0.662 | **1.000** | **1.000** |
| guardrail policy gates | 417 | 0.705 | **0.873** | 0.842 | 0.609 | 0.851 | 0.813 |
| hierarchical, broad to fine classification | 151 | 0.417 | **0.921** | 0.894 | 0.722 | 0.894 | **0.921** |
| passage rerank and line search | 149 | 0.738 | **0.987** | 0.960 | 0.611 | 0.980 | 0.980 |
| tool and argument selection | 122 | 0.738 | **0.984** | 0.959 | 0.746 | 0.967 | 0.975 |
| **All** | **1286** | **0.694** | **0.931** | **0.898** | **0.618** | **0.921** | **0.901** |

Table 1. Question-level accuracy. 241 cases, 0 errored cases in every run.

unridden is ahead on all seven suites against Kev and Laya. Kev is behind on
all seven and above the trivial baseline on all seven. Laya is below the
trivial baseline on five of seven suites and overall; it beats that baseline
only on hierarchical classification (0.722 against 0.417) and, by 0.008, on
tool and argument selection. so1 on the same weights is the closest row: its
separate mode is within 1 point overall and ahead on two suites, its packed
mode is 3 points behind overall and ahead on one.

| Cut | unridden | Kev-9B | Laya | so1 separate | so1 packed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Choice, n = 400 | 0.9325 | 0.8650 | 0.5325 | 0.9200 | 0.9150 |
| Noul, n = 782 | 0.9527 | 0.9386 | 0.6918 | 0.9450 | 0.9143 |
| Score exact, n = 104 | 0.7596 | 0.7212 | 0.3942 | 0.7404 | 0.7500 |
| Score within tolerance, n = 104 | 0.9135 | 0.8942 | 0.5288 | 0.8942 | 0.9038 |
| Score mean absolute error | 0.275 | 0.400 | 0.812 | 0.323 | 0.301 |
| Ranking top-1, n = 32 | 1.000 | 0.906 | 0.625 | 1.000 | 1.000 |
| Ranking pairwise agreement | 1.000 | 0.975 | 0.817 | 1.000 | 1.000 |
| Easy, n = 262 | 0.9656 | 0.9389 | 0.6794 | 0.9618 | 0.9122 |
| Medium, n = 588 | 0.9286 | 0.9082 | 0.6327 | 0.9184 | 0.9133 |
| Hard, n = 436 | 0.9128 | 0.8601 | 0.5619 | 0.8991 | 0.8784 |

Table 2. By question type and by authored difficulty.

| Split over the 1,286 shared questions | Kev-9B | Laya | so1 separate | so1 packed |
| --- | ---: | ---: | ---: | ---: |
| Both correct | 1121 | 759 | 1167 | 1100 |
| Both wrong | 55 | 53 | 72 | 30 |
| Only unridden correct | 76 | 438 | 30 | 97 |
| Only the competitor correct | 34 | 36 | 17 | 59 |

Table 3. Head to head. Kev's 34 wins are 19 Noul, 9 Choice and 6 Score, and 21
of them are in the guardrail suite. Laya's 36 are 22 Noul, 8 Choice and 6
Score, 24 of them in the guardrail suite. so1 separate's 17 wins are 6 in
candidate span and 6 in the guardrail suite; unridden's 30 over it are 15 in
the guardrail suite and 6 in hierarchical classification.

## Calibration, latency, memory

| Measure | unridden | Kev-9B | Laya | so1 separate | so1 packed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Confidence AUROC for correctness | 0.9226 | 0.8905 | 0.6886 | **0.9412** | 0.8872 |
| Noul Brier | 0.0475 | **0.0444** | 0.2068 | 0.0521 | 0.0758 |
| Choice multiclass Brier | **0.1215** | 0.2098 | 0.6241 | 0.1523 | 0.1433 |
| Score multiclass Brier | 0.4681 | **0.3621** | 0.6502 | 0.4949 | 0.4566 |
| All questions, multiclass Brier | **0.1335** | 0.1485 | 0.4982 | 0.1507 | 0.1737 |
| ECE, 15 bins | 0.0655 | **0.0517** | 0.0640 | 0.0737 | 0.0776 |
| Mean confidence | 0.9920 | 0.8531 | 0.5737 | 0.9932 | 0.9789 |
| Mean confidence when right / wrong | 0.995 / 0.958 | 0.882 / 0.596 | 0.608 / 0.518 | 0.997 / 0.949 | 0.988 / 0.897 |
| Request latency, median | 179.8 ms | 153.5 ms | 1503.7 ms | 187.9 ms | **107.7 ms** |
| Request latency, p95 | 508.9 ms | 399.4 ms | 4204.9 ms | 533.6 ms | **290.5 ms** |
| Input tokens, all 241 cases | 246,981 | 140,044 | not comparable | 310,014 | 197,008 |
| Peak GPU memory above pre-load baseline | 17,636 MiB | 19,498 MiB | none | 18,056 MiB | 18,056 MiB |

Table 4. All confidence figures use unridden's max-label-probability formula
recomputed from each system's distribution, so the AUROC, Brier and ECE columns
are like for like. Bold marks the best of the five on that row. The so1
multiclass Brier figures for all questions were computed by the so1 driver from
its own rows; Kev's input-token count uses a different tokenizer and prompt
form and is not an efficiency claim.

**What each latency measures, because they are not the same boundary.**
unridden's 179.8 ms is in-process wall clock around one API request, which
answers a case's questions one at a time against an owned llama.cpp child on
the GPU with prefix reuse. Kev's 153.5 ms is wall clock around an HTTP POST
from the driver to `kev.serve` on localhost on the same GPU; Kev's own
server-reported model time for the same requests was 150.4 ms median and
394.6 ms p95, so HTTP and FastAPI added about 3 ms. Laya's 1503.7 ms is
in-process wall clock around one `laya.Router.predict` per case, which answers
all of a case's questions in a single forward pass, on CPU with a CPU-only
torch build. The Laya column is not a speed comparison at all. Kev's column
also ran without `flash-linear-attention` and `causal_conv1d` installed, so
transformers fell back to reference PyTorch for the Gated DeltaNet and
causal-conv paths; by its own log message that is numerically correct and
slower, which makes Kev's latency pessimistic and leaves its accuracy
unaffected. The two so1 columns are the same boundary and the same GPU as
unridden (in-process wall clock around one call per case to an owned llama.cpp
helper), but not the same work: unridden answers a case one question at a
time and reuses the state prefix across them, so1 separate re-prefills the full
state for every question, and so1 packed writes the state once and reads every
question from one sequence. The input-token row is the denominator behind
those three numbers.

The memory row is the peak above each run's own pre-load baseline: 23,018 MiB
peak against a 5,382 MiB baseline for unridden (a 26B MoE at Q4), 24,416 MiB
peak against a 4,918 MiB baseline for Kev (a 9B backbone in bf16 with an
unmerged adapter), and 22,245 MiB against a 4,189 MiB baseline for the so1
helper, which holds the same model as unridden with a 4,096-token context
instead of 2,048. Laya held no GPU memory; `nvidia-smi` moved only within the
noise of the resident embedding server.

Native confidence statistics, for completeness. Kev reports a rescaled
statistic rather than a maximum probability, and under that statistic its AUROC
for correctness is 0.8006 over the 504 Choice and Score questions that carry
one. Laya reports a normalised-entropy confidence, which is worse than the
recomputed maximum probability on both axes (ECE 0.1686, AUROC 0.6635). Neither
is interchangeable with the numbers in Table 4.

## Kev-9B: where it wins and loses

Kev is the closest independent model here. It trails unridden on every suite and
every question type, by 3.3 points overall, but it does so with a 9B trained
decision model against a 26B general model, and its losses are concentrated
where it is architecturally different: Choice (0.8650 against 0.9325) and
ranking (top-1 0.906 against 1.000). It is closest on Noul, where it wins the
Brier score outright (0.0444 against 0.0475) and where 19 of its 34 wins live.

Its real advantage is the shape of its errors. unridden holds 59 of its 89
wrong answers above 0.99 confidence; Kev holds none of its 131 wrong answers
that high, and its mean confidence on a wrong answer is 0.596 against
unridden's 0.958. In the guardrail suite, only 1 of Kev's 66 misses is held
above 0.9 confidence and its most confident miss there
(`stacktrace-customer-email/ctx_quoted_frame`) sits at 0.93, while unridden
misses `badge-cloning-actionable-request/hz_pii` and
`novelist-threat-dialogue/severity` above 0.9999. unridden still wins the
AUROC, because ranking power and absolute calibration are different things, but
a caller who wants to abstain gets more signal from Kev's number than from
unridden's. That is the single most useful finding on this page.

Two practical costs. First, Kev could not be run on its own default serving
options. Its loader builds the backbone in fp32 whenever merging is on
(`kev/checkpoint.py:128-129`) and moves it to the device before the merge
(`kev/model.py:177`), which puts a 9B backbone at roughly 36 GiB and over the
card. `KEV_MERGE=0` is its own documented knob (`README` line 238), but its own
source says the merged path is closer to the fp32 numbers than the unmerged
adapter (max |dp| 0.017 against 0.029, 0 against 1 argmax flips, on kev-4b over
24 dev records, `kev/checkpoint.py:80-82`), so these figures come from a
marginally different path than its published evaluations. Second, it took
19,498 MiB of GPU memory to serve 9B against unridden's 17,636 MiB for
26B-A4B, and it rounds every probability to two decimals (`kev/api.py:133`),
which quantises its Brier, its reliability table, and any confidence threshold
a caller might set.

One fairness check that came out in Kev's favour: measured with Kev's own
tokenizer, the suite states have a median of 98 tokens and a maximum of 511,
so only 4 of 241 cases exceed its 384-token trained state budget and none
exceeds its 1,024-token state-plus-question budget. These suites sit inside the
population it was trained for, and its state-prefix cache was inactive
throughout because every case has a distinct state.

Where Kev is architecturally ahead: its pointer-head readout has no
single-token label alphabet, so its option ceiling is 255 (`kev/api.py:14`)
rather than unridden's 26. On these suites that never bound, because the
largest Choice question has 17 options.

## Laya: where it wins and loses

Laya is an 842 MB encoder that answers all of a case's questions in one forward
pass and needs no GPU at all. That is the win, and on this evidence it is the
only one. At 0.618 overall it is below the 0.694 trivial baseline, and the
controls say the gap is not an artefact of the harness.

- **Context is not the explanation.** At its own `max_len` of 1024 not one of
  the 1,286 questions had its state truncated, by its own budget accounting.
  The matched-budget control
  that Laya's own benchmark script runs (`max_len` capped to 512) truncated 10
  questions by 269 tokens in total, changed 3 answers, and moved accuracy from
  0.6182 to 0.6174.
- **Option budget is a partial explanation at most.** No option exceeded its
  48-token cap and no question exceeded 20 options, but the 256-token head
  budget clipped 180 option strings across 36 questions and clipped the
  instructions on those same 36, so those two effects are confounded in that
  subset. For scale, 32 of the 400 Choice questions carry 11 or more options
  and the largest carries 17.
- **Checkpoint choice is not the explanation.** The base English checkpoint
  scores 0.600 overall, slightly worse.
- **It is out of distribution.** The `typed-decisions` checkpoint is fine tuned
  on four workflows whose question-id signatures are hard-coded in its router,
  and none of the seven suites matches one. Both Laya numbers here are
  zero-shot for it.
- **It says so itself.** Both checkpoints ship a `choice:11+` temperature of
  0.1006, which Laya refuses and clamps to 0.5 while warning that the affected
  buckets are uncalibrated. Those 32 Choice questions fall in that bucket. The
  warning is recorded verbatim in the run manifest rather than suppressed.

Its calibration is the mirror image of unridden's. The two have almost the
same ECE (0.0640 against 0.0655) for opposite reasons: Laya is underconfident
by about 0.04 and unridden is overconfident by about 0.06. But Laya's
confidence has much weaker ranking power (AUROC 0.689), so the low ECE does not
buy a usable abstention threshold: taking its most confident 10% of answers
still gets only 0.806 of them right, against 1.000 for unridden.

## so1 on unridden's own engine: the readout experiment

`ikermoel/open-alternative-jev` (import name `so1`) was chosen because it is
the one candidate that can be pointed at unridden's exact model, which turns
the comparison into a test of prompt and readout design alone. Its shipped
backends could not do that on this machine: its Transformers path quantizes
only `nn.Linear` modules through bitsandbytes (`so1/backends/hf.py:25-27`,
`transformers/integrations/bitsandbytes.py:189`), and Gemma 4 26B-A4B keeps
91.5% of its parameters in fused 3-D expert tensors
(`modeling_gemma4.py:1287-1288`), which leaves about 45 GiB resident at 4-bit
against a 32.6 GiB card. Its vLLM path would have changed quantization and
kernels at the same time as the readout.

So `so1` was given a third backend, written for this page: a 300-line
llama.cpp helper (`unridden/api/native/so1_probe.cpp`) that loads the same
GGUF with the same model and context parameters as `worker.cpp`'s sequential
configuration, except `n_ctx` 4,096 to fit so1's longest packed sequence
(2,138 tokens), and returns the label logits at whatever positions so1's
decider asks for. so1's own `PromptBuilder` and `Decider` run unchanged. Before
the benchmark, every prompt sent (1,286 separate, 241 packed) was decoded and
re-tokenized by `llama_tokenize`, with 0 id mismatches, the 26 letters resolve
to the same token ids the reference run recorded, and 200 of unridden's own
compiled prompts re-rendered through the HF chat template matched the prompt
token counts in the reference observations exactly. The HF and GGUF templates
are the same template.

Two things were transcribed rather than taken as shipped. so1's default ChatML
`ChatFormat` raises on Gemma 4 (`could not find the user turn start`), so a
Gemma format was supplied (`user_turn_start="<|turn>user"`,
`separator="_<turn|>\n"`), verified against the tokenizer's own three-turn
rendering. And unridden's content is carried over verbatim (serialized state,
instruction text, per-option rubric text, option order) but framed by so1's
prompt: so1's own instruction lines replace unridden's system instruction,
options render as `A. id - rubric` rather than `A: id - rubric`, there is no
`Answer:` prefix, and the readout sits at the end of the generation prompt.

**Separate mode** is so1's no-interference baseline: one sequence per question,
the full state in each. It is the fairest single number on this page for "does
the prompt design matter", and the answer is: by about a point. 0.921 against
0.931 overall, 30 questions only unridden gets right against 17 only so1 gets
right, and the two disagree on 53 answers in total. That is inside the
four-point resolution these suites can carry. so1 separate is ahead on
candidate span extraction and claim support, behind on the guardrail suite by
2.2 points (15 of unridden's 30 wins), and its confidence ranks correctness
slightly better (AUROC 0.941 against 0.923) while being marginally less
calibrated (ECE 0.074 against 0.066). It costs 26% more input tokens and 4% more
latency than unridden, because it re-prefills the state for every question
where unridden reuses it.

**Packed mode** is so1's design contribution: every question of a case in one
sequence, state written once, one readout position per question, later
questions able to attend to earlier questions and to the fixed placeholder
turns between them. On these suites it is 1.74 times faster than separate mode
(107.7 against 187.9 ms median per case), uses 36% fewer tokens, and loses
2 points of accuracy for it. Against separate mode, 158 of 1,286 answers move
(12.3%), 177 questions move probability by more than 0.3, 87 are right only
when separate and 62 only when packed. Noul suffers most: accuracy 0.945 to
0.914, Brier 0.052 to 0.076, and mean p(true) on true targets 0.962 to 0.825.
so1's own documentation puts its interference at 6 to 9% of individual answers
with accuracy unchanged on its benchmarks; on these suites, which were not
written for it, it is larger and does cost accuracy. unridden's opt-in batched
mode ([batched-mode.md](batched-mode.md)) makes the same trade with a different
mechanism, isolating each question in its own sequence over a shared prefix,
and lost 5 to 6 answers rather than 25 for a smaller speed gain.

One artefact is inherent to so1's packing and could not be separated from it:
because so1 reuses the generation prompt for every turn, each completed
placeholder turn in the packed sequence reads
`<|turn>model\n<|channel>thought\n<channel|>_<turn|>` where a real Gemma
transcript would carry an answer. Part of what the packed row measures is the
model's reaction to that.

The remaining five projects in the survey were not benchmarked.

## Threats to validity

- **One run each.** No repeats, single seed, no confidence intervals and no
  significance testing. The 3.3-point unridden-to-Kev gap and the 76-against-34
  one-sided split are point estimates. The suites' own caveat applies to every
  row here: 241 cases cannot resolve overall differences under roughly four
  points or per-suite differences under five, which puts the unridden-to-Kev
  gap at the edge of what this evidence can carry. The Laya gap is far outside
  it.
- **unridden's authors wrote the suites, the gold labels, the scorer and this
  page.** The suites were written blind and the labels were hashed before the
  first inference, but they were written to exercise this API. Two competitors
  were graded by the incumbent's own harness.
- **The suites are written for unridden's envelope.** 26 single-token option
  labels, 2,048 prompt tokens, one question per prompt, and no question that
  depends on a sibling. Where that envelope forced an authored workaround, such
  as splitting a day of month into a tens band and a final digit because 31
  options exceed the label ceiling, the suites keep the workaround and every
  system is scored on it. That handicap is unridden's own and is left
  uncorrected, but it also means neither competitor was measured on the option
  counts its architecture could have handled.
- **Nothing was tuned for anyone**, which cuts both ways. Each system ran its
  own defaults on requests authored in unridden's shapes. A competitor with
  prompts written for it would likely score higher, and so would unridden with
  rubrics written for the guardrail suite it struggles with.
- **The hardware is not matched.** Laya ran on CPU by choice, so its latency
  column measures a different machine as well as a different boundary. Its
  accuracy and calibration are unaffected.
- **Kev ran on a non-default serving option** (`KEV_MERGE=0`), which its own
  documentation describes as slightly less exact than the merged path, and its
  two-decimal rounding quantises every probability-derived number in Table 4.
- **so1 ran on a backend and a chat format it does not ship.** The llama.cpp
  helper and the Gemma `ChatFormat` were written for this page. The tokenizer
  and template equivalence checks above are the evidence that the prompts so1
  built are the prompts the model saw; the prompt content is unridden's,
  folded into so1's frame, so the so1 rows measure so1's design on unridden's
  material rather than so1 as its author would run it.
- **The reference run is a replay.** It was recorded earlier the same day, not
  re-executed alongside the competitors. The suite hashes are identical, so the
  inputs and labels are the same bytes, but the two were not run back to back.
- **Confidence definitions were unified.** Each competitor's native statistic
  is a different quantity and is reported separately rather than mixed into
  Table 4.
- **Everything here is synthetic.** These are hand-authored cases with frozen
  gold labels, not an established public benchmark, and one suite is saturated
  for two of the three systems.

## Evidence

`outputs/` is not committed. The three benchmark drivers, the so1 helper and
the outcome-diff tool are committed with this page. Every figure here comes
from these run directories and the drivers that produced them.

| Item | Path |
| --- | --- |
| unridden reference run | `outputs/usecase-suites-v041/` |
| Kev run, plus GPU samples and rounding report | `outputs/competitor-kev-v1/` |
| Laya run | `outputs/competitor-laya-v1/` |
| Laya matched-context control (`max_len` 512) | `outputs/competitor-laya-v1-matched512/` |
| Laya base English checkpoint control | `outputs/competitor-laya-v1-english/` |
| so1 separate run, with tokenizer checks in `run.json` | `outputs/competitor-so1-v1-separate/` |
| so1 packed run, plus its diff against separate | `outputs/competitor-so1-v1-packed/` |
| Kev driver | `scripts/unridden/bench_competitor_kev.py` |
| Laya driver | `scripts/unridden/bench_competitor_laya.py` |
| so1 driver and llama.cpp backend | `scripts/unridden/bench_competitor_so1.py` |
| so1 logits helper | `unridden/api/native/so1_probe.cpp` |
| Head-to-head outcome diff | `scripts/unridden/compare_competitor_outcomes.py` |

Each run directory carries a `run.json` with the suite hashes, the weight
provenance, and the serving options actually used, and an `observations.jsonl`
with one row per case holding the request, the response, and the scored
outcomes.
