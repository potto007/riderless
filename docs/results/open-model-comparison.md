# Open decision models on the same suites

Two of the open projects in
[../research/open-model-survey.md](../research/open-model-survey.md) were run
against riderless's own seven use-case suites on 2026-09-21, and a third was
attempted and blocked. This page reports what they scored, on what hardware,
and what each number does and does not mean.

Nothing about any hosted service is measured, quoted, or implied here. Where a
candidate's own documentation asserts something that was used as an input to
this work, it is labelled as that project's claim.

## What was compared

| Row | System | Weights | Where it ran |
| --- | --- | --- | --- |
| riderless | this repository, sequential worker, llama.cpp `v0.4.1` | Gemma 4 26B-A4B UD-Q4_K_XL, model sha256 `ef728c8e...36e30c` | RTX 5090, one owned llama.cpp child |
| Kev-9B | `jaredpalmer/kev`, its own `python -m kev.serve` | `jaredpalmer/kev-9b` @ `2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c` (rank-16 LoRA plus pointer head) on `Qwen/Qwen3.5-9B-Base` @ `68c46c4b3498877f3ef123c856ecfde50c39f404` | RTX 5090, HTTP on localhost, `KEV_DTYPE=bf16 KEV_MERGE=0` |
| Laya | `convaiinnovations/laya`, subfolder `typed-decisions`, library call | HF revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, `model.safetensors` sha256 `4fa56de7...7a24e`, 842 MB ModernBERT-large encoder | CPU only (torch 2.9.1+cpu), GPU never touched |

Both competitors are Apache-2.0 in their code. Weight licences were not
verified and must be checked on the model cards before anything here is reused
commercially.

### Method

- **Same suites, same bytes.** All seven suite files were hashed before the
  first request of every run and the seven sha256 values are identical across
  the three runs and the reference run
  (`outputs/*/run.json`, `suites[].sha256`).
- **Same scorer.** Each driver imports `load_suites`, `score_answer`,
  `score_ranking`, `build_summary` and `write_json` from
  `scripts/riderless/run_usecase_suites.py` rather than reimplementing them.
  Choice is exact match on the argmax option, Noul is thresholded at 0.5, Score
  grades the most probable level with the suite's own tolerance, and rankings
  are scored by top-1 hit and pairwise concordance.
- **No tuning for anyone.** Each system was given the suite requests through
  its own documented entry point with its own defaults. No prompt, threshold,
  or temperature was fitted for any of the three.
- **One confidence definition.** The scorer needs a single confidence field, so
  the answers handed to it carry riderless's `max_probability_v1` recomputed
  from each system's own distribution. Each competitor's native confidence
  statistic is recorded separately and is reported separately below.
- **Reference run.** `outputs/usecase-suites-v041` (riderless sequential,
  llama.cpp `v0.4.1`), the run already documented in
  [llama-v0.4.1-revalidation.md](llama-v0.4.1-revalidation.md). It was replayed
  from its stored observations rather than re-executed on the day, so the runs
  are not contemporaneous.

### Recompute, and one mismatch

Every number on this page was recomputed from the `observations.jsonl` rows of
each run, with gold labels and tolerances read from the suite files, by a
script written for this page and not shared with the benchmark drivers. The
riderless and Laya figures reproduced exactly, to every digit reported here.

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

| Suite | Questions | Trivial baseline | riderless | Kev-9B | Laya |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate span and date extraction | 194 | 0.753 | **0.918** | 0.851 | 0.469 |
| claim support verification | 96 | 0.677 | **0.958** | 0.917 | 0.573 |
| entity alignment and record matching | 157 | 0.790 | **1.000** | 0.994 | 0.662 |
| guardrail policy gates | 417 | 0.705 | **0.873** | 0.842 | 0.609 |
| hierarchical, broad to fine classification | 151 | 0.417 | **0.921** | 0.894 | 0.722 |
| passage rerank and line search | 149 | 0.738 | **0.987** | 0.960 | 0.611 |
| tool and argument selection | 122 | 0.738 | **0.984** | 0.959 | 0.746 |
| **All** | **1286** | **0.694** | **0.931** | **0.898** | **0.618** |

Table 1. Question-level accuracy. 241 cases, 0 errored cases in every run.

riderless is ahead on all seven suites against both competitors. Kev is behind
on all seven and above the trivial baseline on all seven. Laya is below the
trivial baseline on five of seven suites and overall; it beats that baseline
only on hierarchical classification (0.722 against 0.417) and, by 0.008, on
tool and argument selection.

| Cut | riderless | Kev-9B | Laya |
| --- | ---: | ---: | ---: |
| Choice, n = 400 | 0.9325 | 0.8650 | 0.5325 |
| Noul, n = 782 | 0.9527 | 0.9386 | 0.6918 |
| Score exact, n = 104 | 0.7596 | 0.7212 | 0.3942 |
| Score within tolerance, n = 104 | 0.9135 | 0.8942 | 0.5288 |
| Score mean absolute error | 0.275 | 0.400 | 0.812 |
| Ranking top-1, n = 32 | 1.000 | 0.906 | 0.625 |
| Ranking pairwise agreement | 1.000 | 0.975 | 0.817 |
| Easy, n = 262 | 0.9656 | 0.9389 | 0.6794 |
| Medium, n = 588 | 0.9286 | 0.9082 | 0.6327 |
| Hard, n = 436 | 0.9128 | 0.8601 | 0.5619 |

Table 2. By question type and by authored difficulty.

| Split over the 1,286 shared questions | Kev-9B | Laya |
| --- | ---: | ---: |
| Both correct | 1121 | 759 |
| Both wrong | 55 | 53 |
| Only riderless correct | 76 | 438 |
| Only the competitor correct | 34 | 36 |

Table 3. Head to head. Kev's 34 wins are 19 Noul, 9 Choice and 6 Score, and 21
of them are in the guardrail suite. Laya's 36 are 22 Noul, 8 Choice and 6
Score, 24 of them in the guardrail suite.

## Calibration, latency, memory

| Measure | riderless | Kev-9B | Laya |
| --- | ---: | ---: | ---: |
| Confidence AUROC for correctness | 0.9226 | 0.8905 | 0.6886 |
| Noul Brier | 0.0475 | **0.0444** | 0.2068 |
| Choice multiclass Brier | **0.1215** | 0.2098 | 0.6241 |
| Score multiclass Brier | 0.4681 | **0.3621** | 0.6502 |
| All questions, multiclass Brier | **0.1335** | 0.1485 | 0.4982 |
| ECE, 15 bins | 0.0655 | **0.0517** | 0.0640 |
| Mean confidence | 0.9920 | 0.8531 | 0.5737 |
| Mean confidence when right / wrong | 0.995 / 0.958 | 0.882 / 0.596 | 0.608 / 0.518 |
| Request latency, median | 179.8 ms | 153.5 ms | 1503.7 ms |
| Request latency, p95 | 508.9 ms | 399.4 ms | 4204.9 ms |
| Peak GPU memory above pre-load baseline | 17,636 MiB | 19,498 MiB | none |

Table 4. All confidence figures use riderless's max-label-probability formula
recomputed from each system's distribution, so the AUROC, Brier and ECE columns
are like for like. Bold marks the best of the three on that row.

**What each latency measures, because they are not the same boundary.**
riderless's 179.8 ms is in-process wall clock around one API request, which
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
unaffected.

The memory row is the peak above each run's own pre-load baseline: 23,018 MiB
peak against a 5,382 MiB baseline for riderless (a 26B MoE at Q4), and
24,416 MiB peak against a 4,918 MiB baseline for Kev (a 9B backbone in bf16
with an unmerged adapter). Laya held no GPU memory; `nvidia-smi` moved only
within the noise of the resident embedding server.

Native confidence statistics, for completeness. Kev reports a rescaled
statistic rather than a maximum probability, and under that statistic its AUROC
for correctness is 0.8006 over the 504 Choice and Score questions that carry
one. Laya reports a normalised-entropy confidence, which is worse than the
recomputed maximum probability on both axes (ECE 0.1686, AUROC 0.6635). Neither
is interchangeable with the numbers in Table 4.

## Kev-9B: where it wins and loses

Kev is the closest thing here to a peer. It trails riderless on every suite and
every question type, by 3.3 points overall, but it does so with a 9B trained
decision model against a 26B general model, and its losses are concentrated
where it is architecturally different: Choice (0.8650 against 0.9325) and
ranking (top-1 0.906 against 1.000). It is closest on Noul, where it wins the
Brier score outright (0.0444 against 0.0475) and where 19 of its 34 wins live.

Its real advantage is the shape of its errors. riderless holds 59 of its 89
wrong answers above 0.99 confidence; Kev holds none of its 131 wrong answers
that high, and its mean confidence on a wrong answer is 0.596 against
riderless's 0.958. In the guardrail suite, only 1 of Kev's 66 misses is held
above 0.9 confidence and its most confident miss there
(`stacktrace-customer-email/ctx_quoted_frame`) sits at 0.93, while riderless
misses `badge-cloning-actionable-request/hz_pii` and
`novelist-threat-dialogue/severity` above 0.9999. riderless still wins the
AUROC, because ranking power and absolute calibration are different things, but
a caller who wants to abstain gets more signal from Kev's number than from
riderless's. That is the single most useful finding on this page.

Two practical costs. First, Kev could not be run on its own default serving
options. Its loader builds the backbone in fp32 whenever merging is on
(`kev/checkpoint.py:128-129`) and moves it to the device before the merge
(`kev/model.py:177`), which puts a 9B backbone at roughly 36 GiB and over the
card. `KEV_MERGE=0` is its own documented knob (`README` line 238), but its own
source says the merged path is closer to the fp32 numbers than the unmerged
adapter (max |dp| 0.017 against 0.029, 0 against 1 argmax flips, on kev-4b over
24 dev records, `kev/checkpoint.py:80-82`), so these figures come from a
marginally different path than its published evaluations. Second, it took
19,498 MiB of GPU memory to serve 9B against riderless's 17,636 MiB for
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
rather than riderless's 26. On these suites that never bound, because the
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

Its calibration is the mirror image of riderless's. The two have almost the
same ECE (0.0640 against 0.0655) for opposite reasons: Laya is underconfident
by about 0.04 and riderless is overconfident by about 0.06. But Laya's
confidence has much weaker ranking power (AUROC 0.689), so the low ECE does not
buy a usable abstention threshold: taking its most confident 10% of answers
still gets only 0.806 of them right, against 1.000 for riderless.

## What could not be run, and why

The survey lists eight projects. Three were dispatched for measurement. The
third, `ikermoel/open-alternative-jev` (import name `so1`), was chosen because
it is the only candidate that can be pointed at riderless's exact model, which
would have isolated the readout design from the model. It is blocked, and the
blocker is structural rather than a matter of finding a better checkpoint. No
accuracy, latency or memory figure exists for it, and none should be inferred.

- `so1`'s Transformers backend offers exactly `load_in_8bit` and `load_in_4bit`,
  which build a `BitsAndBytesConfig` (`so1/backends/hf.py:25-27`). The
  transformers integration those flags drive replaces a module only when it is
  an `nn.Linear` or a `Conv1D` (`transformers/integrations/bitsandbytes.py:189`
  in transformers 5.17.0).
- Gemma 4 26B-A4B stores its experts as fused 3-D parameters, not per-expert
  Linear layers (`transformers/models/gemma4/modeling_gemma4.py:1287-1288`), so
  91.5% of its parameters are outside that gate. Projected resident weights are
  44.99 GiB at 4-bit, 46.01 GiB at 8-bit, and 48.07 GiB at bf16, against a
  32.6 GiB card. These are projections from a meta-device instantiation plus
  the module-type gate, not from an attempted load, and they count weights only.
- The 15 to 16 GiB pre-quantized Gemma 4 checkpoints that do quantize the
  experts achieve it by unfusing them into per-expert Linear modules. A probe
  over one such checkpoint's safetensors header counted 47,648 keys, 46,080 of
  them under `.experts.`, against 60 expert keys in the model built from that
  same repo's config, and Gemma 4 ships no weight converter to bridge them.
- `so1`'s packed mode raises `ValueError: could not find the user turn start in
  the chat prompt` on the Gemma 4 chat template (`so1/prompting.py:104`),
  verified against the real tokenizer with no weights loaded. Its separate mode
  builds correctly and places the readout at the first answer token.
- The one remaining path, `so1`'s in-process vLLM backend against an NVFP4
  checkpoint, was deliberately not taken. It would change the quantization and
  the kernel stack at the same time as the readout, which makes any accuracy
  difference unattributable, and an unprebuilt FlashInfer FP4 kernel shape JIT
  compiles without a job cap on this machine.

The remaining five projects in the survey were not benchmarked.

## Threats to validity

- **One run each.** No repeats, single seed, no confidence intervals and no
  significance testing. The 3.3-point riderless-to-Kev gap and the 76-against-34
  one-sided split are point estimates. The suites' own caveat applies to every
  row here: 241 cases cannot resolve overall differences under roughly four
  points or per-suite differences under five, which puts the riderless-to-Kev
  gap at the edge of what this evidence can carry. The Laya gap is far outside
  it.
- **riderless's authors wrote the suites, the gold labels, the scorer and this
  page.** The suites were written blind and the labels were hashed before the
  first inference, but they were written to exercise this API. Two competitors
  were graded by the incumbent's own harness.
- **The suites are written for riderless's envelope.** 26 single-token option
  labels, 2,048 prompt tokens, one question per prompt, and no question that
  depends on a sibling. Where that envelope forced an authored workaround, such
  as splitting a day of month into a tens band and a final digit because 31
  options exceed the label ceiling, the suites keep the workaround and every
  system is scored on it. That handicap is riderless's own and is left
  uncorrected, but it also means neither competitor was measured on the option
  counts its architecture could have handled.
- **Nothing was tuned for anyone**, which cuts both ways. Each system ran its
  own defaults on requests authored in riderless's shapes. A competitor with
  prompts written for it would likely score higher, and so would riderless with
  rubrics written for the guardrail suite it struggles with.
- **The hardware is not matched.** Laya ran on CPU by choice, so its latency
  column measures a different machine as well as a different boundary. Its
  accuracy and calibration are unaffected.
- **Kev ran on a non-default serving option** (`KEV_MERGE=0`), which its own
  documentation describes as slightly less exact than the merged path, and its
  two-decimal rounding quantises every probability-derived number in Table 4.
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

`outputs/` is not committed, and the two benchmark drivers were written for
these runs and are not committed with this page. Every figure here comes from
these run directories and the drivers that produced them.

| Item | Path |
| --- | --- |
| riderless reference run | `outputs/usecase-suites-v041/` |
| Kev run, plus GPU samples and rounding report | `outputs/competitor-kev-v1/` |
| Laya run | `outputs/competitor-laya-v1/` |
| Laya matched-context control (`max_len` 512) | `outputs/competitor-laya-v1-matched512/` |
| Laya base English checkpoint control | `outputs/competitor-laya-v1-english/` |
| Kev driver | `scripts/riderless/bench_competitor_kev.py` |
| Laya driver | `scripts/riderless/bench_competitor_laya.py` |
| Head-to-head outcome diff | `scripts/riderless/compare_competitor_outcomes.py` |

Each run directory carries a `run.json` with the suite hashes, the weight
provenance, and the serving options actually used, and an `observations.jsonl`
with one row per case holding the request, the response, and the scored
outcomes.
