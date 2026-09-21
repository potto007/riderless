# Use-case suite results

Run on 2026-09-21, one GPU, Gemma 4 26B-A4B UD-Q4_K_XL. Seven suites cover the
families of workload that the Jev interface is documented for. Conformance,
isolation, and failure-path evidence for the API itself is in
[api-validation.md](api-validation.md); the suites themselves are described in
[../usecase-suites.md](../usecase-suites.md).

## Method

- Families were derived from a survey of the documented patterns. Flat intent
  classification was excluded because an earlier public-dataset run covers it.
- One designer per family wrote original cases (no benchmark data) with easy,
  medium, and hard tiers. A second, independent agent then answered every
  question blind, compared against gold, corrected clear label errors (8), and
  flagged ambiguous cases as `contested` (6 cases).
- Labels were frozen before inference. The runner records each suite's SHA-256
  in `run.json` before the model loads; an auditor confirmed all seven hashes
  still matched the files on disk.
- One GPU run, first attempt, no runner edits, no suite edits, nothing tuned
  toward results: 241 requests, 0 errored, 0 generated tokens across all 1,286
  question diagnostics.
- Two adversarial auditors rescored everything from the recorded observations
  with their own scripts. Every headline number reproduced exactly. Their
  criticism of what the numbers mean is folded in below.

## Cases that were not turned into suites

| Documented use case | Why not |
| --- | --- |
| Parallel question batching | It measures batching economics. This backend prefills per question, so the comparison does not exist here. |
| Self-consistency over repeats | This backend is deterministic with no sampling; repeats are bit-identical. |
| Extraction cascade end to end | Needs a generative extractor and an escalation model. Its per-field error checks live inside claim support. |
| Automatic feature discovery | Needs an outer generative model and a downstream trained model; no gold labels for proposed features. |
| A first-stage Choice with hundreds of options | Exceeds the 26-label backend capacity. Its shortlist-and-gate stage lives inside tool selection. |
| Structure reconstruction | Caller-side string assembly. Its line-join and block-type judgments live inside span extraction. |
| Fan-out, gated routing, composite scoring, intent routing | Application code over answers, not model capability. |

## Results

Question-level accuracy from the first frozen run. "Trivial baseline" is the
auditors' stricter one: always answer the most common gold label for that exact
question id within the suite.

| Suite | Questions | Accuracy | Trivial baseline | Auditor verdict |
| --- | --- | --- | --- | --- |
| hierarchical-and-broad-to-fine-classification | 151 | 0.921 | 0.417 | Strongest signal. Leaf choice 1.000 against 0.043. Its `depth` Score sits at baseline. |
| candidate-span-and-date-extraction | 194 | 0.923 | 0.753 | Good. Best traps in the run. |
| passage-rerank-and-line-search | 149 | 0.966 | 0.738 | Good on Choice and Noul. Discount the ranking numbers (below). |
| claim-support-verification | 96 | 0.958 | 0.677 | Usable but small. One ambiguous case. |
| tool-and-argument-selection | 122 | 0.984 | 0.738 | Thin but real. Too near ceiling to track regressions. |
| guardrail-policy-gates | 417 | 0.859 | 0.703 | Too noisy to quote as one score (below). |
| entity-alignment-and-record-matching | 157 | 1.000 | 0.790 | No signal. Saturated. Treat as a smoke test. |
| **Overall** | **1286** | **0.925** | **0.693** | Lift over trivial is about +0.23. |

By primitive: Choice 375/400 (0.938); Noul 737/782 (0.943), Brier 0.053, mean
P(true) 0.957 on true targets and 0.065 on false; Score exact level 77/104
(0.740), within tolerance 94/104 (0.904), mean absolute error 0.280. By
difficulty: easy 0.966, medium 0.927, hard 0.897. Contested cases 34/41 (0.829),
others 1155/1245 (0.928). Latency per request: median 235 ms, p95 689 ms;
353,257 input tokens. No answer-position bias was found: accuracy by gold option
position is flat, and the first position is the weakest bucket.

## What these numbers do not support

- **The questions are not 1,286 independent samples.** The guardrail suite is
  32% of the total and is 13 rubrics replayed over 36 states. The 97 failures
  cluster into roughly 60 cases of behaviour. No confidence interval is claimed.
- **Confidence is not a usable abstention signal.** 94.7% of answers exceed
  0.99. Mean confidence is 0.995 when correct and 0.945 when wrong. Only 18 of
  1,286 answers fall below 0.8. `max_probability_v1` must not gate decisions
  without task-specific calibration.
- **Ranking at 32/32 top-1 and 120/120 pairwise is inflated.** 30 of 32 rankings
  declare only a top-1, so pairwise mostly restates top-1. Five wins were decided
  by margins under 1.1e-5, which is the numerical floor. A fair reading is 27 of
  32 decided by a real margin.
- **Score exact accuracy of 0.74 is mostly two rubrics.** 19 of 27 misses are
  the guardrail `severity` rubric and 6 are hierarchical `depth`.
- **About 30% of the "hard" tier is not hard** (battery negatives and near
  verbatim lookups), so 0.897 on hard is diluted.
- **Three gold labels were judged unsound by the auditors.** Two were revised
  (below); one was kept deliberately.

## The isolation trap, seen in the data

Each question is compiled into its own prompt containing only the state and that
question's own instructions and criteria. Two guardrail rubrics referred to "the
routing policy" and "the policies stated in this request", but that policy text
lived only in the sibling routing question. The model never saw it, and those
two rubrics were the worst in the run. This is the isolation design working as
specified, and a trap for callers: anything a question depends on must be in the
state or in that question's own instructions.

## Label revision and rerun

After the first run, two of the three cases the auditors called unsound were
fixed and everything was rerun once. This is regression evidence, not an unseen
evaluation, because the fixes were made after seeing the first results.

- `pii-bulk-export-buried/ctx_third_party`: gold changed from true to false. The
  rubric asks for a specific real person other than the sender; only the sender
  is named. Target-only change.
- `find-overlapping-room-booking`: the query presupposed a meeting the document
  never mentions, which made `none` defensible. The query now asks which Room A
  booking overlaps the 14:00 to 15:00 slot. Gold unchanged, contested flag
  removed. The model now answers all three questions correctly.
- `sde-invoice-wrong-row`: **kept as is.** The auditor called it ambiguous, but
  "is the extracted line one description wrong" is settled by the field name:
  the source's line one description is `pallet transfer`. The model answered true
  at 0.997 and 0.990. That is a model error worth keeping.

| | First run | After revision |
| --- | --- | --- |
| Overall | 1189/1286 = 0.925 | 1193/1286 = 0.928 |
| Trivial baseline (per question id) | 0.693 | 0.694 |
| passage-rerank-and-line-search | 144/149 | 147/149 |
| guardrail-policy-gates | 358/417 | 359/417 |
| Other five suites | unchanged | unchanged |

The only movement is the four questions in the two revised cases. Every other
answer is identical across the two runs.

A third run of the same 241 requests with shared-prefix reuse enabled scored
1194 of 1286; see [prefix-reuse.md](prefix-reuse.md) for why a handful of
borderline answers move between configurations.

## Genuine model failures worth keeping as regression items

- `next-thursday-reschedule`: resolved a relative date into concrete components
  despite an explicit instruction not to, while passing the same trap elsewhere.
- `refill-gate-controlled-drug`, `courier-gate-unsupported-country`: the
  instructions separate "which tool matches" from "is it permitted"; the model
  collapsed both into `none` at 0.997 and 1.000.
- Guardrail over-flagging: benign professional contexts (nurse documentation,
  paralegal form lookup, approved phishing simulation) routed to review or block.

## Recommended next revision (not done here)

1. Report the per-question-id baseline, case-level accuracy, and distinct
   question ids per suite in `summary.json`, and emit the contested split there.
2. Give rankings a full expected order and a minimum-margin guard, or drop
   pairwise from the headline.
3. Inline the policy into the two guardrail rubrics, and rebalance the hazard
   battery (74 to 82% false today). Any revised suite is regression evidence,
   not an unseen evaluation.
4. Rebuild entity alignment at a difficulty where it can fail.
5. Add cases near the documented limits: none exceeds 17 options, 5 score
   levels, 13 questions per request, or 705 prompt tokens.
