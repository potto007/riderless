# Re-validation on llama.cpp v0.4.1

Run on 2026-09-21 on one RTX 5090 with Gemma 4 26B-A4B UD-Q4_K_XL. This page
exists because this project stopped hard-pinning one llama.cpp commit and
started recording a tested default instead
([0003](../decisions/0003-tested-default-revision-instead-of-a-hard-pin.md)).
Moving the tested revision changes which kernels run, so the harness and the
suites were re-run rather than reattributed.

## Which numbers came from which revision

| Page | Revision the numbers were measured on |
| --- | --- |
| [api-validation.md](api-validation.md) | `afeebe103bd99cda8f5dfaefcabadf890db7fda7` |
| [usecase-suites.md](usecase-suites.md) | `afeebe103bd99cda8f5dfaefcabadf890db7fda7` |
| [prefix-reuse.md](prefix-reuse.md) | `afeebe103bd99cda8f5dfaefcabadf890db7fda7` |
| [worker-build-comparison.md](worker-build-comparison.md) | `afeebe103bd99cda8f5dfaefcabadf890db7fda7`, held fixed on both sides of that comparison |
| This page | `v0.4.1` = commit `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, compared against `afeebe1` |
| [batched-mode.md](batched-mode.md) | `v0.4.1` = commit `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` |

Those pages are left exactly as they were measured. Nothing on them has been
rewritten to look like a v0.4.1 result.

## What was run

The v0.4.1 base runtime was built with CUDA for compute capability 120 only
(`--cuda-architectures 120`), and the worker was compiled against it with
`-Wall -Wextra -Werror`. Runtime configuration is unchanged on both sides:
context 2048, batch and ubatch 256, 8 threads, causal attention, `swa_full`,
CUDA fusion and graphs on, callbacks off, shared-prefix reuse on.

Both reference runs are the shared-prefix-reuse runs, which are the current
default: the 134-request validation run noted at the end of
[api-validation.md](api-validation.md), and the 241-request suite run reported
in [prefix-reuse.md](prefix-reuse.md).

Two confounds are disclosed rather than hidden:

- The reference runs predate the project rename, so their recorded model id is
  the older one. The comparison normalises that id and nothing else.
- Two suite cases had prompt-visible text edited during the rename, before this
  change: `candidate-span-and-date-extraction/terminal-transcript-vs-prose` and
  `passage-rerank-and-line-search/find-etl-restart-command`. Their compiled
  requests therefore differ between the two runs and are not evidence about the
  revision. Neither case changed an answer, and they account for the 16-token
  difference in prefilled tokens. The other 239 cases compile byte-identically.

## Conformance, isolation and failure paths

| Check | `afeebe1` | `v0.4.1` |
| --- | --- | --- |
| Harness result | pass | pass |
| Semantic smoke | 36/36 (Choice 17/17, Score 5/5, Noul 14/14) | 36/36 (Choice 17/17, Score 5/5, Noul 14/14) |
| Requests recorded | 134, of which 10 expected rejections | 134, of which 10 expected rejections |
| Generated tokens | 0 | 0 |
| Isolation comparisons | 72 | 72 |
| Max isolation probability delta | 0.0 | 0.0 |
| Max isolation raw-logit delta | 0.0 | 0.0 |
| Executed input tokens | 57,935 | 57,935 |
| Reused input tokens | 37,050 | 37,050 |

Isolation is still exact, which was the condition for publishing this at all.

The failure-path harness was re-run too. Both scenarios behave as documented: a
50 ms request timeout returns 408 `timeout` with the exact owned child reaped
and 529 afterwards on `/health` and on the next request; killing the exact owned
child by PID between requests returns 500 `internal_error` with no path or
stderr in the body, then 529 on both.

## Task quality: seven suites, 1,286 questions

| Metric | `afeebe1` | `v0.4.1` |
| --- | --- | --- |
| Overall accuracy | 1194/1286 = 0.9285 | 1197/1286 = 0.9308 |
| Choice | 377/400 = 0.9425 | 373/400 = 0.9325 |
| Noul | 740/782 = 0.9463 | 745/782 = 0.9527 |
| Score, exact level | 77/104 = 0.7404 | 79/104 = 0.7596 |
| Score, within tolerance | 94/104 = 0.9038 | 95/104 = 0.9135 |
| Ranking top-1 | 32/32 | 32/32 |
| Errored cases | 0 | 0 |
| Generated tokens | 0 | 0 |

By suite:

| Suite | `afeebe1` | `v0.4.1` |
| --- | --- | --- |
| candidate-span-and-date-extraction | 179/194 | 178/194 |
| claim-support-verification | 92/96 | 92/96 |
| entity-alignment-and-record-matching | 157/157 | 157/157 |
| guardrail-policy-gates | 359/417 | 364/417 |
| hierarchical-and-broad-to-fine-classification | 139/151 | 139/151 |
| passage-rerank-and-line-search | 148/149 | 147/149 |
| tool-and-argument-selection | 120/122 | 120/122 |

Overall accuracy moved by +0.23 points. That is not a quality improvement worth
claiming: it is 13 borderline questions landing differently, 8 of which happened
to fall on the right side.

## How much moved

| Measure | Value |
| --- | --- |
| Answers that changed | 13 of 1,286 (1.0%) |
| Of those, in cases whose compiled request is identical | 13 of 13 |
| Net correctness change | +3 (8 became correct, 5 became incorrect) |
| Largest probability move on any single label | 0.751 |

The largest move is `claim-support-verification/oil-pan-spec-elsewhere/relation_a`,
which flipped from `silent` to `contradicts`. The rest are the familiar
borderline population: date `day_unit` digits, guardrail hazard and context
Nouls, one `severity` Score level, one taxonomy `l1_division`.

This is the same effect already documented for a batch-shape change in
[prefix-reuse.md](prefix-reuse.md): about 1% of questions sit close enough to a
decision boundary that any change in how the arithmetic is carried out moves
them. A llama.cpp release changes kernels, so it moves them too. Determinism
here means bit-identical repeats within one configuration, not stability across
revisions.

## Latency and memory

| Measure | `afeebe1` | `v0.4.1` |
| --- | --- | --- |
| Validation API latency, median | 97.3 ms | 95.8 ms |
| Validation API latency, p95 | 259.2 ms | 240.5 ms |
| Suite request latency, median | 194.4 ms | 179.8 ms |
| Suite request latency, p95 | 521.2 ms | 508.9 ms |
| Suite tokens prefilled | 246,997 | 246,981 |
| Peak GPU memory, whole card | 22,756 MiB | 23,018 MiB |
| GPU memory before load | 5,151 MiB | 5,382 MiB |
| Peak minus pre-load baseline | 17,605 MiB | 17,636 MiB |

Both latency samples are single runs on an otherwise idle card, three repeats
per benchmark point, so the few percent of apparent speedup is descriptive and
not a claim. The worker's own footprint is unchanged within 31 MiB; the
difference in whole-card peak is the difference in what else the desktop held
before the model loaded.

## What this does not establish

- One run per revision. Nothing here separates a revision effect from run-to-run
  variation on the 13 changed answers, beyond the fact that repeats within a
  configuration are bit-identical.
- The suites are the same hand-authored cases with the same caveats as in
  [usecase-suites.md](usecase-suites.md). A +0.23 point move over 1,286
  clustered, non-independent questions is inside the noise those caveats
  describe.
- No revision between `afeebe1` and `v0.4.1` was tested. The two endpoints are
  782 commits apart.
