# The use-case suites

Seven suites in [`unridden/examples/usecases/`](../unridden/examples/usecases/)
hold 241 cases and 1,286 questions of original, hand-authored material with gold
labels. They exist to measure task quality beyond the 36-question conformance
smoke set, across the families of workload this kind of API is used for.

Measured outcomes are in [results/usecase-suites.md](results/usecase-suites.md).
This page describes what the suites contain and how to run and score them.

## The seven suites

| Suite | Cases | Questions | What it measures |
| --- | --- | --- | --- |
| `hierarchical-and-broad-to-fine-classification` | 36 | 151 | One compact three-level intake taxonomy (7 divisions plus an explicit other, 5 groups per division, 4 categories per group), one Choice per level, with the gold parent path named in deeper instructions. Cases are built around near-miss sibling pairs, so per-node accuracy and full-path exact match can both be read from one run. Includes beam-frontier cases where a rival branch's gold answer is `none_applicable`, and paired determined/underdetermined twins whose confidences are compared within the pair. |
| `candidate-span-and-date-extraction` | 33 | 194 | Selecting a verbatim candidate span (with an explicit none option), decomposing a date expression into components that caller code resolves, and recovering document structure from extracted lines. Day of month is deliberately split into a ten-day band Choice and a final-digit Choice, because 31 options would exceed the 26-label ceiling. |
| `passage-rerank-and-line-search` | 36 | 149 | Two result semantics kept apart: a Noul battery whose P(true) values order a shortlist, and a Choice over tagged line or candidate ids paired with an independent Noul that says whether an answer is present at all. Every ranking case has exactly one passage that entails the answer, so top-1 is unambiguous. |
| `claim-support-verification` | 36 | 96 | Placing a claim against a source excerpt in one of three relations (supports, contradicts, silent), separating answer evidence from premise contradiction, and flagging a per-field extraction error including a wrongly empty field. The discriminating axis is the silent class: a third of the relation cases are claims the excerpt neither states nor denies while sharing its vocabulary. |
| `tool-and-argument-selection` | 32 | 122 | Assembling a tool call in one request: a Choice over a catalogue that always carries an explicit none option, closed-enumeration Choices for arguments, Nouls for optional-argument presence and for set members, and applicability gates that can reject an otherwise correct tool. Catalogues contain overlapping siblings, so the pick turns on one clause. |
| `guardrail-policy-gates` | 36 | 417 | A fixed six-hazard Noul battery, one ordered severity Score, and one routing Choice over hazardous messages and near-miss benign text that shares their surface vocabulary. Sixteen of the 36 cases are benign. Every gold label is derivable from a policy sentence written into the question instructions. Two graded triples carry the ordinal signal. |
| `entity-alignment-and-record-matching` | 32 | 157 | One ordered three-level Score (different, review, same) plus field-level Nouls over candidate pairs whose surface similarity is deliberately misleading. Every case is decidable from the state alone: matching policy, authoritative identifier, and any relationship needed for the judgment are written into the state. |

Each suite file carries a `description` that states its intended metrics and,
explicitly, what it does not measure. Read that before quoting a number from it.

## Suite file format

```json
{
  "suite": "<slug, unique across suites>",
  "family": "<human-readable family>",
  "source": ["<design reference>", "..."],
  "description": "<what it measures, what it does not>",
  "cases": [
    {
      "name": "<unique within the suite>",
      "difficulty": "easy | medium | hard",
      "rationale": "<why the gold label is what it is>",
      "request": {"state": {...}, "questions": {...}},
      "targets": {
        "<question id>": {"type": "choice", "expected": "<option id>"},
        "<question id>": {"type": "noul", "expected": true},
        "<question id>": {"type": "score", "expected_level": 2, "tolerance": 1}
      },
      "rankings": [
        {"name": "<ranking id>",
         "question_ids": ["q1", "q2", "q3"],
         "expected_order": ["q2"]}
      ]
    }
  ]
}
```

Rules the loader enforces before any model is touched: a case's `request` must
not set `model`; targets must cover exactly the request's question ids and match
each question's type; a Choice target must name a declared option; a Score target
must be a level inside the criteria range with a nonnegative integer tolerance;
ranked questions must all be Noul; `expected_order` must be a unique subset of
the ranked ids; case names and suite slugs must be unique. Anything else is a
hard failure of the whole run, so a malformed suite cannot burn GPU time.

`expected_order` is a prefix, not necessarily a total order: any ranked id it
omits is treated as worse than every listed one when pairs are counted.

## Running them

```bash
PYTHONPATH=. python scripts/unridden/run_usecase_suites.py \
  --suites unridden/examples/usecases \
  --out runs/usecase-suites-001 \
  --worker build/api-worker/build/unridden-worker \
  --manifest build/api-worker/build.json \
  --model-path models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --allow-gpu
```

- `--allow-gpu` is mandatory and has no default. The runner refuses to start
  without it, so a stray invocation cannot load a model onto a busy GPU.
- `--out` must not exist. Nothing is ever overwritten.
- `--batch-size` sets both batch and ubatch. `--no-share-prefix` restores the
  one-full-prefill-per-question behaviour. Both change the prefill batch shape,
  so hold them fixed when comparing two runs.
- `--batched` turns on the opt-in batched mode of
  [decisions/0004](decisions/0004-optional-batched-question-evaluation.md), and
  `--batched-context` sizes its KV cache. It too changes the batch shape, so it
  is a third configuration to hold fixed, not a free speedup. The run records
  all four settings in `run.json`.
- `scripts/unridden/compare_observations.py --left <a>/observations.jsonl
  --right <b>/observations.jsonl` reduces two runs to answer changes, the
  largest probability and raw logit moves, tokens, and latency.
- One backend serves all 241 requests, with one model load.
- The runner prints `SUITE_COMPLETE <slug>` after each suite and one terminal
  line: `USECASE_RUN_COMPLETE questions=... accuracy=... errored_cases=...` or
  `USECASE_RUN_FAILED ...`.

Order of operations, which is the point of the design: every suite is loaded and
fully validated, then each suite file's SHA-256 and the worker, manifest, and
model hashes are written into `run.json`, and only then is the model loaded. The
labels are frozen on disk and recorded before they can be influenced by a result.

## Output

| File | Contents |
| --- | --- |
| `run.json` | State (`starting`, `running`, `complete`, `failed`), suite hashes and case counts, worker/manifest/model hashes, startup seconds, the owned child's pid, and timestamps. A run interrupted mid-flight is left as `running`, which is how you tell it apart from evidence. |
| `observations.jsonl` | One line per case: the full request, the full response including per-question diagnostics, per-question outcomes, ranking outcomes, and latency. Written as the run proceeds, so a failed run keeps everything up to the failure. |
| `summary.json` | Aggregates (below). Written only on a complete run. |
| `native-worker.log` | The worker's own stderr, captured at DEBUG. |
| `native-stderr-tail.json` | On failure only: the last lines of worker stderr. |

A rejected request is recorded as an errored case and does not stop the run; a
nonzero generated-token count anywhere does stop it.

## Scoring

The runner scores each answer against its target:

- **Choice**: correct if the returned option id equals the expected one.
- **Noul**: thresholded at 0.5. It also reports the Brier score and the mean
  P(true) separately on true and false targets.
- **Score**: the most probable level is compared with the expected level, exactly
  and within the case's tolerance. Ties resolve to the lowest level. Mean
  absolute error uses the returned fractional expectation, not the level.
- **Rankings**: candidates are ordered by P(true), ties broken by declared order.
  Reports top-1 hits and concordant pairs over every (better, worse) pair implied
  by `expected_order`.

`summary.json` carries those metrics overall, per suite, and per difficulty
tier, plus a five-bin reliability table, a majority-label baseline per question
type, and the full list of failed questions and failed rankings so a regression
can be diffed rather than re-eyeballed.

Two cautions carried over from the auditors of the first run. The per-type
majority baseline that the runner emits is weaker than the per-question-id
baseline quoted in the results, so do not read the runner's own baseline as the
headline comparison. And pairwise ranking agreement mostly restates top-1 when
`expected_order` declares only a winner.
