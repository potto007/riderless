# 0001. Read full-depth label logits from one owned native child

- Status: accepted
- Date: 2026-09-21
- Note: renumbered from an earlier internal record, with the machine-specific
  context removed. The part about a fresh context per question is superseded by
  [0002](0002-share-state-prefix-within-a-request.md); the rest stands.

## Context

An earlier proof of concept showed that a decision can be read out of a forward
pass with zero generated tokens, but only for one registered three-class task
with a trained head at a fixed model block. The goal here is a local decision
API for caller-defined Choice, Score, and Noul questions, using the request
shape that TypeSafe documents publicly for Jev.

An older internal facade answered such questions by asking a generative server
for one token (`n_predict: 1`) and returned an argmax Score. That is neither
zero-generation nor conformant with the documented Score semantics.

## Decision

1. New question semantics use a **full-depth finite-label readout**: prefill the
   compiled prompt, read final-position logits over validated single-token
   labels (A to Z, so at most 26 options), softmax over the allowed labels. No
   sampler, no answer token, no autoregressive loop, no speculative draft. A
   head trained for one task is never applied to other questions, even
   three-option ones.
2. The model runs in **one owned native child process** built against a pinned
   llama.cpp revision, speaking bounded JSONL over stdio with correlation ids.
   It opens no port. Any ambiguous transport failure, timeout, or cancellation
   reaps exactly that child, and the backend stays unavailable (529) until the
   app restarts. There is no automatic restart.
3. **Each question gets its own prompt** built from the state plus that
   question's own instructions and criteria. Question ids and sibling questions
   never enter a prompt. The worker verifies the context memory before prefill.
4. Result semantics: Choice returns the argmax option plus the full
   distribution; Score is the zero-based expectation `sum(i * p_i)`; Noul is
   `P(true)`; confidence is `max_probability_v1` and is not calibrated.
5. Provenance: create-only native build snapshots whose manifest pins the
   executable and runtime hashes, plus a model hash pin that startup enforces.
   Caller text that spells a model control token is rejected, because the prompt
   is tokenized with special-token parsing for the chat template.

## Alternatives rejected

- **Extend a trained early-exit head to arbitrary questions.** A head is trained
  for one task's meaning. A question-conditioned universal head is a separate
  training and evaluation project.
- **Call a generative server with `n_predict: 1` and logprobs.** That generates
  a token, depends on whichever model that server has loaded, and cannot
  guarantee a cleared cache per question.
- **Share the state prefix across questions in one context.** Faster, but it
  needs its own isolation proof. v1 chose the auditable baseline first, and
  0002 revisits it with that proof.
- **Restart the child automatically after a failure.** A silent restart reloads
  about 17 GiB onto a GPU without anyone checking headroom.

## Consequences

- Measured at the time: about 30 ms per 110-token question, linear in question
  count because the state is prefilled again for every question; 22.5 GiB peak
  GPU memory.
- Isolation is provable: 72 mixed, solo, reordered, renamed, and repeated
  comparisons differ by exactly 0.0, and a negative-control worker with the
  cache clear removed fails the same harness.
- A policy or definition placed in one question is invisible to its siblings.
  Callers must put shared context in the state. A use-case suite fell into this
  trap and scored worst on exactly those rubrics.
- Task quality on seven Jev-style use-case suites is 0.925 against a 0.693
  trivial baseline (first frozen suite run; the revised run with prefix reuse is
  0.928 against 0.694, see [use-case suites](../results/usecase-suites.md)), and
  confidence does not separate right from wrong answers
  (0.995 against 0.945), so it must not gate decisions without calibration.
- Standing use would need a supervised service unit and logging. None exists;
  v1 is validated through the in-process ASGI app and the batch CLI only.
- Evidence: [API validation](../results/api-validation.md) and
  [use-case suites](../results/usecase-suites.md).
