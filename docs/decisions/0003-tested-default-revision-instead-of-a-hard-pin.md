# 0003. A tested default llama.cpp revision instead of a hard pin

- Status: accepted
- Date: 2026-09-21

## Context

The worker links llama.cpp: `libllama`, the internal `llama-common` library for
the chat templates, and the ggml backends. Until now one commit,
`afeebe103bd99cda8f5dfaefcabadf890db7fda7`, was written into the build as
`FROZEN_LLAMA_REVISION`, and three places refused anything else: the base
runtime script fetched only that commit, the worker build refused a base built
from another one, and API startup refused a manifest that named another one.

That commit had no property that earned it. It was whatever an earlier
proof of concept happened to have checked out. The refusal it justified was
integrity, and integrity is what the hashes already provide: the build records a
sha256 for every shared library, the worker executable, and the worker sources,
and startup rehashes all of them. Refusing a different revision on top of that
protected nothing; it only told anyone building the project that their fresh
llama.cpp was wrong, and it gave a README instruction that will age badly with
every release.

What a recorded revision genuinely supports is provenance: the measurements in
`docs/results/` were taken on one revision, and a reader deserves to know when
they are looking at numbers from a build that is not the one measured.

## Decision

The build records the release this project is tested against, as both a tag and
the commit that tag resolved to:

```python
TESTED_LLAMA_TAG = "v0.4.1"
TESTED_LLAMA_REVISION = "b29c606e28a01b1bc8c1351026a0fa6e616bf6c4"
```

- `build_base_runtime.py` fetches `TESTED_LLAMA_TAG` by default and accepts any
  other tag or commit through `--revision`. Asking for the tested tag is checked
  against the recorded commit, so an upstream tag that moves is a build failure
  rather than a silent substitution. The base `build.json` records what was
  requested (`llama_ref`), what it resolved to (`llama_revision`), and whether
  that is the tested one.
- Integrity is unchanged. Every hash check in the worker build and at API
  startup runs exactly as before, whatever the revision.
- A revision other than the tested one logs one warning saying that the
  published measurements were taken on the tested revision and may not
  reproduce. It is never a refusal.
- `GET /v1/models` reports `runtime.llama_revision` and
  `runtime.tested_revision` beside the other provenance it already carried, so
  the fact travels with a running service and not only with a build directory.

## Alternatives rejected

- **Keep the hard pin, bump the commit.** Cheapest change, and it keeps the
  original defect: the next reader is still told that one commit is the only
  correct one, with nothing behind the claim.
- **Drop the revision from the manifest entirely.** Integrity would survive, but
  a result could then never be traced to the llama.cpp that produced it, and the
  numbers in `docs/results/` would become unattributable.
- **Refuse untested revisions unless an override flag is passed.** A flag that
  everybody passes is not a safety property, and it would still have to be
  explained in the README on every release.

## Consequences

- Answers move between revisions. Kernels change upstream, so the same prompt
  can land on a different code path and shift a borderline label. The v0.4.1
  re-validation measures how much: see
  [the re-validation results](../results/llama-v0.4.1-revalidation.md).
- Published numbers now carry the revision they were measured on. Tables
  measured on `afeebe1` stay labelled as such rather than being quietly
  reattributed to v0.4.1.
- The worker source is no longer portable backwards. llama.cpp v0.4.1 moved the
  common library off `nlohmann::ordered_json` and onto its own `common_json`,
  and moved the bundled sha256 helper into `vendor/hash`. The CMake project
  accepts either helper location; the chat-template call site targets v0.4.1
  only. "Other revisions work if the common library API still matches" is the
  honest claim, and it is the one the README makes.
