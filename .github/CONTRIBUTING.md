# Contributing to unridden

Bug reports, fixes, tests, documentation, and new use-case suites are welcome.
unridden is in early development and has no tagged release, so the HTTP
contract, the CLI flags, and the native worker protocol can still change.

Participation follows the [Code of Conduct](CODE_OF_CONDUCT.md). Report
suspected vulnerabilities through the private route in the
[security policy](SECURITY.md).

## Contributor signoff

Human authors, including the maintainer, must certify their contributions
under the [Developer Certificate of Origin 1.1](https://developercertificate.org/).
Read it before adding a `Signed-off-by` trailer to each authored commit:

```sh
git commit -s -m "fix: describe the change" --trailer "Github-Issue:#123"
```

Omit the issue trailer when no issue applies. `-s` records your name and email
from Git configuration. Check that they identify you before committing. The
signoff is a public, lasting record; a GitHub noreply email is acceptable when
it matches your commit identity. A DCO signoff is a rights attestation,
separate from a cryptographic commit signature or GitHub's Verified badge.

By signing off, you certify that you may submit the work under the applicable
license. Code and documentation use [Apache-2.0](../LICENSE) unless a file
states another license; the Code of Conduct retains its CC BY-SA 4.0 license.
Confirm any required employer permission. Preserve third-party licenses and
attribution, and raise unclear rights with the maintainer before submitting
the material.

For AI-assisted contributions, a human must review the result, its sources and
licenses, and their authority to submit it, then authorize their own signoff.
A tool must not add someone else's certification without that authorization.

Each human co-author must supply a signoff as well as any `Co-authored-by`
credit. The DCO app checks commit metadata; it does not establish ownership or
verify every co-author's certification. Maintainers review those cases before
merge.

If your latest commit is missing your signoff, and you can make the
certification:

```sh
git commit --amend --no-edit --signoff
git push --force-with-lease
```

For several unsigned commits, use interactive rebase to edit and sign only the
commits you can certify, then push with `--force-with-lease`. Coordinate
before rewriting a shared contribution branch. Keep the original authors and
their signoffs; never sign for someone else. Published `main` history is not
rewritten.

The DCO check must pass alongside CI before merge. Web commits require signoff
through GitHub's editor. When squashing a PR, preserve the authors' signoffs
in the final commit message. Automated bot and merge commits are exempt from
the app's metadata check; human-authored material still needs a human
certification.

## What this project will and will not take

This is a non-generative decision API. A request prefills a compiled prompt
and reads final-position logits over single-token labels; it emits zero
generated tokens. Changes that add sampling, append an answer token, or feed a
generated token back into the model are out of scope.

Two limits are legal, not stylistic, and a pull request that crosses either
will be closed:

- No benchmark, measurement, comparison, or verdict about TypeSafe's Jev API
  or any other TypeSafe service. That includes accuracy, latency, token, cost,
  calibration, and determinism numbers, and anything derived from calling it.
  Our own numbers on our own workloads are fine. The request and response
  shape follows TypeSafe's publicly documented Jev interface; this project is
  independent and is not affiliated with or endorsed by TypeSafe. Do not add
  their logos or claim parity.
- No model weights, no llama.cpp or ggml sources, and no other vendored
  third-party code in this repository. Users build llama.cpp and download
  Gemma themselves under the respective terms. See [NOTICE](../NOTICE).

Also keep secrets, API keys, machine-specific absolute paths, email addresses,
and personal data out of code, fixtures, docs, and issue text.

## Find or propose work

Search [existing issues](https://github.com/potto007/unridden/issues)
before opening one. Describe the problem you want to solve and how someone can
reproduce it. For a substantial feature or a change to the HTTP contract or
the worker protocol, discuss the approach in an issue before implementing it.
Small fixes and documentation changes can go straight to a pull request.

Issues labeled [good first issue](https://github.com/potto007/unridden/labels/good%20first%20issue)
or [help wanted](https://github.com/potto007/unridden/labels/help%20wanted)
are places to look for work. Comment on an issue you plan to take so others
can coordinate with you. New bug reports and feature requests start with
`needs-triage`; maintainers handle the triage labels.

## Development setup

Use Linux or macOS, Git, Python 3.12 or newer, and
[uv](https://docs.astral.sh/uv/). The native worker additionally needs a C++17
compiler and CMake 3.16 or newer; the GPU path needs your own llama.cpp build
and a Gemma GGUF, which this repository does not ship.

Fork the repository, clone your fork, and create a branch for your change:

```sh
git clone https://github.com/YOUR_USERNAME/unridden.git
cd unridden
git remote add upstream https://github.com/potto007/unridden.git
git switch -c fix/describe-the-change
uv sync --dev
```

You do not need a GPU, a model, or a llama.cpp build to work on most of the
project. The Python test suite runs entirely against fakes, and the C++ helper
test links nothing but the standard library. Only end-to-end validation
against a real model needs hardware; loading the default Gemma GGUF takes
roughly 18 GiB of VRAM, so check your free VRAM before opting in with `--gpu`.

## Make and check your change

Keep a pull request focused on one change. Match the surrounding style. For a
behavior change, add a regression that exercises what a caller can observe
through the HTTP API, the CLI, or the worker protocol; the tests in
`unridden/tests` show the pattern.

Run the checks CI runs, from the repository root:

```sh
uv run ruff format --check .
uv run ruff check .
uv run mypy --strict --explicit-package-bases unridden scripts
uv run pytest
```

`uv run ruff format .` applies the formatting rather than just reporting it.

The C++ helper test for `unridden/api/native/worker-utils.h` is
self-contained and needs no llama.cpp:

```sh
mkdir -p build
c++ -std=c++17 -Wall -Wextra -Werror -O2 \
  -o build/unridden-worker-utils-test \
  unridden/api/native/worker-utils-test.cpp
./build/unridden-worker-utils-test
```

CI builds and runs it exactly that way. Running it through CMake and `ctest`
instead also configures the full worker target, so `cmake` fails its configure
step unless you pass `LLAMA_SOURCE` and `LLAMA_BUILD` pointing at a llama.cpp
checkout and a matching build. Use the direct compile above when you only want
the helper test.

When a change makes or reverses an architecture decision, add an ADR under
`docs/decisions`. Supersede accepted ADRs rather than editing them.

For documentation-only changes, check examples and links and say that no code
tests were needed.

## Submit a pull request

1. Use a short, imperative Conventional Commit subject, such as
   `fix: reject an empty choice label set`, within 50 characters. When a
   commit addresses an issue, add its trailer with
   `git commit -s --trailer "Github-Issue:#123"`.
2. Push your branch to your fork and open a PR against `potto007/unridden`'s
   `main` branch. Draft PRs are useful for work that needs early feedback.
3. Explain the change, link the issue, and list the checks you ran. Use
   `Closes #123` only when the PR fully addresses that issue.
4. Keep your branch current with `main` and respond to review comments. The
   DCO check and all CI checks must pass, and review conversations must be
   resolved before merge.

Discuss the code and its behavior respectfully. Be specific when reporting a
problem or suggesting a change, and allow time for maintainer review.
