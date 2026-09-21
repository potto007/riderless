# Security policy

## Supported code

riderless is in early development and has no tagged releases yet. We
investigate suspected vulnerabilities in the current `main` branch. Include
the commit you tested so we can identify the affected code.

There is no maintained release series or backport commitment. When releases
exist, this section will identify which versions receive security fixes.

## Report a vulnerability privately

Use [GitHub private vulnerability reporting](https://github.com/potto007/riderless/security/advisories/new).
Do not open a public issue or pull request with exploit details or live
credentials. Reports are reviewed by the maintainer, `potto007`, through the
private advisory; people needed to investigate or fix the problem may be
invited.

Include what you know:

- Affected component and tested commit.
- Required configuration, access, and other prerequisites, including the
  Python version, the llama.cpp revision, and whether the GPU path was used.
- Security impact and steps to reproduce.
- A minimal proof using made-up credentials and a small request payload.
- Any mitigation you have found.

Remove API keys, tokens, machine-specific absolute paths, and personal or
customer data from attachments and from any request or state text you paste.
Test only on systems you control or have permission to assess. Avoid accessing
other people's data or disrupting live deployments.

If details have already been posted publicly, link that post in the private
report and avoid adding further exploit detail there.

## Scope

Reports about these components belong here:

- The `riderless` Python package: the HTTP application, the request compiler,
  the label mapping and schema validation, and the CLI.
- The native worker under `riderless/api/native`, its build script, and the
  manifest and hash pinning it performs at build time and at startup.
- The validation and use-case scripts under `scripts/riderless`.

Examples of what we consider in scope: a caller-supplied state or label set
that escapes prompt or protocol framing, a request that makes the worker read
or write a path outside its configured inputs, a path that lets a request run
an unpinned worker executable or an unpinned model, resource exhaustion that
survives the configured request, response, and timeout limits, and any leak of
local file contents through an API response or error.

Out of scope here:

- Faults in llama.cpp or ggml themselves. Report those to
  [llama.cpp](https://github.com/ggml-org/llama.cpp/security). Report them
  here too if this project's use of them creates or worsens the exposure.
- The quality, calibration, or correctness of a model's answers. A wrong
  decision from a model is not a vulnerability.
- Exposing the API on an untrusted network. It is intended for a trusted
  local caller, serves no authentication of its own, and defaults to a
  loopback listener; deploying it otherwise is an operator decision.
- Any finding whose write-up requires measuring a third-party hosted API.
  Report the local fault and leave the third-party numbers out.

Ordinary bugs, usage questions, and feature requests belong in
[GitHub Issues](https://github.com/potto007/riderless/issues).
Conduct concerns follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Triage and disclosure

The maintainer reviews reports, asks for missing details, and coordinates
fixes and disclosure with the reporter through the private advisory. Timing
depends on severity and maintainer availability; there is no guaranteed
acknowledgement or fix deadline and no bug bounty.

For a confirmed issue, we aim to publish an advisory alongside a fix or useful
mitigation, with affected commits or versions and a fix reference. Reporter
credit is included when the reporter wants it. Keep exploit details and
proposed fixes in the private advisory while we coordinate publication.
