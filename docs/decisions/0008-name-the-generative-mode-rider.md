# 0008. Name the generative snapshot mode "Rider"

- Status: accepted
- Date: 2026-09-26
- Supersedes the naming in [0007](0007-generative-output-from-snapshots.md).
  Its decision, scope, bounds and measurements are unchanged.

## Context

0007 shipped generation from a snapshot as `POST /v2/outputs`, a worker
`generate` command and an `output_mode` handshake flag. "Output" says nothing
about why this route is the exception. The project is named for Haidt's rider
and elephant: Unridden reads the elephant's decision and never lets the rider
narrate. This mode is exactly where the rider speaks.

## Decision

- The mode is called **Rider**. The route is `POST /v2/rider`.
- Public schema: `RiderRequest`, `RiderResponse`, `RiderStep`, `RiderTiming`,
  `RiderUsage`, bounded by `MAX_RIDER_TOKENS` (still 1024).
- Worker protocol: the command is `ride`, its reply type is `ride_result`, and
  the handshake flag is `rider_mode`. The handshake schema forbids unknown
  fields, so a worker built between 0007 and this rename (which reports
  `output_mode`) is refused at startup; a worker older than 0007 reports
  neither flag, starts, and is refused only for Rider
  (`capability_unavailable`).
- No alias for `/v2/outputs`: it was never in a release.

## Consequences

- A snapshot worker built from 0007 must be rebuilt before the `/v2` backend
  starts at all; there is no release carrying the `output_mode` name.
- The zero-generated-tokens claim reads: it holds for `/v1` and every `/v2`
  route except Rider.
