"""A local, non-generative decision API.

State plus typed questions in, label probability distributions out, read from
final-position logits over single-token labels. No sampled token is ever
produced, so there is nothing to parse and no malformed answer to recover from.

The public surface lives in `unridden.api`.
"""
