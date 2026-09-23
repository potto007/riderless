"""Experimental /v2 reusable state-snapshot service (design 2026-09-22).

This package is the Python side of the snapshot protocol
(`riderless-snapshot-v1`): strict wire and worker schemas, a deterministic
context-profile compiler, an in-process snapshot store with leases and a byte
budget, a bounded JSONL backend that owns one `riderless-snapshot-worker`
child, and the request-serialised service the app wires in when snapshots are
enabled. The v1 API is untouched; nothing here loads a model or touches a GPU.
"""
