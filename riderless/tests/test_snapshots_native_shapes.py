"""Recorded native-shaped responses parsed through the worker models.

These fixtures mirror the exact JSON `snapshot-worker.cpp` emits (with the `id`
the main loop injects into every non-hello response), so a drift between the
worker and these models fails here rather than in a live run.
"""

from __future__ import annotations

import base64
import struct

from riderless.api.snapshots.schema import (
    WorkerCreated,
    WorkerDropped,
    WorkerErrorMessage,
    WorkerHello,
    WorkerInspect,
    WorkerLoaded,
    WorkerPromoted,
    WorkerResult,
    WorkerSaved,
    WorkerState,
    WorkerVectors,
)


def _f32(values: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode()


def _tensor(rows: int, width: int, representation: str) -> dict[str, object]:
    values = [float(i) for i in range(rows * width)]
    shape = [width] if rows == 1 else [rows, width]
    return {
        "dtype": "f32",
        "byte_order": "little",
        "shape": shape,
        "representation": representation,
        "base64": _f32(values),
    }


def _row(
    snapshot_id: str, blocks: int, parent: str | None, kind: str
) -> dict[str, object]:
    if blocks == 18:
        blob = {"lower_kv": 40, "upper_kv": 0, "h18": 40, "h30": 0}
    else:
        blob = {"lower_kv": 0, "upper_kv": 40, "h18": 0, "h30": 40}
    return {
        "snapshot_id": snapshot_id,
        "completed_blocks": blocks,
        "parent": parent,
        "kind": kind,
        "tokens": 10,
        "prompt_sha256": "a" * 64,
        "bytes": blob,
    }


def test_hello_shape() -> None:
    hello = WorkerHello.model_validate(
        {
            "type": "hello",
            "protocol": "riderless-snapshot-v1",
            "profile": "split18-30-v1",
            "model_id": "local-gemma-riderless-v1",
            "model_name": "gemma-4-26b",
            "model_sha256": "a" * 64,
            "runtime_sha256": "b" * 64,
            "labels": ["A", "B"],
            "label_token_ids": [11, 12],
            "context_size": 2048,
            "batch_size": 256,
            "ubatch_size": 256,
            "threads": 8,
            "n_layer": 30,
            "n_embd": 2816,
            "split_block": 18,
            "reference_context": False,
            "context_prompt_version": "riderless-gemma-context-v1",
            "generated_tokens": 0,
            "callbacks_enabled": False,
        }
    )
    assert hello.n_embd == 2816


def test_created_shape() -> None:
    created = WorkerCreated.model_validate(
        {
            "type": "created",
            "prompt_sha256": "a" * 64,
            "tokens": 10,
            "snapshots": [
                _row("snap_a", 18, None, "context"),
                _row("snap_b", 30, "snap_a", "context"),
            ],
            "readout": None,
            "block_tokens": {"lower": 10, "upper": 10},
            "timing_ms": {"lower": 1.5, "upper": 2.5, "total": 4.0},
            "generated_tokens": 0,
            "id": "c1",
        }
    )
    assert {row.completed_blocks for row in created.snapshots} == {18, 30}
    assert created.snapshots[0].kind == "context"


def test_promoted_shape() -> None:
    promoted = WorkerPromoted.model_validate(
        {
            "type": "promoted",
            "snapshot": _row("snap_b", 30, "snap_a", "context"),
            "block_tokens": {"lower": 0, "upper": 10},
            "timing_ms": {"upper": 3.0, "total": 3.0},
            "generated_tokens": 0,
            "id": "p1",
        }
    )
    assert promoted.snapshot.completed_blocks == 30


def test_evaluate_result_shape() -> None:
    result = WorkerResult.model_validate(
        {
            "type": "result",
            "model_sha256": "a" * 64,
            "runtime_sha256": "b" * 64,
            "generated_tokens": 0,
            "callbacks_enabled": False,
            "execution_mode": "split18-30",
            "questions": [
                {
                    "id": "status",
                    "label_logits": [0.0, 1.0],
                    "label_token_ids": [11, 12],
                    "allowed_label_mass": 0.9,
                    "full_vocabulary_argmax": {"token_id": 12, "logit": 3.0},
                    "prompt_sha256": "a" * 64,
                    "prompt_tokens": 16,
                    "processed_tokens": 6,
                    "reused_tokens": 10,
                    "cache_cleared": False,
                    "evaluation_mode": "sequential",
                    "batch_sequences": 1,
                    "timing_ms": 2.0,
                    "snapshot": {
                        "parent": "snap_b",
                        "suffix_tokens": 6,
                        "block_tokens": {"lower": 6, "upper": 6},
                        "restore": "resident",
                        "restored_bytes": 0,
                        "restore_ms": 0.0,
                        "inference_ms": 2.0,
                        "child": None,
                    },
                }
            ],
            "id": "e1",
        }
    )
    assert result.questions[0].snapshot.suffix_tokens == 6


def test_state_shape() -> None:
    state = WorkerState.model_validate(
        {
            "parent": "snap_b",
            "suffix_tokens": 4,
            "block_tokens": {"lower": 4, "upper": 4},
            "restore": "host",
            "restored_bytes": 4096,
            "restore_ms": 1.0,
            "inference_ms": 2.0,
            "child": _row("snap_c", 30, "snap_b", "readout"),
            "type": "state",
            "prompt_sha256": "a" * 64,
            "prompt_tokens": 14,
            "timing_ms": 3.0,
            "vectors": {
                "last_residual": _tensor(1, 4, "raw_residual_after_block_30"),
                "last_normalized": _tensor(1, 4, "post_final_norm_head_input"),
            },
            "top_logits": [{"token_id": 5, "logit": 1.0}],
            "generated_tokens": 0,
            "id": "s1",
        }
    )
    assert state.timing_ms == 3.0
    assert state.vectors["last_residual"].shape == [4]


def test_inspect_shape() -> None:
    inspect = WorkerInspect.model_validate(
        {
            **_row("snap_b", 30, "snap_a", "context"),
            "type": "snapshot",
            "token_ids": [1, 2, 3],
            "resident": True,
            "holds": {
                "h18": False,
                "h30": True,
                "last_normalized": True,
                "upper_kv": True,
            },
            "id": "i1",
        }
    )
    assert inspect.resident is True
    assert inspect.holds.h30 is True


def test_vectors_shape() -> None:
    vectors = WorkerVectors.model_validate(
        {
            "type": "vectors",
            "snapshot_id": "snap_b",
            "which": "h30",
            "rows": 100,
            "row_begin": 0,
            "row_end": 2,
            "tensor": _tensor(2, 4, "raw_residual_after_block_30"),
            "id": "v1",
        }
    )
    assert vectors.rows == 100
    assert vectors.tensor.shape == [2, 4]


def test_save_load_drop_shapes() -> None:
    saved = WorkerSaved.model_validate(
        {
            "type": "saved",
            "snapshot_id": "snap_b",
            "files": {
                "native.json": {"bytes": 200, "sha256": "a" * 64},
                "lower_kv.bin": {"bytes": 4096, "sha256": "b" * 64},
            },
            "id": "sv1",
        }
    )
    assert "native.json" in saved.files

    loaded = WorkerLoaded.model_validate(
        {
            **_row("snap_b", 30, "snap_a", "context"),
            "type": "loaded",
            "id": "ld1",
        }
    )
    assert loaded.completed_blocks == 30

    dropped = WorkerDropped.model_validate(
        {"type": "dropped", "snapshot_id": "snap_b", "freed_bytes": 4136, "id": "d1"}
    )
    assert dropped.freed_bytes == 4136


def test_error_shapes() -> None:
    typed = WorkerErrorMessage.model_validate(
        {
            "type": "error",
            "code": "snapshot_not_found",
            "reason": None,
            "message": "snap_x is unknown",
            "id": "err1",
        }
    )
    assert typed.code == "snapshot_not_found"
    assert typed.reason is None

    budget = WorkerErrorMessage.model_validate(
        {
            "type": "error",
            "code": "invalid_request",
            "reason": "budget",
            "message": "N + len(Q) exceeds context",
            "id": "err2",
        }
    )
    assert budget.reason == "budget"
