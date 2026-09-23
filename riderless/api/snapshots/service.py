"""Request-serialised orchestration for the /v2 snapshot routes.

Like `ApiService`, one request runs at a time and a second is refused with a
busy signal; the model is never asked to interleave two branches. The service
owns the promotion/branching decisions the design describes and keeps the store
and worker consistent: it leases a parent for the life of a call, restores it
if it was evicted, memoises promotion, and registers any saved child.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from riderless.api.schema import MODEL_ID, Answer, Usage
from riderless.api.snapshots.backend import SnapshotBackend
from riderless.api.snapshots.compiler import (
    CompiledSnapshotQuestion,
    Message,
    compile_context_followup,
    compile_context_prompt,
    compile_create,
    compile_readout_followup,
    compile_readout_prompt,
)
from riderless.api.snapshots.errors import (
    CapabilityUnavailable,
    FollowupUnsupported,
    SnapshotProtocolError,
    SnapshotRequestError,
    SnapshotUnavailableError,
)
from riderless.api.snapshots.mapping import map_answer
from riderless.api.snapshots.schema import (
    SNAPSHOT_PROFILE,
    BlockTokens,
    Persistence,
    Promotion,
    SnapshotCreateRequest,
    SnapshotCreateResponse,
    SnapshotMetadata,
    SnapshotRef,
    SnapshotRow,
    SnapshotTiming,
    SnapshotUsage,
    StateEvaluationRequest,
    StateEvaluationResponse,
    V2DecisionRequest,
    V2DecisionResponse,
    VectorArtifact,
    WorkerHello,
    WorkerSnapshotRow,
    WorkerTensor,
)
from riderless.api.snapshots.store import SnapshotRecord, SnapshotStore, new_snapshot_id

# The export kind and the representation tag the worker must stamp on it, so a
# raw residual is never mistaken for a post-norm head input.
_REPRESENTATION = {
    "last_residual": "raw_residual_after_block_30",
    "last_normalized": "post_final_norm_head_input",
}


class SnapshotBusyError(RuntimeError):
    """The one-request snapshot service is occupied."""


def _row_bytes(row: WorkerSnapshotRow) -> int:
    blob = row.bytes
    return blob.lower_kv + blob.upper_kv + blob.h18 + blob.h30


@dataclass(frozen=True, slots=True)
class BranchPlan:
    requested_parent: str
    effective_parent: str
    promotion: Promotion
    promotion_ms: float
    mode: Literal["context", "readout"]
    messages: list[Message]
    answer_prefix: str
    # The context anchor a saved child records, so a later replace_question can
    # branch from it. None for a readout-only chain.
    child_context_parent: str | None


class SnapshotService:
    def __init__(
        self,
        backend: SnapshotBackend,
        *,
        store_root: str,
        host_bytes: int,
        default_ttl: int,
        request_timeout: float,
        startup_timeout: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._backend = backend
        self._store_root = store_root
        self._host_bytes = host_bytes
        self._default_ttl = default_ttl
        self._request_timeout = request_timeout
        self._startup_timeout = startup_timeout
        self._clock = clock
        self.profile: WorkerHello | None = None
        self.store: SnapshotStore | None = None
        self._guard = asyncio.Lock()
        self._busy = False
        self._pending_ids: list[str] = []

    async def start(self) -> None:
        profile = await asyncio.wait_for(
            self._backend.start(), timeout=self._startup_timeout
        )
        self.store = SnapshotStore(
            root=Path(self._store_root),
            native=self._backend,
            model_sha256=profile.model_sha256,
            runtime_sha256=profile.runtime_sha256,
            profile=profile.profile,
            n_embd=profile.n_embd,
            context_size=profile.context_size,
            host_bytes=self._host_bytes,
            clock=self._clock,
        )
        await self.store.expire()
        self.profile = profile

    async def close(self) -> None:
        await self._backend.close()

    @property
    def ready(self) -> bool:
        return (
            self.profile is not None and self.store is not None and self._backend.ready
        )

    @contextlib.asynccontextmanager
    async def _serialized(self) -> AsyncIterator[tuple[WorkerHello, SnapshotStore]]:
        if self.profile is None or self.store is None or not self._backend.ready:
            raise SnapshotUnavailableError("snapshot backend unavailable")
        async with self._guard:
            if self._busy:
                raise SnapshotBusyError("snapshot backend busy")
            self._busy = True
        try:
            await self.store.expire()
            yield self.profile, self.store
        except BaseException:
            await self._rollback(self.store, self._pending_ids)
            raise
        finally:
            self._pending_ids = []
            async with self._guard:
                self._busy = False

    async def _rollback(self, store: SnapshotStore, ids: list[str]) -> None:
        """Reclaim worker IDs whose request did not publish a complete result."""
        for snapshot_id in reversed(ids):
            try:
                await store.delete(snapshot_id)
            except SnapshotUnavailableError:
                # A timed-out transport already reaped its worker.
                continue
            except Exception:
                # Unregistered IDs are still held by a healthy native worker.
                # Do not mask the request's original failure during rollback.
                with contextlib.suppress(Exception):
                    await self._backend.drop(snapshot_id)

    # -- create ------------------------------------------------------------

    async def create(
        self, request: SnapshotCreateRequest, *, owner: str
    ) -> SnapshotCreateResponse:
        async with self._serialized() as (profile, store):
            compiled = compile_create(request.input, profile.labels)
            checkpoints = {
                str(block): new_snapshot_id() for block in request.checkpoints
            }
            requested_ids = set(checkpoints.values())
            freeze = (
                {"kind": "context", "content_bytes": compiled.content_bytes}
                if compiled.boundary == "context"
                else {"kind": "readout"}
            )
            wants_30 = 30 in request.checkpoints
            labels = compiled.labels if (wants_30 and compiled.labels) else None
            try:
                result = await self._backend.create(
                    messages=compiled.messages,
                    answer_prefix=compiled.answer_prefix,
                    freeze=freeze,
                    checkpoints=checkpoints,
                    labels=labels,
                    top_logits=0,
                    timeout=self._request_timeout,
                )
                returned = {row.snapshot_id for row in result.snapshots}
                if returned != requested_ids:
                    raise SnapshotProtocolError(
                        "worker returned unexpected snapshot ids"
                    )
                rows: list[SnapshotRow] = []
                for row in sorted(
                    result.snapshots, key=lambda item: item.completed_blocks
                ):
                    record = await store.register(
                        snapshot_id=row.snapshot_id,
                        owner=owner,
                        completed_blocks=row.completed_blocks,
                        boundary=compiled.boundary,
                        parent=row.parent,
                        context_parent=None,
                        messages=compiled.messages,
                        answer_prefix=compiled.answer_prefix,
                        tokens=row.tokens,
                        host_bytes=_row_bytes(row),
                        persistence=request.persistence,
                        prompt_sha256=result.prompt_sha256,
                        ttl_seconds=request.ttl_seconds,
                    )
                    rows.append(
                        SnapshotRow(
                            id=record.id,
                            completed_blocks=record.completed_blocks,
                            boundary=record.boundary,
                            parent=record.parent,
                            context_parent=record.context_parent,
                            capabilities=record.capabilities(),  # type: ignore[arg-type]
                        )
                    )
                return SnapshotCreateResponse(snapshots=rows)
            except BaseException:
                await self._rollback(store, list(checkpoints.values()))
                raise

    # -- planning ----------------------------------------------------------

    async def _ensure_30(
        self, store: SnapshotStore, record: SnapshotRecord, persistence: Persistence
    ) -> tuple[str, Promotion, float]:
        if record.completed_blocks == 30:
            return record.id, "none", 0.0
        existing = store.memoized_promotion(record.id)
        if existing is not None:
            return existing, "reused", 0.0
        with store.lease(record.id):
            await store.restore(record.id)
            new_id = new_snapshot_id()
            try:
                promoted = await self._backend.promote(
                    snapshot_id=record.id,
                    new_id=new_id,
                    timeout=self._request_timeout,
                )
                if promoted.snapshot.snapshot_id != new_id:
                    raise SnapshotProtocolError("promotion returned an unexpected id")
                await store.register(
                    snapshot_id=promoted.snapshot.snapshot_id,
                    owner=record.owner,
                    completed_blocks=30,
                    boundary=record.boundary,
                    parent=record.id,
                    context_parent=(
                        record.context_parent if record.boundary == "readout" else None
                    ),
                    messages=record.messages,
                    answer_prefix=record.answer_prefix,
                    tokens=promoted.snapshot.tokens,
                    host_bytes=_row_bytes(promoted.snapshot),
                    persistence=persistence,
                    prompt_sha256=promoted.snapshot.prompt_sha256,
                    ttl_seconds=self._default_ttl,
                    promotion_of=record.id,
                )
            except BaseException:
                await self._rollback(store, [new_id])
                raise
        store.memoize_promotion(record.id, promoted.snapshot.snapshot_id)
        return promoted.snapshot.snapshot_id, "performed", promoted.timing_ms.total

    async def _plan(self, store: SnapshotStore, ref: SnapshotRef) -> BranchPlan:
        record = store.get(ref.id)
        if ref.relationship == "replace_question":
            if record.boundary != "readout" or record.context_parent is None:
                raise FollowupUnsupported(
                    "snapshot has no context parent to replace its question"
                )
            context = store.get(record.context_parent)
            effective, promotion, promotion_ms = await self._ensure_30(
                store, context, context.persistence
            )
            return BranchPlan(
                requested_parent=ref.id,
                effective_parent=effective,
                promotion=promotion,
                promotion_ms=promotion_ms,
                mode="context",
                messages=context.messages,
                answer_prefix=context.answer_prefix,
                child_context_parent=effective,
            )
        effective, promotion, promotion_ms = await self._ensure_30(
            store, record, record.persistence
        )
        if record.boundary == "context":
            return BranchPlan(
                requested_parent=ref.id,
                effective_parent=effective,
                promotion=promotion,
                promotion_ms=promotion_ms,
                mode="context",
                messages=record.messages,
                answer_prefix=record.answer_prefix,
                child_context_parent=effective,
            )
        return BranchPlan(
            requested_parent=ref.id,
            effective_parent=effective,
            promotion=promotion,
            promotion_ms=promotion_ms,
            mode="readout",
            messages=record.messages,
            answer_prefix=record.answer_prefix,
            child_context_parent=record.context_parent,
        )

    def _check_result_provenance(
        self, profile: WorkerHello, model_sha256: str, runtime_sha256: str
    ) -> None:
        if model_sha256 != profile.model_sha256:
            raise SnapshotProtocolError("worker model provenance differs")
        if runtime_sha256 != profile.runtime_sha256:
            raise SnapshotProtocolError("worker runtime provenance differs")

    async def _register_child(
        self,
        store: SnapshotStore,
        row: WorkerSnapshotRow,
        plan: BranchPlan,
        messages: list[Message],
        answer_prefix: str,
        owner: str,
        persistence: Persistence,
    ) -> None:
        await store.register(
            snapshot_id=row.snapshot_id,
            owner=owner,
            completed_blocks=30,
            boundary="readout",
            parent=plan.effective_parent,
            context_parent=plan.child_context_parent,
            messages=messages,
            answer_prefix=answer_prefix,
            tokens=row.tokens,
            host_bytes=_row_bytes(row),
            persistence=persistence,
            prompt_sha256=row.prompt_sha256,
            ttl_seconds=self._default_ttl,
        )

    # -- decisions ---------------------------------------------------------

    async def decide(
        self, request: V2DecisionRequest, *, owner: str
    ) -> V2DecisionResponse:
        if request.readout.completed_blocks != 30:
            raise CapabilityUnavailable("no registered early head for an 18 readout")
        async with self._serialized() as (profile, store):
            plan = await self._plan(store, request.snapshot)
            compiled: dict[str, CompiledSnapshotQuestion] = {}
            for question_id, question in request.questions.items():
                if plan.mode == "context":
                    compiled[question_id] = compile_context_followup(
                        plan.messages, question_id, question, profile.labels
                    )
                else:
                    compiled[question_id] = compile_readout_followup(
                        plan.messages,
                        plan.answer_prefix,
                        question_id,
                        question,
                        profile.labels,
                    )
            keep = request.save_result_snapshot
            saves = {
                question_id: (new_snapshot_id() if keep else None)
                for question_id in request.questions
            }
            self._pending_ids = [value for value in saves.values() if value is not None]
            with store.lease(plan.effective_parent):
                parent = await store.restore(plan.effective_parent)
                payloads = [
                    compiled[question_id].worker_payload(saves[question_id])
                    for question_id in request.questions
                ]
                result = await self._backend.evaluate(
                    snapshot_id=plan.effective_parent,
                    readout_blocks=30,
                    questions=payloads,
                    timeout=self._request_timeout,
                )
                self._check_result_provenance(
                    profile, result.model_sha256, result.runtime_sha256
                )
                if len(result.questions) != len(request.questions):
                    raise SnapshotProtocolError("worker question count differs")
                answers: dict[str, Answer] = {}
                suffix_tokens: dict[str, int] = {}
                child: dict[str, str] = {}
                lower = upper = processed = restored = 0
                restore_ms = inference_ms = 0.0
                for question_id, raw in zip(
                    request.questions, result.questions, strict=True
                ):
                    answers[question_id] = map_answer(
                        compiled[question_id], raw, profile.label_token_ids
                    )
                    if raw.snapshot.parent != plan.effective_parent:
                        raise SnapshotProtocolError("branch parent differs")
                    suffix_tokens[question_id] = raw.snapshot.suffix_tokens
                    lower += raw.snapshot.block_tokens.lower
                    upper += raw.snapshot.block_tokens.upper
                    processed += raw.processed_tokens
                    restored += raw.snapshot.restored_bytes
                    restore_ms += raw.snapshot.restore_ms
                    inference_ms += raw.snapshot.inference_ms
                    save_id = saves[question_id]
                    if save_id is not None:
                        row = raw.snapshot.child
                        if row is None or row.snapshot_id != save_id:
                            raise SnapshotProtocolError("worker did not save the child")
                        await self._register_child(
                            store,
                            row,
                            plan,
                            compiled[question_id].messages,
                            compiled[question_id].answer_prefix,
                            owner,
                            parent.persistence,
                        )
                        child[question_id] = save_id
                    elif raw.snapshot.child is not None:
                        raise SnapshotProtocolError("worker saved an unrequested child")
            usage = SnapshotUsage(
                requested_parent=plan.requested_parent,
                effective_parent=plan.effective_parent,
                promotion=plan.promotion,
                profile=SNAPSHOT_PROFILE,
                suffix_tokens=suffix_tokens,
                block_tokens=BlockTokens(lower=lower, upper=upper),
                reused_prefix_tokens=parent.tokens,
                restored_bytes=restored,
                timing=SnapshotTiming(
                    restore_ms=restore_ms,
                    promotion_ms=plan.promotion_ms,
                    inference_ms=inference_ms,
                ),
                child=child or None,
            )
            return V2DecisionResponse(
                model=MODEL_ID,  # type: ignore[arg-type]
                answers=answers,
                usage=Usage(input_tokens=processed),
                snapshot_usage=usage,
            )

    # -- state evaluation --------------------------------------------------

    async def state_eval(
        self, request: StateEvaluationRequest, *, owner: str
    ) -> StateEvaluationResponse:
        if request.readout.completed_blocks != 30:
            raise CapabilityUnavailable("no registered early head for an 18 readout")
        async with self._serialized() as (profile, store):
            plan = await self._plan(store, request.snapshot)
            if plan.mode == "context":
                branch = compile_context_prompt(plan.messages, request.prompt)
            else:
                branch = compile_readout_prompt(
                    plan.messages, plan.answer_prefix, request.prompt
                )
            save_id = new_snapshot_id() if request.save_result_snapshot else None
            self._pending_ids = [save_id] if save_id is not None else []
            with store.lease(plan.effective_parent):
                parent = await store.restore(plan.effective_parent)
                state = await self._backend.state_eval(
                    snapshot_id=plan.effective_parent,
                    messages=branch.messages,
                    answer_prefix=branch.answer_prefix,
                    export=request.readout.export,
                    top_logits=request.readout.top_logits,
                    save_as=save_id,
                    timeout=self._request_timeout,
                )
                vectors: dict[str, VectorArtifact] = {}
                for kind, artifact in state.vectors.items():
                    if kind not in request.readout.export:
                        raise SnapshotProtocolError("worker exported extra vector")
                    if artifact.representation != _REPRESENTATION[kind]:
                        raise SnapshotProtocolError("worker vector tag is wrong")
                    vectors[kind] = artifact
                child_id: str | None = None
                if save_id is not None:
                    row = state.child
                    if row is None or row.snapshot_id != save_id:
                        raise SnapshotProtocolError("worker did not save the child")
                    await self._register_child(
                        store,
                        row,
                        plan,
                        branch.messages,
                        branch.answer_prefix,
                        owner,
                        parent.persistence,
                    )
                    child_id = save_id
                elif state.child is not None:
                    raise SnapshotProtocolError("worker saved an unrequested child")
            return StateEvaluationResponse(
                model=MODEL_ID,  # type: ignore[arg-type]
                suffix_tokens=state.suffix_tokens,
                block_tokens=state.block_tokens,
                reused_prefix_tokens=parent.tokens,
                restored_bytes=state.restored_bytes,
                timing=SnapshotTiming(
                    restore_ms=state.restore_ms,
                    promotion_ms=plan.promotion_ms,
                    inference_ms=state.inference_ms,
                ),
                vectors=vectors,  # type: ignore[arg-type]
                top_logits=state.top_logits,
                child=child_id,
                usage=Usage(input_tokens=state.suffix_tokens),
            )

    # -- metadata, deletion, vectors --------------------------------------

    async def metadata(self, snapshot_id: str) -> SnapshotMetadata:
        async with self._serialized() as (_, store):
            return store.metadata(snapshot_id)

    async def delete(self, snapshot_id: str) -> int:
        async with self._serialized() as (_, store):
            return await store.delete(snapshot_id)

    async def vectors(
        self, snapshot_id: str, *, which: str, row_begin: int, row_end: int
    ) -> WorkerTensor:
        if which not in {"h18", "h30", "last_normalized"}:
            raise SnapshotRequestError("unknown vector name", reason="budget")
        if row_end <= row_begin or row_end - row_begin > 64:
            raise SnapshotRequestError("vector row range invalid", reason="budget")
        async with self._serialized() as (_, store):
            with store.lease(snapshot_id):
                await store.restore(snapshot_id)
                worker = await self._backend.vectors(
                    snapshot_id=snapshot_id,
                    which=which,
                    row_begin=row_begin,
                    row_end=row_end,
                    timeout=self._request_timeout,
                )
        # The worker stamps the representation tag; it is validated on parse.
        return worker.tensor

    def models(self) -> dict[str, object]:
        profile = self.profile
        if profile is None:
            return {"models": []}
        return {
            "models": [
                {
                    "id": profile.model_id,
                    "profile": profile.profile,
                    "prompt_version": profile.context_prompt_version,
                    "limits": {
                        "context_tokens": profile.context_size,
                        "checkpoints": [18, 30],
                        "questions": 32,
                        "vector_rows": 64,
                        "host_bytes": self._host_bytes,
                    },
                    "capabilities": {
                        "operations": ["continue", "promote", "inspect"],
                        "generation": False,
                        "early_head": "none",
                        "readout_blocks": [30],
                    },
                }
            ]
        }


__all__ = ["BranchPlan", "SnapshotBusyError", "SnapshotService"]
