"""Backend interface and native transport errors."""

from __future__ import annotations

from typing import Protocol

from systemone.api.compiler import CompiledBatch
from systemone.api.schema import BackendProfile, WorkerBatchResult


class BackendUnavailableError(RuntimeError):
    """The owned native child is not available for work."""


class BackendProtocolError(RuntimeError):
    """The owned native child violated its bounded protocol."""


class BackendRequestError(ValueError):
    """The native worker rejected a preflighted request."""

    def __init__(self, message: str, *, reason: str = "budget") -> None:
        super().__init__(message)
        self.reason = reason


class BackendExecutionError(RuntimeError):
    """The native worker failed after request execution began."""


class Backend(Protocol):
    """One resident model backend.

    Contract: when `evaluate` times out, is cancelled, or sees any ambiguous
    transport failure, the implementation must stop the in-flight inference and
    set `ready` to False before the exception leaves. The service layer relies on
    this and does not invalidate a backend itself.
    """

    profile: BackendProfile | None
    ready: bool

    async def start(self) -> BackendProfile: ...

    async def close(self) -> None: ...

    async def evaluate(
        self, batch: CompiledBatch, *, timeout: float
    ) -> WorkerBatchResult: ...
