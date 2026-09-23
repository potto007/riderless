"""Typed snapshot errors shared by the store, backend and service.

These are the vocabulary the routes map to HTTP status codes. Keeping them in
one module avoids a cycle between the store (which raises budget/not-found) and
the backend (which raises the same names when the worker does).
"""

from __future__ import annotations


class SnapshotError(Exception):
    """Base class for every typed snapshot failure."""


class SnapshotUnavailableError(SnapshotError):
    """The snapshot backend is not available for work."""


class SnapshotProtocolError(SnapshotError):
    """The worker violated its bounded protocol; the child is reaped."""


class SnapshotExecutionError(SnapshotError):
    """The worker failed after inference began; the child is reaped."""


class SnapshotRequestError(SnapshotError):
    """A request the worker rejected in preflight (budget or control tokens)."""

    def __init__(self, message: str, *, reason: str = "budget") -> None:
        super().__init__(message)
        self.reason = reason


class SnapshotNotFound(SnapshotError):
    """No snapshot with the referenced id (unknown or expired)."""


class SnapshotExists(SnapshotError):
    """A create tried to register an id that already exists."""


class PrefixMismatch(SnapshotError):
    """A branch's rendered prompt does not start with the frozen tokens."""


class FollowupUnsupported(SnapshotError):
    """The template cannot continue this readout snapshot."""


class CapabilityUnavailable(SnapshotError):
    """The requested operation is not available for this snapshot or profile."""


class IntegrityError(SnapshotError):
    """A persisted blob failed its checksum, size or profile check on load."""


class SnapshotStoreFull(SnapshotError):
    """The host byte budget cannot fit a snapshot even after eviction."""


class SnapshotInUse(SnapshotError):
    """A delete targeted a snapshot that is leased or still referenced."""
