"""Fetching a worker bundle: flavor choice, checksums, attestation, unpacking.

No network. `download` is replaced by a copy out of a directory that stands in
for the release's asset list, so every path through the verification runs
against real bytes on disk.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from unridden.api.native import fetch as fetch_module
from unridden.api.native.bundle import (
    MANIFEST_NAME,
    MANIFEST_SCHEMA_VERSION,
    RUNTIME_RELATIVE,
    SNAPSHOT_WORKER_RELATIVE,
    WORKER_RELATIVE,
    asset_name,
    pack,
)
from unridden.api.native.fetch import (
    CHECKSUM_FILE,
    DEFAULT_OUTPUTS,
    FetchError,
    detect_flavor,
    fetch,
    parse_checksums,
    release_tag,
)

VERSION = "0.2.0"
TAG = f"v{VERSION}"
REVISION = "b29c606e28a01b1bc8c1351026a0fa6e616bf6c4"


def _release(tmp_path: Path, *, corrupt: bool = False) -> Path:
    """A directory of files named exactly as the release publishes them."""
    source = tmp_path / "built"
    worker = source / WORKER_RELATIVE
    worker.parent.mkdir(parents=True)
    worker.write_bytes(b"#!/bin/false\n")
    runtime = source / RUNTIME_RELATIVE
    runtime.mkdir()
    (runtime / "libggml-base.so.0.24.0").write_bytes(b"base")
    (source / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "llama_revision": REVISION,
                "executable": str(WORKER_RELATIVE),
                "runtime_dir": str(RUNTIME_RELATIVE),
            }
        )
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    lines = []
    for flavor in ("cuda13", "cuda12", "cpu"):
        name = asset_name(VERSION, flavor)
        pack(source, assets / name)
        recorded = hashlib.sha256((assets / name).read_bytes()).hexdigest()
        if corrupt and flavor == "cuda13":
            recorded = "0" * 64
        lines.append(f"{recorded}  {name}")
    (assets / CHECKSUM_FILE).write_text("\n".join(lines) + "\n")
    return assets


@pytest.fixture
def offline_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    assets = _release(tmp_path)
    _serve(assets, monkeypatch)
    return assets


def _serve(assets: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def copy(url: str, destination: Path) -> Path:
        source = assets / url.rsplit("/", 1)[-1]
        if not source.is_file():
            raise FetchError(f"{url} returned HTTP 404")
        shutil.copyfile(source, destination)
        return destination

    monkeypatch.setattr(fetch_module, "download", copy)


def _no_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)


def _gh(monkeypatch: pytest.MonkeyPatch, returncode: int) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    def verify(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, returncode, "", "no attestation")

    monkeypatch.setattr(subprocess, "run", verify)


def test_a_verified_bundle_lands_where_the_api_looks(
    tmp_path: Path, offline_release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_gh(monkeypatch)
    output = tmp_path / "build" / "api-worker"

    result = fetch(
        output=output,
        flavor="cuda13",
        requested_version=VERSION,
        repository="potto007/unridden",
        require_attestation=False,
    )

    assert (output / WORKER_RELATIVE).is_file()
    assert (output / MANIFEST_NAME).is_file()
    assert result["tag"] == TAG
    assert result["asset"] == asset_name(VERSION, "cuda13")
    assert result["llama_revision"] == REVISION
    assert result["tested_revision"] is True
    assert result["attested"] is False


def test_a_checksum_that_does_not_match_installs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(_release(tmp_path, corrupt=True), monkeypatch)
    _no_gh(monkeypatch)
    output = tmp_path / "api-worker"

    with pytest.raises(FetchError, match="hashes to"):
        fetch(
            output=output,
            flavor="cuda13",
            requested_version=VERSION,
            repository="potto007/unridden",
            require_attestation=False,
        )

    assert not output.exists()


def test_a_flavor_the_release_does_not_publish_is_named(
    tmp_path: Path, offline_release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_gh(monkeypatch)
    (offline_release / CHECKSUM_FILE).write_text(
        f"{'1' * 64}  {asset_name(VERSION, 'cpu')}\n"
    )

    with pytest.raises(FetchError, match="does not list"):
        fetch(
            output=tmp_path / "api-worker",
            flavor="cuda13",
            requested_version=VERSION,
            repository="potto007/unridden",
            require_attestation=False,
        )


def test_a_missing_gh_warns_but_requiring_attestation_refuses(
    tmp_path: Path, offline_release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_gh(monkeypatch)

    with pytest.raises(FetchError, match="gh is not installed"):
        fetch(
            output=tmp_path / "strict",
            flavor="cpu",
            requested_version=VERSION,
            repository="potto007/unridden",
            require_attestation=True,
        )
    assert not (tmp_path / "strict").exists()

    result = fetch(
        output=tmp_path / "lenient",
        flavor="cpu",
        requested_version=VERSION,
        repository="potto007/unridden",
        require_attestation=False,
    )
    assert result["attested"] is False


def test_a_failing_attestation_is_a_warning_unless_it_was_required(
    tmp_path: Path, offline_release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _gh(monkeypatch, returncode=1)

    with pytest.raises(FetchError, match="attestation verify failed"):
        fetch(
            output=tmp_path / "strict",
            flavor="cpu",
            requested_version=VERSION,
            repository="potto007/unridden",
            require_attestation=True,
        )

    _gh(monkeypatch, returncode=0)
    result = fetch(
        output=tmp_path / "attested",
        flavor="cpu",
        requested_version=VERSION,
        repository="potto007/unridden",
        require_attestation=False,
    )
    assert result["attested"] is True


def test_an_existing_output_is_never_written_over(
    tmp_path: Path, offline_release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_gh(monkeypatch)
    taken = tmp_path / "api-worker"
    taken.mkdir()

    with pytest.raises(FetchError, match="exists"):
        fetch(
            output=taken,
            flavor="cpu",
            requested_version=VERSION,
            repository="potto007/unridden",
            require_attestation=False,
        )


@pytest.mark.parametrize(
    ("driver", "expected"),
    [
        (None, "cpu"),
        ("581.15", "cuda13"),
        ("580.65.06", "cuda13"),
        ("580.64.99", "cuda12"),
        ("550.144.03", "cuda12"),
        ("525.60.13", "cuda12"),
        ("470.256.02", "cpu"),
    ],
)
def test_the_flavor_follows_the_installed_driver(
    monkeypatch: pytest.MonkeyPatch, driver: str | None, expected: str
) -> None:
    monkeypatch.setattr(
        fetch_module,
        "_driver_version",
        lambda: (
            None if driver is None else tuple(int(part) for part in driver.split("."))
        ),
    )

    flavor, reason = detect_flavor()

    assert flavor == expected
    assert reason


def test_the_release_tag_defaults_to_the_installed_version() -> None:
    assert release_tag("0.2.0") == "v0.2.0"
    assert release_tag("v0.2.0") == "v0.2.0"
    assert release_tag(None).startswith("v")


def test_checksum_lines_are_read_the_way_sha256sum_writes_them() -> None:
    parsed = parse_checksums(
        f"{'a' * 64}  first.tar.gz\n{'B' * 64} *second.tar.gz\njunk\n"
    )

    assert parsed == {"first.tar.gz": "a" * 64, "second.tar.gz": "b" * 64}
    with pytest.raises(FetchError, match="no checksums"):
        parse_checksums("nothing here\n")


def test_a_snapshot_bundle_is_fetched_by_kind(
    tmp_path: Path, offline_release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_gh(monkeypatch)
    source = tmp_path / "snapshot-built"
    worker = source / SNAPSHOT_WORKER_RELATIVE
    worker.parent.mkdir(parents=True)
    worker.write_bytes(b"#!/bin/false\n")
    (source / RUNTIME_RELATIVE).mkdir()
    (source / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "llama_revision": REVISION,
                "executable": str(SNAPSHOT_WORKER_RELATIVE),
                "runtime_dir": str(RUNTIME_RELATIVE),
            }
        )
    )
    name = asset_name(VERSION, "cuda13", "snapshot-worker")
    pack(source, offline_release / name)
    digest = hashlib.sha256((offline_release / name).read_bytes()).hexdigest()
    with (offline_release / CHECKSUM_FILE).open("a") as sums:
        sums.write(f"{digest}  {name}\n")
    output = tmp_path / "build" / "api-snapshot-worker"

    result = fetch(
        output=output,
        kind="snapshot-worker",
        flavor="cuda13",
        requested_version=VERSION,
        repository="potto007/unridden",
        require_attestation=False,
    )

    assert (output / SNAPSHOT_WORKER_RELATIVE).is_file()
    assert result["kind"] == "snapshot-worker"
    assert result["asset"] == name


def test_the_default_outputs_are_where_apiconfig_looks() -> None:
    from unridden.api.app import ApiConfig

    defaults = ApiConfig()
    assert defaults.manifest_path == DEFAULT_OUTPUTS["worker"] / MANIFEST_NAME
    assert defaults.snapshot_manifest_path == (
        DEFAULT_OUTPUTS["snapshot-worker"] / MANIFEST_NAME
    )
    assert defaults.snapshot_worker_path == (
        DEFAULT_OUTPUTS["snapshot-worker"] / SNAPSHOT_WORKER_RELATIVE
    )
