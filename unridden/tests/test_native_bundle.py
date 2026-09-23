"""The bundle's shape: what a tarball holds and where a manifest points."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from unridden.api.app import ApiConfig
from unridden.api.native.bundle import (
    MANIFEST_NAME,
    MANIFEST_SCHEMA_VERSION,
    RUNTIME_RELATIVE,
    WORKER_RELATIVE,
    asset_name,
    pack,
    resolve_manifest_paths,
    unpack,
)


def _bundle(directory: Path, **overrides: Any) -> Path:
    worker = directory / WORKER_RELATIVE
    worker.parent.mkdir(parents=True)
    worker.write_bytes(b"#!/bin/false\n")
    worker.chmod(0o755)
    runtime = directory / RUNTIME_RELATIVE
    runtime.mkdir()
    (runtime / "libggml-cuda.so.0.24.0").write_bytes(b"backend")
    # The unversioned link is how `ggml_backend_load_best` finds a backend, so
    # a bundle that loses it loads no GPU support at all.
    (runtime / "libggml-cuda.so").symlink_to("libggml-cuda.so.0.24.0")
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "llama_revision": "0" * 40,
        "executable": str(WORKER_RELATIVE),
        "runtime_dir": str(RUNTIME_RELATIVE),
    }
    manifest.update(overrides)
    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2))
    return path


def test_relative_members_resolve_against_the_manifest(tmp_path: Path) -> None:
    directory = tmp_path / "api-worker"
    directory.mkdir()
    manifest_path = _bundle(directory)

    executable, runtime = resolve_manifest_paths(
        json.loads(manifest_path.read_text()), manifest_path
    )

    assert executable == directory / WORKER_RELATIVE
    assert runtime == directory / RUNTIME_RELATIVE


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("/somewhere/unridden-worker", "relative"),
        ("../outside/unridden-worker", "escapes"),
        ("", "does not name"),
    ],
)
def test_a_relocatable_manifest_may_not_point_outside_itself(
    tmp_path: Path, value: str, message: str
) -> None:
    directory = tmp_path / "api-worker"
    directory.mkdir()
    manifest_path = _bundle(directory, executable=value)

    with pytest.raises(ValueError, match=message):
        resolve_manifest_paths(json.loads(manifest_path.read_text()), manifest_path)


def test_a_schema_1_manifest_keeps_its_absolute_paths(tmp_path: Path) -> None:
    directory = tmp_path / "api-worker"
    directory.mkdir()
    manifest_path = _bundle(
        directory,
        schema_version=1,
        executable=str(directory / WORKER_RELATIVE),
        runtime_dir=str(directory / RUNTIME_RELATIVE),
    )

    executable, runtime = resolve_manifest_paths(
        json.loads(manifest_path.read_text()), manifest_path
    )

    assert executable == directory / WORKER_RELATIVE
    assert runtime == directory / RUNTIME_RELATIVE


def test_an_unknown_schema_version_is_named_in_the_error(tmp_path: Path) -> None:
    directory = tmp_path / "api-worker"
    directory.mkdir()
    manifest_path = _bundle(directory, schema_version=99)

    with pytest.raises(ValueError, match="schema_version 99"):
        resolve_manifest_paths(json.loads(manifest_path.read_text()), manifest_path)


def test_the_default_layout_is_what_apiconfig_already_looks_for() -> None:
    defaults = ApiConfig()

    assert defaults.worker_path == Path("build/api-worker") / WORKER_RELATIVE
    assert defaults.manifest_path == Path("build/api-worker") / MANIFEST_NAME


def test_a_packed_bundle_unpacks_to_the_same_layout(tmp_path: Path) -> None:
    directory = tmp_path / "api-worker"
    directory.mkdir()
    _bundle(directory)

    archive = pack(directory, tmp_path / "worker.tar.gz")
    with tarfile.open(archive) as tar:
        names = sorted(tar.getnames())
        link = tar.getmember(str(RUNTIME_RELATIVE / "libggml-cuda.so"))
    restored = unpack(archive, tmp_path / "elsewhere")

    # No leading component: extracting into build/api-worker is the install.
    assert not any(name.startswith(("/", "..")) for name in names)
    assert str(WORKER_RELATIVE) in names
    assert MANIFEST_NAME in names
    assert link.issym() and link.linkname == "libggml-cuda.so.0.24.0"
    assert (restored / WORKER_RELATIVE).is_file()
    assert (restored / RUNTIME_RELATIVE / "libggml-cuda.so").is_symlink()
    assert (restored / MANIFEST_NAME).read_text() == (
        directory / MANIFEST_NAME
    ).read_text()


def test_packing_refuses_anything_that_is_not_the_bundle_layout(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "api-worker"
    directory.mkdir()
    _bundle(directory, executable="somewhere-else/unridden-worker")

    with pytest.raises(ValueError, match="executable"):
        pack(directory, tmp_path / "worker.tar.gz")


def test_packing_refuses_an_absolute_schema_1_manifest(tmp_path: Path) -> None:
    directory = tmp_path / "api-worker"
    directory.mkdir()
    _bundle(
        directory,
        schema_version=1,
        executable=str(directory / WORKER_RELATIVE),
        runtime_dir=str(directory / RUNTIME_RELATIVE),
    )

    with pytest.raises(ValueError, match="relocatable"):
        pack(directory, tmp_path / "worker.tar.gz")


def test_unpacking_never_overwrites_an_existing_install(tmp_path: Path) -> None:
    directory = tmp_path / "api-worker"
    directory.mkdir()
    _bundle(directory)
    archive = pack(directory, tmp_path / "worker.tar.gz")
    (tmp_path / "taken").mkdir()

    with pytest.raises(ValueError, match="exists"):
        unpack(archive, tmp_path / "taken")


def test_asset_names_carry_the_version_platform_and_flavor() -> None:
    assert (
        asset_name("0.2.0", "cuda13") == "unridden-worker-0.2.0-linux-x64-cuda13.tar.gz"
    )
    with pytest.raises(ValueError, match="flavor"):
        asset_name("0.2.0", "cuda11")
