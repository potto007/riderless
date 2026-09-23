"""Build the full-only API worker against a llama.cpp base runtime.

The output is a self-contained bundle (`unridden.api.native.bundle`): the
executable, a copy of the shared libraries it links, and a manifest that names
both relative to itself. That is the same shape a published release asset
unpacks to, so a build here and a `worker fetch` are interchangeable installs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from unridden.api.native.bundle import (
    MANIFEST_NAME,
    MANIFEST_SCHEMA_VERSION,
    RUNTIME_RELATIVE,
    WORKER_RELATIVE,
)

# The llama.cpp release this project is tested against. The tag is what a reader
# fetches; the commit it resolved to is what makes a moved tag detectable. Other
# revisions are allowed and only warned about: the build and the startup check
# still verify every hash, so integrity never depends on which revision it is.
TESTED_LLAMA_TAG = "v0.4.1"
TESTED_LLAMA_REVISION = "b29c606e28a01b1bc8c1351026a0fa6e616bf6c4"

# The bundled sha256 helper moved from the gguf-hash example into `vendor` in
# llama.cpp v0.4.1. The file itself is unchanged, so both layouts are accepted.
SHA256_HELPER_DIRECTORIES = (
    "vendor/hash/sha256",
    "examples/gguf-hash/deps/sha256",
)

LOGGER = logging.getLogger("unridden.api.native.build")
UNTESTED_REVISION_WARNING = (
    "llama.cpp revision %s is not the tested revision %s (%s); the published "
    "measurements were taken on the tested revision and may not reproduce"
)
Row = dict[str, Any]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def bundle_digest(checksums: dict[str, str]) -> str:
    encoded = json.dumps(checksums, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def is_tested_revision(revision: str) -> bool:
    """True for the tested revision; otherwise warn and return False."""
    if revision == TESTED_LLAMA_REVISION:
        return True
    LOGGER.warning(
        UNTESTED_REVISION_WARNING,
        revision,
        TESTED_LLAMA_REVISION,
        TESTED_LLAMA_TAG,
    )
    return False


def sha256_helper(headers: Path) -> Path:
    for relative in SHA256_HELPER_DIRECTORIES:
        candidate = headers / relative / "sha256.c"
        if candidate.is_file():
            return candidate
    raise ValueError("Base runtime snapshot has no bundled sha256 helper")


def validate_base(base: Path) -> tuple[Row, Path, Path]:
    manifest_path = base / "build.json"
    manifest = json.loads(manifest_path.read_text())
    revision = manifest.get("llama_revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("Base runtime manifest does not name a llama revision")
    headers = base / "headers"
    libraries = base / "runtime" / "bin"
    if not headers.is_dir() or not libraries.is_dir():
        raise ValueError("Incomplete base runtime")
    checksums = manifest.get("runtime_sha256")
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("Base runtime lacks checksums")
    for name, checksum in checksums.items():
        if not isinstance(name, str) or not isinstance(checksum, str):
            raise TypeError("Invalid base runtime checksum manifest")
        if digest(libraries / name) != checksum:
            raise ValueError(f"Base runtime changed: {name}")
    is_tested_revision(revision)
    return manifest, headers, libraries


def stage_runtime(
    libraries: Path, destination: Path, checksums: dict[str, str]
) -> None:
    """Copy the base runtime into the bundle and re-hash what was copied.

    The unversioned `libfoo.so` symlinks come along: `ggml_backend_load_best`
    finds a backend through them, and the worker's DT_NEEDED entries name the
    `libfoo.so.N` links. Hashing after the copy means the manifest vouches for
    the files that ship, not for the ones they were copied from.
    """
    shutil.copytree(libraries, destination, symlinks=True)
    for name, checksum in checksums.items():
        copied = destination / name
        if not copied.is_file():
            raise ValueError(f"Runtime copy is missing {name}")
        if digest(copied) != checksum:
            raise ValueError(f"Runtime copy differs for {name}")


def build(base: Path, output: Path) -> Row:
    base = base.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError("Build output exists; choose a new directory")
    base_manifest, headers, libraries = validate_base(base)
    source = Path(__file__).parent.resolve()
    # CMake's scratch tree records absolute paths and object files, so it is
    # kept out of the bundle and removed once the binary has been copied in.
    # A failed build raises before that and leaves it in place to look at.
    cmake_dir = output / "cmake"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(cmake_dir),
            f"-DLLAMA_SOURCE={headers}",
            f"-DLLAMA_BUILD={base / 'runtime'}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(cmake_dir), "-j", "4"],
        check=True,
    )
    subprocess.run(
        ["ctest", "--test-dir", str(cmake_dir), "--output-on-failure"],
        check=True,
    )
    runtime_checksums = dict(base_manifest["runtime_sha256"])
    stage_runtime(libraries, output / RUNTIME_RELATIVE, runtime_checksums)
    executable = output / WORKER_RELATIVE
    executable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cmake_dir / "unridden-worker", executable)
    shutil.rmtree(cmake_dir)
    source_files = [
        source / "CMakeLists.txt",
        source / "build.py",
        source / "worker.cpp",
        source / "worker-utils.h",
        source / "worker-utils-test.cpp",
    ]
    sha_source = sha256_helper(headers)
    sha_header = sha_source.with_suffix(".h")
    revision = str(base_manifest["llama_revision"])
    manifest: Row = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        # Provenance of the base this worker is linked against. `llama_ref` is
        # what was asked for, `llama_revision` is what that resolved to.
        "llama_revision": revision,
        "llama_ref": str(base_manifest.get("llama_ref", revision)),
        "tested_llama_tag": TESTED_LLAMA_TAG,
        "tested_llama_revision": TESTED_LLAMA_REVISION,
        "tested_revision": revision == TESTED_LLAMA_REVISION,
        # The base build's own path is deliberately absent: it names the
        # machine that built this, and a bundle must carry no such path. Its
        # manifest hash is the provenance that survives relocation.
        "base_build_json_sha256": digest(base / "build.json"),
        "cuda": bool(base_manifest.get("cuda", False)),
        "cuda_architectures": base_manifest.get("cuda_architectures"),
        "cuda_redistributables": base_manifest.get("cuda_redistributables", []),
        # Both paths are relative to this manifest, which is what lets the
        # bundle be unpacked anywhere and still verify.
        "executable": str(WORKER_RELATIVE),
        "runtime_dir": str(RUNTIME_RELATIVE),
        "runtime_sha256": runtime_checksums,
        "runtime_bundle_sha256": bundle_digest(runtime_checksums),
        "source_sha256": {path.name: digest(path) for path in source_files},
        "helper_sha256": {
            "sha256.c": digest(sha_source),
            "sha256.h": digest(sha_header),
        },
        "executable_sha256": digest(executable),
        "runtime_config": {
            "context": 2048,
            "batch": 256,
            "ubatch": 256,
            "threads": 8,
            "attention": "causal",
            "swa_full": True,
            "cuda_fusion": True,
            "cuda_graphs": True,
        },
        "callbacks_enabled": False,
        "generated_tokens": 0,
        "execution_mode": "full",
    }
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = build(args.base, args.output)
    except (TypeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
