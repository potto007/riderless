"""Build the full-only API worker against a llama.cpp base runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

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

LOGGER = logging.getLogger("riderless.api.native.build")
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


def build(base: Path, output: Path) -> Row:
    base = base.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError("Build output exists; choose a new directory")
    base_manifest, headers, libraries = validate_base(base)
    source = Path(__file__).parent.resolve()
    build_dir = output / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build_dir),
            f"-DLLAMA_SOURCE={headers}",
            f"-DLLAMA_BUILD={base / 'runtime'}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "-j", "4"],
        check=True,
    )
    subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "--output-on-failure"],
        check=True,
    )
    executable = build_dir / "riderless-worker"
    source_files = [
        source / "CMakeLists.txt",
        source / "build.py",
        source / "worker.cpp",
        source / "worker-utils.h",
        source / "worker-utils-test.cpp",
    ]
    runtime_checksums = dict(base_manifest["runtime_sha256"])
    sha_source = sha256_helper(headers)
    sha_header = sha_source.with_suffix(".h")
    revision = str(base_manifest["llama_revision"])
    manifest: Row = {
        "schema_version": 1,
        # Provenance of the base this worker is linked against. `llama_ref` is
        # what was asked for, `llama_revision` is what that resolved to.
        "llama_revision": revision,
        "llama_ref": str(base_manifest.get("llama_ref", revision)),
        "tested_llama_tag": TESTED_LLAMA_TAG,
        "tested_llama_revision": TESTED_LLAMA_REVISION,
        "tested_revision": revision == TESTED_LLAMA_REVISION,
        "base_build": str(base),
        "base_build_json_sha256": digest(base / "build.json"),
        "runtime_dir": str(libraries),
        "runtime_sha256": runtime_checksums,
        "runtime_bundle_sha256": bundle_digest(runtime_checksums),
        "source_sha256": {path.name: digest(path) for path in source_files},
        "helper_sha256": {
            "sha256.c": digest(sha_source),
            "sha256.h": digest(sha_header),
        },
        "executable": str(executable),
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
    (output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
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
