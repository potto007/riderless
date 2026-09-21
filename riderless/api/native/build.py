"""Build the full-only API worker against the pinned llama.cpp base runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

FROZEN_LLAMA_REVISION = "afeebe103bd99cda8f5dfaefcabadf890db7fda7"
Row = dict[str, Any]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def bundle_digest(checksums: dict[str, str]) -> str:
    encoded = json.dumps(checksums, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_base(base: Path) -> tuple[Row, Path, Path]:
    manifest_path = base / "build.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("llama_revision") != FROZEN_LLAMA_REVISION:
        raise ValueError("Base runtime revision differs from the frozen revision")
    headers = base / "headers"
    libraries = base / "runtime" / "bin"
    if not headers.is_dir() or not libraries.is_dir():
        raise ValueError("Incomplete frozen base runtime")
    checksums = manifest.get("runtime_sha256")
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("Base runtime lacks checksums")
    for name, checksum in checksums.items():
        if not isinstance(name, str) or not isinstance(checksum, str):
            raise TypeError("Invalid base runtime checksum manifest")
        if digest(libraries / name) != checksum:
            raise ValueError(f"Base runtime changed: {name}")
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
    sha_source = headers / "examples" / "gguf-hash" / "deps" / "sha256" / "sha256.c"
    sha_header = sha_source.with_suffix(".h")
    manifest: Row = {
        "schema_version": 1,
        "llama_revision": FROZEN_LLAMA_REVISION,
        "base_build": str(base),
        "base_build_json_sha256": digest(base / "build.json"),
        "runtime_dir": str(libraries),
        "runtime_sha256": runtime_checksums,
        "runtime_bundle_sha256": bundle_digest(runtime_checksums),
        "source_sha256": {path.name: digest(path) for path in source_files},
        "frozen_helper_sha256": {
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
