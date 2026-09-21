"""Produce the pinned llama.cpp base runtime the API worker links against.

The worker in `systemone/api/native` is deliberately not built against whatever
llama.cpp happens to be installed. It links one frozen revision, and both the
build manifest and the API startup check verify that the shared libraries on
disk still hash to what the manifest recorded. This script creates that base.

The output directory is the `--base` argument of `systemone.api.native.build`
and contains exactly three things:

    <out>/headers/          pristine source snapshot of the frozen revision
    <out>/runtime/bin/      the shared libraries built from that snapshot
    <out>/build.json        revision plus a sha256 for every library

Nothing here is redistributed with this project: llama.cpp (MIT) is fetched
from its own repository at build time.

Example:

    python scripts/systemone/build_base_runtime.py --out build/llama-base --cuda

The compile is capped at four parallel jobs. Raising it on a small machine is
how an unattended build turns into an out-of-memory crash, so `--jobs` refuses
anything higher.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

# The one revision this project builds against. Changing it invalidates every
# recorded runtime hash, so treat it as a version bump, not a config knob.
FROZEN_LLAMA_REVISION = "afeebe103bd99cda8f5dfaefcabadf890db7fda7"
UPSTREAM_REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"
MAX_JOBS = 4

# `llama-common` pulls in the rest; the others are listed so a missing one is a
# build failure here rather than a link failure in the worker.
CPU_LIBRARIES = ("llama", "llama-common", "ggml", "ggml-base", "ggml-cpu")
CUDA_LIBRARY = "ggml-cuda"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def fetch_revision(destination: Path, repository: str) -> Path:
    """Fetch exactly the frozen commit into a throwaway checkout."""
    destination.mkdir(parents=True)
    run(["git", "-C", str(destination), "init", "--quiet"])
    run(["git", "-C", str(destination), "remote", "add", "origin", repository])
    run(
        [
            "git",
            "-C",
            str(destination),
            "fetch",
            "--quiet",
            "--depth",
            "1",
            "origin",
            FROZEN_LLAMA_REVISION,
        ]
    )
    return destination


def verify_revision(source: Path) -> None:
    kind = subprocess.run(
        ["git", "-C", str(source), "cat-file", "-t", FROZEN_LLAMA_REVISION],
        capture_output=True,
        text=True,
        check=False,
    )
    if kind.returncode != 0 or kind.stdout.strip() != "commit":
        raise ValueError(
            f"{source} does not contain the frozen revision "
            f"{FROZEN_LLAMA_REVISION}; fetch it or drop --llama-source"
        )


def export_headers(source: Path, headers: Path) -> None:
    """Extract the commit's tree, not the working copy, so it cannot be dirty."""
    archive = subprocess.check_output(
        ["git", "-C", str(source), "archive", FROZEN_LLAMA_REVISION]
    )
    headers.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(headers, filter="data")
    if not (headers / "include" / "llama.h").is_file():
        raise ValueError("Exported snapshot is missing include/llama.h")
    sha_helper = headers / "examples/gguf-hash/deps/sha256/sha256.c"
    if not sha_helper.is_file():
        raise ValueError("Exported snapshot is missing the gguf-hash sha256 helper")


def configure(headers: Path, runtime: Path, *, cuda: bool) -> list[str]:
    options = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        "-DLLAMA_BUILD_COMMON=ON",
        "-DLLAMA_BUILD_TOOLS=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_BUILD_APP=OFF",
        "-DLLAMA_BUILD_UI=OFF",
        f"-DGGML_CUDA={'ON' if cuda else 'OFF'}",
    ]
    run(["cmake", "-S", str(headers), "-B", str(runtime), *options])
    return options


def compile_libraries(runtime: Path, *, cuda: bool, jobs: int) -> None:
    targets = [*CPU_LIBRARIES]
    if cuda:
        targets.append(CUDA_LIBRARY)
    run(["cmake", "--build", str(runtime), "--target", *targets, "-j", str(jobs)])


def inventory(libraries: Path, *, cuda: bool) -> dict[str, str]:
    """Hash the real library file behind each unversioned `lib<name>.so` link."""
    names = [*CPU_LIBRARIES]
    if cuda:
        names.append(CUDA_LIBRARY)
    checksums: dict[str, str] = {}
    for name in names:
        link = libraries / f"lib{name}.so"
        if not link.exists():
            raise FileNotFoundError(link)
        resolved = link.resolve()
        checksums[resolved.name] = digest(resolved)
    return checksums


def build(
    out: Path,
    *,
    llama_source: Path | None,
    repository: str,
    cuda: bool,
    jobs: int,
    keep_checkout: bool,
) -> dict[str, Any]:
    if out.exists():
        raise ValueError("Output directory exists; choose a new one")
    if not 1 <= jobs <= MAX_JOBS:
        raise ValueError(f"--jobs must be between 1 and {MAX_JOBS}")
    out.mkdir(parents=True)
    checkout: Path | None = None
    if llama_source is None:
        checkout = fetch_revision(out / "llama.cpp-checkout", repository)
        source = checkout
    else:
        source = llama_source.resolve()
        verify_revision(source)

    headers = out / "headers"
    runtime = out / "runtime"
    export_headers(source, headers)
    options = configure(headers, runtime, cuda=cuda)
    compile_libraries(runtime, cuda=cuda, jobs=jobs)

    libraries = runtime / "bin"
    if not libraries.is_dir():
        raise ValueError(f"Expected shared libraries in {libraries}")
    checksums = inventory(libraries, cuda=cuda)

    if checkout is not None and not keep_checkout:
        shutil.rmtree(checkout)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "llama_revision": FROZEN_LLAMA_REVISION,
        "llama_repository": repository if llama_source is None else str(source),
        "runtime_dir": str(libraries),
        "runtime_sha256": checksums,
        "cuda": cuda,
        # Kept for readers that expect the older field name.
        "cpu_only": not cuda,
        "cmake_options": options,
    }
    (out / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="new directory to create; becomes --base of the worker build",
    )
    parser.add_argument(
        "--llama-source",
        type=Path,
        default=None,
        help="existing llama.cpp clone containing the frozen revision; "
        "omitted means fetch that one commit from --repository",
    )
    parser.add_argument("--repository", default=UPSTREAM_REPOSITORY)
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="build the CUDA backend as well; needs a CUDA toolkit",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=MAX_JOBS,
        help=f"parallel compile jobs, 1 to {MAX_JOBS}",
    )
    parser.add_argument(
        "--keep-checkout",
        action="store_true",
        help="keep the fetched clone instead of deleting it after export",
    )
    args = parser.parse_args()
    try:
        manifest = build(
            args.out.resolve(),
            llama_source=args.llama_source,
            repository=args.repository,
            cuda=args.cuda,
            jobs=args.jobs,
            keep_checkout=args.keep_checkout,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
