"""Produce the llama.cpp base runtime the API worker links against.

The worker in `riderless/api/native` is deliberately not built against whatever
llama.cpp happens to be installed. It links one recorded revision, and both the
build manifest and the API startup check verify that the shared libraries on
disk still hash to what the manifest recorded. This script creates that base.

The default is the release this project is tested against, `TESTED_LLAMA_TAG`.
`--revision` builds any other tag or commit instead: the hash checks are the
same, only the published measurements stop applying, which the build says out
loud. Fetching the tested tag also checks that it still resolves to the
recorded commit, so a moved tag is a build failure rather than a silent swap.

The output directory is the `--base` argument of `riderless.api.native.build`
and contains exactly three things:

    <out>/headers/          pristine source snapshot of the built revision
    <out>/runtime/bin/      the shared libraries built from that snapshot
    <out>/build.json        revision plus a sha256 for every library

Nothing here is redistributed with this project: llama.cpp (MIT) is fetched
from its own repository at build time.

Example:

    python scripts/riderless/build_base_runtime.py --out build/llama-base \\
        --cuda --cuda-architectures 120

The compile is capped at four parallel jobs. Raising it on a small machine is
how an unattended build turns into an out-of-memory crash, so `--jobs` refuses
anything higher. A CUDA build compiles every kernel once per architecture, so
naming only the architectures you own is the other half of that cap.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

# The tested revision lives with the worker build, so the base runtime and the
# startup check can never disagree about which revision that is.
from riderless.api.native.build import (
    TESTED_LLAMA_REVISION,
    TESTED_LLAMA_TAG,
    is_tested_revision,
    sha256_helper,
)

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


def fetch_revision(destination: Path, repository: str, revision: str) -> str:
    """Fetch exactly one revision into a throwaway checkout; name it locally."""
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
            revision,
        ]
    )
    # A refspec-less fetch writes FETCH_HEAD and no local branch or tag.
    return "FETCH_HEAD"


def verify_revision(source: Path, revision: str) -> str:
    kind = subprocess.run(
        ["git", "-C", str(source), "cat-file", "-t", revision],
        capture_output=True,
        text=True,
        check=False,
    )
    if kind.returncode != 0 or kind.stdout.strip() not in {"commit", "tag"}:
        raise ValueError(
            f"{source} does not contain revision {revision}; "
            "fetch it or drop --llama-source"
        )
    return revision


def resolve_commit(source: Path, name: str) -> str:
    """The commit `name` points at, peeling an annotated tag on the way."""
    resolved = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "--verify", f"{name}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    commit = resolved.stdout.strip()
    if resolved.returncode != 0 or len(commit) != 40:
        raise ValueError(f"Could not resolve {name} to a commit in {source}")
    return commit


def check_tested_ref(revision: str, commit: str) -> bool:
    """Refuse a tested ref that moved; warn for anything else untested."""
    if revision not in {TESTED_LLAMA_TAG, TESTED_LLAMA_REVISION}:
        return is_tested_revision(commit)
    if commit != TESTED_LLAMA_REVISION:
        raise ValueError(
            f"{revision} resolves to {commit}, not the recorded tested commit "
            f"{TESTED_LLAMA_REVISION}; upstream moved the tag"
        )
    return True


def export_headers(source: Path, name: str, headers: Path) -> None:
    """Extract the commit's tree, not the working copy, so it cannot be dirty."""
    archive = subprocess.check_output(["git", "-C", str(source), "archive", name])
    headers.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(headers, filter="data")
    if not (headers / "include" / "llama.h").is_file():
        raise ValueError("Exported snapshot is missing include/llama.h")
    sha256_helper(headers)


def configure(
    headers: Path,
    runtime: Path,
    *,
    cuda: bool,
    cuda_architectures: str | None,
) -> list[str]:
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
    if cuda_architectures is not None:
        options.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}")
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
    revision: str,
    llama_source: Path | None,
    repository: str,
    cuda: bool,
    cuda_architectures: str | None,
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
        checkout = out / "llama.cpp-checkout"
        source = checkout
        name = fetch_revision(checkout, repository, revision)
    else:
        source = llama_source.resolve()
        name = verify_revision(source, revision)
    commit = resolve_commit(source, name)
    tested = check_tested_ref(revision, commit)

    headers = out / "headers"
    runtime = out / "runtime"
    export_headers(source, name, headers)
    options = configure(
        headers, runtime, cuda=cuda, cuda_architectures=cuda_architectures
    )
    compile_libraries(runtime, cuda=cuda, jobs=jobs)

    libraries = runtime / "bin"
    if not libraries.is_dir():
        raise ValueError(f"Expected shared libraries in {libraries}")
    checksums = inventory(libraries, cuda=cuda)

    if checkout is not None and not keep_checkout:
        shutil.rmtree(checkout)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        # What was asked for, and what it actually resolved to.
        "llama_ref": revision,
        "llama_revision": commit,
        "tested_llama_tag": TESTED_LLAMA_TAG,
        "tested_llama_revision": TESTED_LLAMA_REVISION,
        "tested_revision": tested,
        "llama_repository": repository if llama_source is None else str(source),
        "runtime_dir": str(libraries),
        "runtime_sha256": checksums,
        "cuda": cuda,
        "cuda_architectures": cuda_architectures,
        # Kept for readers that expect the older field name.
        "cpu_only": not cuda,
        "cmake_options": options,
    }
    (out / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
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
        "--revision",
        default=TESTED_LLAMA_TAG,
        help=f"llama.cpp tag or commit to build; default {TESTED_LLAMA_TAG}, the "
        "release this project is tested against. Any other value builds and "
        "verifies the same way but is reported as untested",
    )
    parser.add_argument(
        "--llama-source",
        type=Path,
        default=None,
        help="existing llama.cpp clone containing --revision; "
        "omitted means fetch that one revision from --repository",
    )
    parser.add_argument("--repository", default=UPSTREAM_REPOSITORY)
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="build the CUDA backend as well; needs a CUDA toolkit",
    )
    parser.add_argument(
        "--cuda-architectures",
        default=None,
        help="value for CMAKE_CUDA_ARCHITECTURES, for example 120 for an "
        "RTX 5090. Omitted leaves llama.cpp's own default, which is the "
        "locally detected architecture with GGML_NATIVE and otherwise a broad "
        "set covering every commonly used GPU. Naming only the architectures "
        "you own is what keeps a CUDA build's memory and time bounded",
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
            revision=args.revision,
            llama_source=args.llama_source,
            repository=args.repository,
            cuda=args.cuda,
            cuda_architectures=args.cuda_architectures,
            jobs=args.jobs,
            keep_checkout=args.keep_checkout,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
