"""The on-disk shape of a worker install, and how a manifest names its files.

A worker install is one directory, and everything the API needs is inside it:

    <dir>/build.json                the build manifest
    <dir>/build/riderless-worker    the executable
    <dir>/runtime/*.so*             the shared libraries it links and dlopens

That directory is the unit that moves. A local build writes it, `pack` turns it
into a tarball, `unpack` puts it back somewhere else, and nothing in it names
the machine that produced it: schema 2 manifests record `executable` and
`runtime_dir` relative to the manifest, so the same bytes verify wherever they
land. `ApiConfig`'s defaults point at `build/api-worker`, so a bundle unpacked
there needs no flags at all.

Schema 1 manifests recorded absolute paths and are still read, unchanged, so an
install built before 0.2.0 keeps starting. They are simply not relocatable.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

# Bumped from 1 when `executable` and `runtime_dir` became bundle-relative.
MANIFEST_SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)

MANIFEST_NAME = "build.json"
WORKER_RELATIVE = Path("build/riderless-worker")
RUNTIME_RELATIVE = Path("runtime")
# Exactly what a bundle contains; `pack` refuses to ship anything else, so
# CMake scratch or a stray log can never ride along into a release asset.
BUNDLE_MEMBERS = (Path(MANIFEST_NAME), WORKER_RELATIVE.parent, RUNTIME_RELATIVE)

FLAVORS = ("cuda13", "cuda12", "cpu")
PLATFORM = "linux-x64"


def asset_name(version: str, flavor: str) -> str:
    """The release asset for one version and flavor."""
    if flavor not in FLAVORS:
        raise ValueError(f"unknown flavor {flavor}")
    return f"riderless-worker-{version}-{PLATFORM}-{flavor}.tar.gz"


def _resolve_member(root: Path, raw: Any, field: str, schema: int) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"manifest does not name {field}")
    path = Path(raw)
    if schema == 1:
        # The old form was written with str(Path.resolve()) and is absolute.
        if not path.is_absolute():
            raise ValueError(f"schema 1 manifest {field} is not an absolute path")
        return path.resolve()
    if path.is_absolute():
        raise ValueError(f"manifest {field} must be relative to the manifest directory")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"manifest {field} escapes the bundle directory")
    return resolved


def manifest_schema_version(manifest: dict[str, Any]) -> int:
    version = manifest.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"manifest schema_version {version!r} is not supported; this build "
            f"reads {', '.join(str(item) for item in SUPPORTED_SCHEMA_VERSIONS)}"
        )
    return int(version)


def resolve_manifest_paths(
    manifest: dict[str, Any], manifest_path: Path
) -> tuple[Path, Path]:
    """The executable and runtime directory a manifest points at.

    Schema 2 resolves both against the manifest's own directory, which is what
    makes an unpacked bundle verify the same wherever it was unpacked.
    """
    schema = manifest_schema_version(manifest)
    root = manifest_path.resolve().parent
    executable = _resolve_member(root, manifest.get("executable"), "executable", schema)
    runtime_dir = _resolve_member(
        root, manifest.get("runtime_dir"), "runtime_dir", schema
    )
    return executable, runtime_dir


def read_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest: Any = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be an object")
    return manifest


def pack(directory: Path, archive: Path) -> Path:
    """Archive a bundle so that extracting it recreates the same directory.

    Members are stored without a leading component, so `tar -xzf` into
    `build/api-worker` lands `build.json` exactly where `ApiConfig` expects it.
    """
    directory = directory.resolve()
    manifest_path = directory / MANIFEST_NAME
    manifest = read_manifest(manifest_path)
    if manifest_schema_version(manifest) != MANIFEST_SCHEMA_VERSION:
        raise ValueError("only a relocatable manifest can be packed")
    executable, runtime_dir = resolve_manifest_paths(manifest, manifest_path)
    if executable != directory / WORKER_RELATIVE:
        raise ValueError(f"bundle executable is not {WORKER_RELATIVE}")
    if runtime_dir != directory / RUNTIME_RELATIVE:
        raise ValueError(f"bundle runtime_dir is not {RUNTIME_RELATIVE}")
    if archive.exists():
        raise ValueError("archive exists; choose a new file")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for member in BUNDLE_MEMBERS:
            source = directory / member
            if not source.exists():
                raise ValueError(f"bundle is missing {member}")
            # Symlinks are stored as symlinks: the ggml backend loader finds a
            # backend through its unversioned `libggml-cuda.so` link.
            tar.add(source, arcname=str(member), recursive=True)
    return archive


def unpack(archive: Path, directory: Path) -> Path:
    """Extract a bundle into a directory that must not already exist."""
    if directory.exists():
        raise ValueError("output exists; choose a new directory")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        # `data` refuses absolute members, `..` escapes and device nodes.
        tar.extractall(directory, filter="data")
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"archive has no {MANIFEST_NAME} at its root")
    if not (directory / WORKER_RELATIVE).is_file():
        raise ValueError(f"archive has no {WORKER_RELATIVE}")
    (directory / WORKER_RELATIVE).chmod(0o755)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packer = commands.add_parser("pack", help="archive a built bundle")
    packer.add_argument("--directory", type=Path, required=True)
    packer.add_argument("--output", type=Path, required=True)
    unpacker = commands.add_parser("unpack", help="extract a bundle archive")
    unpacker.add_argument("--archive", type=Path, required=True)
    unpacker.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "pack":
            print(pack(args.directory, args.output))
        else:
            print(unpack(args.archive, args.output))
    except (OSError, TypeError, ValueError, tarfile.TarError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
