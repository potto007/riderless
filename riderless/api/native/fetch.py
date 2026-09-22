"""Install a published worker bundle instead of compiling one.

`riderless-api worker fetch` downloads the release asset built for the
installed riderless version, checks it against the release's `SHA256SUMS`,
asks `gh attestation verify` whether GitHub actually built it, and unpacks it
into `build/api-worker`, where `ApiConfig`'s defaults already look.

Nothing here weakens the startup checks. A fetched bundle carries the same
manifest a local build writes, and the API still re-hashes the executable,
every runtime library and the model before the worker is allowed to start.
What this replaces is the compiler, not the verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path

from riderless.api.native.build import TESTED_LLAMA_REVISION, TESTED_LLAMA_TAG
from riderless.api.native.bundle import (
    FLAVORS,
    MANIFEST_NAME,
    asset_name,
    read_manifest,
    unpack,
)

LOGGER = logging.getLogger("riderless.api.native.fetch")

RELEASE_REPOSITORY = "potto007/riderless"
CHECKSUM_FILE = "SHA256SUMS"
DEFAULT_OUTPUT = Path("build/api-worker")
DOWNLOAD_TIMEOUT = 120.0

# The driver each CUDA flavor needs, from NVIDIA's own minimum-driver table.
# CUDA 13.x wants 580.65.06 or newer on Linux x86_64; CUDA 12.x runs on
# 525.60.13 or newer through minor version compatibility.
CUDA13_MINIMUM_DRIVER = (580, 65)
CUDA12_MINIMUM_DRIVER = (525, 60)


class FetchError(ValueError):
    """The bundle could not be installed. Nothing was left behind."""


def _driver_version() -> tuple[int, ...] | None:
    """The NVIDIA driver version, or None when there is no usable nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        report = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if report.returncode != 0:
        return None
    first = report.stdout.strip().splitlines()
    if not first:
        return None
    parts = first[0].strip().split(".")
    if not parts or not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def detect_flavor() -> tuple[str, str]:
    """Pick a flavor from the installed driver, and say why."""
    driver = _driver_version()
    if driver is None:
        return "cpu", "no usable nvidia-smi, so no GPU is assumed"
    printed = ".".join(str(part) for part in driver)
    if driver[:2] >= CUDA13_MINIMUM_DRIVER:
        return "cuda13", f"driver {printed} supports CUDA 13"
    if driver[:2] >= CUDA12_MINIMUM_DRIVER:
        return "cuda12", f"driver {printed} supports CUDA 12 but not CUDA 13"
    return "cpu", f"driver {printed} is older than CUDA 12 requires"


def release_tag(requested: str | None) -> str:
    if requested is None:
        try:
            requested = installed_version("riderless")
        except PackageNotFoundError as error:  # pragma: no cover - install shape
            raise FetchError(
                "riderless is not installed, so its version is unknown; pass --version"
            ) from error
    return requested if requested.startswith("v") else f"v{requested}"


def download(url: str, destination: Path) -> Path:
    request = urllib.request.Request(url, headers={"User-Agent": "riderless-fetch"})
    try:
        with (
            urllib.request.urlopen(  # noqa: S310 - https URL built from constants
                request, timeout=DOWNLOAD_TIMEOUT
            ) as response,
            destination.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
    except urllib.error.HTTPError as error:
        raise FetchError(f"{url} returned HTTP {error.code}") from error
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        raise FetchError(f"could not download {url}: {error}") from error
    return destination


def parse_checksums(text: str) -> dict[str, str]:
    """Read `sha256sum` output: a hex digest, separator, then a file name."""
    checksums: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(\S+)", line.strip())
        if match:
            checksums[match.group(2)] = match.group(1).lower()
    if not checksums:
        raise FetchError(f"{CHECKSUM_FILE} lists no checksums")
    return checksums


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_attestation(archive: Path, repository: str, *, required: bool) -> bool:
    """Ask GitHub whether it built this file. Absent `gh` is a warning."""
    if shutil.which("gh") is None:
        message = (
            "gh is not installed, so the GitHub build attestation for "
            f"{archive.name} was not checked"
        )
        if required:
            raise FetchError(message)
        LOGGER.warning("%s", message)
        return False
    report = subprocess.run(
        ["gh", "attestation", "verify", str(archive), "--repo", repository],
        capture_output=True,
        text=True,
        check=False,
    )
    if report.returncode == 0:
        return True
    detail = (report.stderr or report.stdout).strip().splitlines()
    message = (
        f"gh attestation verify failed for {archive.name}: "
        f"{detail[-1] if detail else 'no output'}"
    )
    if required:
        raise FetchError(message)
    LOGGER.warning("%s", message)
    return False


def fetch(
    *,
    output: Path,
    flavor: str | None,
    requested_version: str | None,
    repository: str,
    require_attestation: bool,
) -> dict[str, object]:
    if output.exists():
        raise FetchError("output exists; choose a new directory")
    if flavor is None:
        flavor, reason = detect_flavor()
        print(f"flavor: {flavor} ({reason})")
    elif flavor not in FLAVORS:
        raise FetchError(f"unknown flavor {flavor}")
    tag = release_tag(requested_version)
    name = asset_name(tag.removeprefix("v"), flavor)
    base = f"https://github.com/{repository}/releases/download/{tag}"

    with tempfile.TemporaryDirectory(prefix="riderless-fetch-") as scratch:
        staging = Path(scratch)
        sums = download(f"{base}/{CHECKSUM_FILE}", staging / CHECKSUM_FILE)
        checksums = parse_checksums(sums.read_text())
        expected = checksums.get(name)
        if expected is None:
            raise FetchError(
                f"{CHECKSUM_FILE} for {tag} does not list {name}; "
                f"the release publishes: {', '.join(sorted(checksums))}"
            )
        archive = download(f"{base}/{name}", staging / name)
        actual = digest(archive)
        if actual != expected:
            raise FetchError(
                f"{name} hashes to {actual}, not the {expected} {CHECKSUM_FILE} records"
            )
        attested = verify_attestation(archive, repository, required=require_attestation)
        try:
            unpack(archive, output)
        except Exception:
            # An install is all or nothing: a half-extracted directory would
            # otherwise fail the "must not already exist" rule on the retry.
            shutil.rmtree(output, ignore_errors=True)
            raise

    manifest = read_manifest(output / MANIFEST_NAME)
    revision = str(manifest.get("llama_revision", ""))
    tested = revision == TESTED_LLAMA_REVISION
    return {
        "output": str(output),
        "tag": tag,
        "flavor": flavor,
        "asset": name,
        "sha256": actual,
        "attested": attested,
        "llama_revision": revision,
        "tested_revision": tested,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="riderless-api worker fetch")
    add_arguments(parser)
    return run(parser, parser.parse_args(argv))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"directory to unpack into; must not exist. Default {DEFAULT_OUTPUT}, "
        "which is where the API already looks",
    )
    parser.add_argument(
        "--flavor",
        choices=FLAVORS,
        default=None,
        help="which build to install. Default: chosen from the NVIDIA driver "
        "nvidia-smi reports, or cpu when there is none",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="release tag to install, with or without the leading v. "
        "Default: the installed riderless version",
    )
    parser.add_argument(
        "--repository",
        default=RELEASE_REPOSITORY,
        help=f"GitHub repository publishing the release. Default {RELEASE_REPOSITORY}",
    )
    parser.add_argument(
        "--require-attestation",
        action="store_true",
        help="fail instead of warning when the GitHub build attestation cannot "
        "be verified, including when gh is not installed",
    )


def run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        result = fetch(
            output=args.output,
            flavor=args.flavor,
            requested_version=args.version,
            repository=args.repository,
            require_attestation=args.require_attestation,
        )
    except (FetchError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    if result["tested_revision"]:
        print(
            f"llama.cpp {result['llama_revision']} is the tested revision "
            f"({TESTED_LLAMA_TAG}); the published measurements were taken on it."
        )
    else:
        print(
            f"llama.cpp {result['llama_revision']} is NOT the tested revision "
            f"{TESTED_LLAMA_REVISION} ({TESTED_LLAMA_TAG}); the published "
            "measurements may not reproduce."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
