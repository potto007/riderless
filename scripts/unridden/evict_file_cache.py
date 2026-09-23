"""Evict specific files from the page cache without touching anything else.

Usage: python evict_file_cache.py PATH [PATH ...]

Each PATH may be a file or a directory (walked recursively). Uses
posix_fadvise(POSIX_FADV_DONTNEED), which asks the kernel to drop the cached
pages of exactly these files. It needs no privileges and does not affect other
processes' caches, unlike /proc/sys/vm/drop_caches. Prints the number of files
and bytes advised, plus free memory before and after.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def available_gib() -> float:
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return float("nan")


def evict(path: Path) -> int:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return path.stat().st_size


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    before = available_gib()
    files = 0
    total = 0
    for raw in argv:
        root = Path(raw)
        if root.is_file():
            targets = [root]
        else:
            targets = [p for p in root.rglob("*") if p.is_file()]
        for target in targets:
            try:
                total += evict(target)
                files += 1
            except OSError as error:
                print(f"skip {target}: {error}", file=sys.stderr)
    after = available_gib()
    print(
        f"evicted {files} files, {total / 2**30:.1f} GiB advised; "
        f"MemAvailable {before:.1f} -> {after:.1f} GiB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
