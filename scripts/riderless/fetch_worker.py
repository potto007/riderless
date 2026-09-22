"""Install a published worker bundle from a checkout, without installing it.

This is a shim. The command of record is

    uv run python -m riderless.api.cli worker fetch

and everything it accepts, this accepts: there is one implementation, in
`riderless.api.native.fetch`. The script exists so the fetch can be run from a
clone the same way the other `scripts/riderless` drivers are.
"""

from __future__ import annotations

from riderless.api.native.fetch import main

if __name__ == "__main__":
    raise SystemExit(main())
