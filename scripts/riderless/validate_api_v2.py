"""End-to-end checks of the /v2 snapshot routes against a running server.

Start the API with the snapshot worker (and, on a card that cannot hold two
copies of the model, without the v1 backend), then point this at it:

    riderless-api --gpu --no-v1 --snapshots \\
        --snapshot-worker build/snapshot-worker/build/riderless-snapshot-worker \\
        --snapshot-manifest build/snapshot-worker/build.json --port 18090 ...
    python scripts/riderless/validate_api_v2.py --url http://127.0.0.1:18090

Each check prints PASS/FAIL; the exit code is nonzero if any check failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx

Row = dict[str, Any]
STATE = (
    "Hello. The shop already issued my refund last Tuesday, but it has not "
    "reached my card. Please check its status. I am calm and happy to wait."
)
ROUTE = {
    "type": "choice",
    "instructions": "Choose the primary intent supported by the message.",
    "criteria": {
        "new_refund": "Request a refund that has not yet been issued",
        "missing_refund": "Check a refund already issued but not received",
        "duplicate_charge": "Report being charged twice",
    },
}
ISSUED = {
    "type": "noul",
    "instructions": "Does the message say the refund was already issued?",
}


class Checks:
    def __init__(self) -> None:
        self.failed = 0

    def check(self, name: str, passed: bool, detail: object = "") -> None:
        self.failed += 0 if passed else 1
        print(f"{'PASS' if passed else 'FAIL'} {name} {str(detail)[:300]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18090")
    args = parser.parse_args()
    checks = Checks()
    with httpx.Client(base_url=args.url, timeout=300.0) as client:

        def post(path: str, body: Row) -> httpx.Response:
            return client.post(path, json=body)

        models = client.get("/v2/models")
        checks.check("models", models.status_code == 200, models.text)

        created = post(
            "/v2/snapshots",
            {
                "input": {"kind": "context", "state": STATE},
                "checkpoints": [18, 30],
                "persistence": "disk",
                "ttl_seconds": 600,
            },
        )
        checks.check("create 18+30", created.status_code == 200, created.text)
        if created.status_code != 200:
            sys.exit(1)
        rows = {row["completed_blocks"]: row for row in created.json()["snapshots"]}
        s18, s30 = rows[18]["id"], rows[30]["id"]
        checks.check("30 parent is 18", rows[30].get("parent") == s18, rows[30])

        def decide(snapshot: str, questions: Row, **extra: Any) -> httpx.Response:
            return post(
                "/v2/decisions",
                {
                    "snapshot": {"id": snapshot, "relationship": "followup"},
                    "questions": questions,
                    "readout": {"completed_blocks": extra.pop("blocks", 30)},
                    **extra,
                },
            )

        first = decide(s18, {"route": ROUTE})
        body = first.json()
        checks.check(
            "decide from 18 promotes",
            first.status_code == 200
            and body["snapshot_usage"]["promotion"] == "performed"
            and body["usage"]["output_tokens"] == 0,
            body.get("snapshot_usage"),
        )
        again = decide(s18, {"route": ROUTE}).json()
        checks.check(
            "promotion memoized",
            again["snapshot_usage"]["promotion"] == "reused",
            again["snapshot_usage"],
        )
        paired = decide(s30, {"route": ROUTE}).json()
        checks.check(
            "promoted 18 == paired 30",
            paired["answers"]["route"]["probabilities"]
            == body["answers"]["route"]["probabilities"],
            [paired["answers"]["route"], body["answers"]["route"]],
        )
        checks.check(
            "expected answer",
            paired["answers"]["route"]["choice"] == "missing_refund",
            paired["answers"]["route"],
        )

        a1 = decide(s30, {"issued": ISSUED}).json()
        decide(s30, {"route": ROUTE})
        a2 = decide(s30, {"issued": ISSUED}).json()
        both = decide(s30, {"route": ROUTE, "issued": ISSUED}).json()
        checks.check(
            "branch isolation A,B,A and joint",
            a1["answers"]["issued"]
            == a2["answers"]["issued"]
            == both["answers"]["issued"]
            and both["answers"]["route"] == paired["answers"]["route"],
            [a1["answers"], both["answers"]],
        )

        early = decide(s30, {"route": ROUTE}, blocks=18)
        checks.check(
            "readout 18 refused",
            early.status_code == 422
            and early.json()["error"]["code"] == "capability_unavailable",
            early.text,
        )

        state = post(
            "/v2/state-evaluations",
            {
                "snapshot": {"id": s30, "relationship": "followup"},
                "prompt": "Consider whether the refund was already issued.",
                "readout": {
                    "completed_blocks": 30,
                    "export": ["last_residual", "last_normalized", "top_logits"],
                    "top_logits": 5,
                },
                "save_result_snapshot": True,
            },
        )
        state_body = state.json()
        checks.check(
            "state evaluation",
            state.status_code == 200
            and set(state_body.get("vectors", {}))
            == {"last_residual", "last_normalized"}
            and len(state_body.get("top_logits", [])) == 5
            and state_body["usage"]["output_tokens"] == 0,
            {k: v for k, v in state_body.items() if k != "vectors"},
        )
        child = state_body.get("child")
        if child:
            follow = decide(child, {"issued": ISSUED})
            checks.check(
                "followup to a prompt readout snapshot answers or is refused",
                follow.status_code in (200, 409),
                follow.text,
            )
            replace = post(
                "/v2/decisions",
                {
                    "snapshot": {"id": child, "relationship": "replace_question"},
                    "questions": {"issued": ISSUED},
                    "readout": {"completed_blocks": 30},
                },
            )
            checks.check(
                "replace_question uses the context parent",
                replace.status_code == 200
                and replace.json()["snapshot_usage"]["effective_parent"] != child,
                replace.text,
            )

        meta = client.get(f"/v2/snapshots/{s18}")
        checks.check("metadata", meta.status_code == 200, meta.text)
        vectors = client.get(
            f"/v2/snapshots/{s18}/vectors",
            params={"which": "h18", "row_begin": 0, "row_end": 4},
        )
        checks.check(
            "vectors",
            vectors.status_code == 200 and "base64" in vectors.text,
            vectors.text[:200],
        )
        too_many = client.get(
            f"/v2/snapshots/{s18}/vectors",
            params={"which": "h18", "row_begin": 0, "row_end": 65},
        )
        checks.check("vector bound", too_many.status_code == 422, too_many.text)

        mismatch = post(
            "/v2/snapshots",
            {
                "input": {"kind": "context", "state": "x" * 20000},
                "checkpoints": [30],
                "ttl_seconds": 60,
            },
        )
        checks.check(
            "budget refused", mismatch.status_code in (413, 422), mismatch.text[:200]
        )

        if child:
            deleted = client.delete(f"/v2/snapshots/{child}")
            gone = client.get(f"/v2/snapshots/{child}")
            checks.check(
                "delete",
                deleted.status_code in (200, 204) and gone.status_code == 404,
                [deleted.text, gone.text],
            )
    print(json.dumps({"failed": checks.failed}))
    sys.exit(1 if checks.failed else 0)


if __name__ == "__main__":
    main()
