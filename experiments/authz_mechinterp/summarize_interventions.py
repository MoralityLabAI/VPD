from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(args.input):
        groups[(row["condition"], float(row["coefficient"]))].append(row)

    summary = []
    for (condition, coefficient), rows in sorted(groups.items()):
        n = len(rows)
        summary.append(
            {
                "condition": condition,
                "coefficient": coefficient,
                "n": n,
                "exact_commit_proposal_rate": sum(
                    bool(r["exact_commit_proposal"]) for r in rows
                )
                / n,
                "scope_violation_rate": sum(bool(r["scope_violation"]) for r in rows)
                / n,
                "authorization_request_rate": sum(
                    bool(r["requests_authorization"]) for r in rows
                )
                / n,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
