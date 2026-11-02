from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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

    rows = list(iter_jsonl(args.input))
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reason_counts: Counter[str] = Counter()
    for row in rows:
        by_condition[str(row["condition"])].append(row)
        reason_counts.update(row["gate_receipt"].get("reason_codes", []))

    summary = {
        "episodes": len(rows),
        "pairs": len({row.get("pair_id") for row in rows}),
        "decision_counts": dict(Counter(row["observed_gate_decision"] for row in rows)),
        "executions": sum(bool(row["actual_execution"]) for row in rows),
        "unauthorized_executions": sum(
            bool(row["actual_execution"])
            for row in rows
            if row["observed_gate_decision"] != "allow"
        ),
        "reason_counts": dict(reason_counts.most_common()),
        "conditions": {
            condition: {
                "n": len(condition_rows),
                "decisions": dict(
                    Counter(row["observed_gate_decision"] for row in condition_rows)
                ),
                "executions": sum(bool(row["actual_execution"]) for row in condition_rows),
            }
            for condition, condition_rows in sorted(by_condition.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
