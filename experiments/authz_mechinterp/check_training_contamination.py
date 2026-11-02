from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def canonical_messages(messages: list[dict[str, Any]]) -> str:
    return json.dumps(messages, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--training-traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation = list(iter_jsonl(args.evaluation))
    training = list(iter_jsonl(args.training_traces))
    training_message_sets = {
        canonical_messages(row.get("messages", [])) for row in training
    }
    training_contents = {
        str(message.get("content", ""))
        for row in training
        for message in row.get("messages", [])
        if str(message.get("content", "")).strip()
    }
    exact_prompt_rows = [
        row["episode_id"]
        for row in evaluation
        if canonical_messages(row["messages"]) in training_message_sets
    ]
    exact_message_matches = [
        {
            "episode_id": row["episode_id"],
            "role": message["role"],
        }
        for row in evaluation
        for message in row["messages"]
        if str(message.get("content", "")) in training_contents
    ]
    payload = {
        "schema": "authz_training_contamination_check_v1",
        "evaluation_path": str(args.evaluation),
        "evaluation_sha256": sha256_file(args.evaluation),
        "training_traces_path": str(args.training_traces),
        "training_traces_sha256": sha256_file(args.training_traces),
        "evaluation_rows": len(evaluation),
        "training_rows": len(training),
        "exact_prompt_matches": exact_prompt_rows,
        "exact_message_matches": exact_message_matches,
        "exact_match_check_passed": not exact_prompt_rows and not exact_message_matches,
        "scope": "Exact canonical prompt and exact individual-message overlap only.",
        "limitation": "Semantic, paraphrase, tokenizer-level, and pretraining contamination remain unknown.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if not payload["exact_match_check_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
