"""Bounded CPU-only canary of two reviewed, hash-pinned recovered source files."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from random import Random

from experiments.authz_mechinterp.graph_coloring_text_adapter_v1 import adapt_example, grade_answer
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json

SOURCE = Path("E:/Recovered_C_projects/Remote_clones/prime_intellect_research_environments/environments/reasoning/prolog/prolog/generators")
PINS = {"base.py": "6257dca4bab8914c31621fddfa160c05bd89cef4a71df07cd0835f51bbd9cbd4",
        "graph_coloring.py": "fe3c56631a14e2ded2cc360f79f9bdc77460bfb0e18cc0636418e0a5e23eaed2"}
NAMESPACE = "_vpd_reviewed_graph_canary"


def checked_source_bytes(directory):
    contents = {name: (directory/name).read_bytes() for name in PINS}
    if any(hashlib.sha256(contents[name]).hexdigest() != expected for name, expected in PINS.items()):
        raise ValueError("reviewed source hash mismatch; no source may execute")
    return contents


def load_generator(directory):
    # Compile the exact checked bytes; never run the source package's initializer.
    contents = checked_source_bytes(directory)
    package = types.ModuleType(NAMESPACE)
    package.__path__ = []
    sys.modules[NAMESPACE] = package
    for name in ("base.py", "graph_coloring.py"):
        module_name = f"{NAMESPACE}.{Path(name).stem}"
        module = types.ModuleType(module_name)
        module.__file__ = str(directory/name)
        module.__package__ = NAMESPACE
        sys.modules[module_name] = module
        exec(compile(contents[name], str(directory/name), "exec"), module.__dict__)
    return sys.modules[f"{NAMESPACE}.graph_coloring"].GraphColoringGenerator


def run_worker():
    generator = load_generator(SOURCE)()
    questions, graders, checks = [], [], []
    for seed in (0, 1, 2, 3):
        first = generator.generate(Random(seed), "medium")
        second = generator.generate(Random(seed), "medium")
        q, g = adapt_example(first.metadata)
        q2, g2 = adapt_example(second.metadata)
        if (q, g) != (q2, g2):
            raise ValueError("generator/adapter not deterministic for this seed")
        graph, reference = q["constraints"], g["reference_assignment"]
        permuted = [(c+1) % graph["max_colors"] for c in reference]
        if not grade_answer(graph, json.dumps(reference)) or not grade_answer(graph, json.dumps(permuted)):
            raise ValueError("valid reference or color permutation rejected")
        if not graph["edges"]:
            raise ValueError("canary needs an edge for its invalid-answer check")
        a, b = graph["edges"][0]
        conflicting = reference.copy()
        conflicting[b-1] = conflicting[a-1]
        if grade_answer(graph, json.dumps(conflicting)):
            raise ValueError("edge-conflicting assignment accepted")
        id = f"graph-coloring-development-medium-seed-{seed}"
        questions.append({"id": id, "role": "development_only", **q})
        graders.append({"id": id, **g})
        checks.append({"id": id, "vertices": graph["vertex_count"], "edges": len(graph["edges"]),
                       "max_colors": graph["max_colors"], "deterministic": True,
                       "reference_pass": True, "permuted_coloring_pass": True, "conflicting_coloring_rejected": True})
    hashes = [q["public_instance_sha256"] for q in questions]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate labeled public graph in canary")
    return {"questions": questions, "graders": graders, "checks": checks, "python_version": sys.version}


def generate_report():
    completed = subprocess.run([sys.executable, "-B", "-m", __name__ if __name__ != "__main__" else
                                "experiments.authz_mechinterp.check_graph_coloring_generator_canary_v1", "--worker"],
                               capture_output=True, text=True, check=True, timeout=20)
    result = json.loads(completed.stdout)
    body = {"schema": "sch.vpd-graph-coloring-generator-canary.v1", "checks": result["checks"],
            "python_version": result["python_version"], "source_directory": str(SOURCE), "source_pins": PINS,
            "questions_content_sha256": content_sha256(result["questions"]),
            "graders_content_sha256": content_sha256(result["graders"]),
            "provenance": {"runner_sha256": sha256_file(Path(__file__)),
                           "adapter_sha256": sha256_file(Path(__file__).with_name("graph_coloring_text_adapter_v1.py"))},
            "limits": ["Four medium-difficulty development seeds only; not validation across difficulties or a fresh holdout.",
                       "Distinct labeled hashes do not establish non-isomorphic graphs, semantic independence or absence of prior exposure.",
                       "No Prolog/container execution; source-compatible constraints with a new JSON answer interface.",
                       "The source color budget is not certified chromatic optimality."],
            "boundary": {"puzzle_generator_executed": True, "worker_timeout_seconds": 20,
                         "model_inference_performed": False, "sae_training_performed": False,
                         "paid_compute_launched": False, "additional_families_admitted": 0}}
    return {**body, "report_id": content_sha256(body)}, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--output-directory", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run_worker(), allow_nan=False))
        return
    report, result = generate_report()
    if args.verify:
        observed = json.loads(args.verify.read_text(encoding="utf-8"))
        if observed != report:
            raise ValueError("canary report does not reproduce")
        for name in ("questions", "graders"):
            if json.loads((args.verify.parent/f"{name}.json").read_text(encoding="utf-8")) != result[name]:
                raise ValueError(f"{name} do not reproduce")
    else:
        args.output_directory.mkdir(parents=True, exist_ok=False)
        write_json(args.output_directory/"questions.json", result["questions"])
        write_json(args.output_directory/"graders.json", result["graders"])
        write_json(args.output_directory/"report.json", report)
    print(json.dumps({"report_id": report["report_id"], "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
