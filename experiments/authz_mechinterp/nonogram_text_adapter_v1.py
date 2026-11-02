"""New text/JSON nonogram interface; independent public-constraint grading.

Not the recovered Prolog evaluation. References and generator metadata are
private; any binary grid satisfying the published run clues earns credit.
"""
from __future__ import annotations

import json
from itertools import groupby

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256


def public_constraints(metadata):
    rows, cols = metadata["num_rows"], metadata["num_cols"]
    if type(rows) is not int or type(cols) is not int or not 1 <= rows <= 16 or not 1 <= cols <= 16:
        raise ValueError("integer grid dimensions in 1..16 required")
    result = {"num_rows": rows, "num_cols": cols}
    for key, count, span in (("row_clues", rows, cols), ("col_clues", cols, rows)):
        clues = metadata[key]
        if not isinstance(clues, list) or len(clues) != count:
            raise ValueError("clue count differs from dimension")
        if any(not isinstance(line, list) or any(type(n) is not int or n < 1 for n in line)
               or sum(line)+max(0, len(line)-1) > span for line in clues):
            raise ValueError("positive run lengths with required separating gaps must fit")
        result[key] = [line.copy() for line in clues]
    if sum(map(sum, result["row_clues"])) != sum(map(sum, result["col_clues"])):
        raise ValueError("row/column filled-cell totals differ")
    return result


def runs(line):
    return [sum(1 for _ in group) for bit, group in groupby(line) if bit == 1]


def valid_grid(constraints, grid):
    c = public_constraints(constraints)
    if (not isinstance(grid, list) or len(grid) != c["num_rows"]
            or any(not isinstance(row, list) or len(row) != c["num_cols"]
                   or any(type(bit) is not int or bit not in (0, 1) for bit in row) for row in grid)):
        return False
    return ([runs(row) for row in grid] == c["row_clues"]
            and [runs(column) for column in zip(*grid)] == c["col_clues"])


def grade_answer(constraints, answer):
    if not isinstance(answer, str) or len(answer) > 8192:
        return False
    try:
        grid = json.loads(answer)
    except (json.JSONDecodeError, RecursionError):
        return False
    return valid_grid(constraints, grid)


def adapt_example(metadata, reference):
    constraints = public_constraints(metadata)
    if not valid_grid(constraints, reference):
        raise ValueError("private reference violates public constraints")
    fingerprint = content_sha256({"task": "nonogram", **constraints})
    prompt = ("Solve the nonogram specified below. Each clue lists the lengths of consecutive runs of 1s "
              "in that row or column, in order. Separate adjacent runs by at least one 0. "
              "An empty clue list means all cells in that line are 0. "
              "Return only a JSON array of rows containing integers 0 or 1. "
              "Any grid satisfying every row and column clue is accepted.\n"
              + json.dumps(constraints, sort_keys=True, separators=(",", ":")))
    return ({"schema": "sch.vpd-nonogram-text-question.v1", "family": "nonogram",
             "public_instance_sha256": fingerprint, "constraints": constraints, "prompt": prompt},
            {"schema": "sch.vpd-nonogram-text-grader.v1", "public_instance_sha256": fingerprint,
             "reference_grid": [row.copy() for row in reference],
             "scoring": "constraint_validity_not_reference_exact_match"})
