from copy import deepcopy
from itertools import product
import json
import re

import pytest

from experiments.authz_mechinterp.nonogram_text_adapter_v1 import adapt_example, grade_answer, public_constraints, valid_grid


def oracle_clues(grid):
    def lengths(line):
        return [len(match) for match in re.findall("1+", "".join(map(str, line)))]
    return dict(num_rows=len(grid), num_cols=len(grid[0]),
                row_clues=[lengths(row) for row in grid], col_clues=[lengths(col) for col in zip(*grid)])


def test_exhaustive_two_by_three_against_regex_oracle():
    grids = [[list(bits[:3]), list(bits[3:])] for bits in product((0, 1), repeat=6)]
    constraints = [oracle_clues(grid) for grid in grids]
    for expected in constraints:
        for grid, actual in zip(grids, constraints):
            assert valid_grid(expected, grid) == (expected == actual)


def test_multiple_valid_solutions_and_reference_invariance():
    a, b = [[1, 0], [0, 1]], [[0, 1], [1, 0]]
    c = oracle_clues(a)
    qa, ga = adapt_example(c, a)
    qb, gb = adapt_example(c, b)
    assert qa == qb and ga != gb
    assert grade_answer(c, json.dumps(a)) and grade_answer(c, json.dumps(b))


def test_private_metadata_not_exposed_and_inputs_not_mutated():
    grid = [[1, 0], [0, 1]]
    metadata = {**oracle_clues(grid), "density": 0.12345, "solution": "PRIVATE", "expected_answer": "PRIVATE"}
    before = deepcopy(metadata)
    question, grader = adapt_example(metadata, grid)
    assert "PRIVATE" not in json.dumps(question) and "0.12345" not in question["prompt"]
    assert set(question["constraints"]) == {"num_rows", "num_cols", "row_clues", "col_clues"}
    assert metadata == before
    grid[0][0] = 0
    assert grader["reference_grid"][0][0] == 1


@pytest.mark.parametrize("answer", ["null", "true", "[[true,0],[0,1]]", "[[1.0,0],[0,1]]", "[[1,1],[0,1]]", "[[1,0]]", "not json", "0"*8193])
def test_invalid_answers(answer):
    assert not grade_answer(oracle_clues([[1, 0], [0, 1]]), answer)


@pytest.mark.parametrize("change", ["dimension", "zero_run", "gap", "total", "clue_count"])
def test_bad_constraints(change):
    c = oracle_clues([[1, 0], [0, 1]])
    if change == "dimension": c["num_rows"] = True
    if change == "zero_run": c["row_clues"][0] = [0]
    if change == "gap": c["row_clues"][0] = [1, 1]
    if change == "total": c["row_clues"][0] = []
    if change == "clue_count": c["row_clues"] = [[1]]
    with pytest.raises(ValueError):
        public_constraints(c)


def test_invalid_reference_is_rejected():
    with pytest.raises(ValueError):
        adapt_example(oracle_clues([[1, 0], [0, 1]]), [[0, 0], [0, 0]])


def test_empty_clues_are_not_zero_runs():
    c = oracle_clues([[0, 0], [0, 0]])
    assert c["row_clues"] == [[], []]
    assert grade_answer(c, "[[0,0],[0,0]]")
