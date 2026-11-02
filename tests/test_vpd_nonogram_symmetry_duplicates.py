from copy import deepcopy
from itertools import product
import re

import pytest

from experiments.authz_mechinterp.audit_nonogram_symmetry_duplicates_v1 import audit_questions, symmetry_class, transform_clues
from experiments.authz_mechinterp.nonogram_text_adapter_v1 import adapt_example


def oracle(grid):
    def lengths(line):
        return [len(m) for m in re.findall("1+", "".join(map(str, line)))]
    return dict(num_rows=len(grid), num_cols=len(grid[0]), row_clues=[lengths(r) for r in grid], col_clues=[lengths(c) for c in zip(*grid)])


@pytest.mark.parametrize("rows,cols", [(2, 3), (3, 3)])
def test_all_small_grids_all_eight_transformations(rows, cols):
    for bits in product((0, 1), repeat=rows*cols):
        grid = [list(bits[i*cols:(i+1)*cols]) for i in range(rows)]
        constraints = oracle(grid)
        original = deepcopy(constraints)
        key = symmetry_class(constraints)["symmetry_class_sha256"]
        for t, r, c in product((False, True), repeat=3):
            transformed_grid = [list(v) for v in zip(*grid)] if t else [line.copy() for line in grid]
            if r: transformed_grid = transformed_grid[::-1]
            if c: transformed_grid = [line[::-1] for line in transformed_grid]
            transformed = transform_clues(constraints, transpose=t, flip_rows=r, flip_cols=c)
            assert transformed == oracle(transformed_grid)
            assert symmetry_class(transformed)["symmetry_class_sha256"] == key
        assert constraints == original


def question(grid, id):
    q, _ = adapt_example(oracle(grid), grid)
    return {"id": id, **q}


def test_transpose_collision_detected_and_separate_puzzle_retained():
    grid = [[1, 1, 0], [0, 0, 1]]
    transposed = [list(v) for v in zip(*grid)]
    a, b, c = question(grid, "a"), question(transposed, "b"), question([[0, 0, 0], [0, 0, 0]], "c")
    assert a["public_instance_sha256"] != b["public_instance_sha256"]
    result = audit_questions([a, b, c])
    assert result["symmetry_classes"] == 2
    assert result["duplicate_classes"][0]["ids"] == ["a", "b"]
    assert result == audit_questions([c, b, a])


def test_duplicate_id_and_changed_constraints_rejected():
    q = question([[1, 0], [0, 1]], "a")
    with pytest.raises(ValueError):
        audit_questions([q, q])
    q["public_instance_sha256"] = "0"*64
    with pytest.raises(ValueError, match="fingerprint"):
        audit_questions([q])


def test_symmetric_grid_has_smaller_orbit():
    assert symmetry_class(oracle([[0, 0], [0, 0]]))["distinct_orientations"] == 1


def test_non_boolean_flag_rejected():
    with pytest.raises(ValueError):
        transform_clues(oracle([[1]]), transpose=1)
