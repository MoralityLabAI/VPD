import itertools
import json
import pytest
from experiments.authz_mechinterp.graph_coloring_text_adapter_v1 import adapt_example, public_graph, grade_answer, valid_assignment


def metadata():
    return {"num_vertices": 3, "chromatic_number": 3, "edges": [[1, 2], [2, 3]],
            "solution": {1: 0, 2: 1, 3: 0}, "private_note": "DO_NOT_SHOW_REFERENCE_METADATA"}


def test_question_whitelist_excludes_private_fields_and_solution():
    question, grader = adapt_example(metadata())
    assert set(question) == {"schema", "family", "public_instance_sha256", "constraints", "prompt"}
    assert "DO_NOT_SHOW_REFERENCE_METADATA" not in json.dumps(question)
    assert "reference_assignment" not in json.dumps(question)
    assert grader["reference_assignment"] == [0, 1, 0]
    assert question["public_instance_sha256"] == grader["public_instance_sha256"]


def test_reference_does_not_determine_question_or_public_hash():
    first = metadata()
    second = metadata()
    second["solution"] = {1: 2, 2: 0, 3: 2}
    assert adapt_example(first)[0] == adapt_example(second)[0]


def test_bound_is_not_exact_or_optimal_color_count():
    question, _ = adapt_example(metadata())
    assert grade_answer(question["constraints"], "[0,1,0]")
    assert grade_answer(question["constraints"], "[2,0,1]")
    assert "fewer" in question["prompt"]
    assert "chromatic_number" not in question["constraints"]


def test_exhaustive_three_vertex_path_accepts_12_colorings():
    graph = public_graph(metadata())
    accepted = [colors for colors in itertools.product(range(3), repeat=3) if valid_assignment(graph, list(colors))]
    assert len(accepted) == 3*2*2


def test_exhaustive_triangle_accepts_six_permutations():
    source = metadata()
    source["edges"].append([1, 3])
    graph = public_graph(source)
    assert sum(valid_assignment(graph, list(c)) for c in itertools.product(range(3), repeat=3)) == 6


@pytest.mark.parametrize("answer", ["[0,0,1]", "[0,1]", "[0,1,3]", "[-1,0,1]", "[true,0,1]", "[0.0,1,0]", '{"1":0,"2":1,"3":0}', "text [0,1,0]", "[0,1,0] trailing", "null", "[NaN,1,0]"])
def test_invalid_answers_rejected(answer):
    assert not grade_answer(public_graph(metadata()), answer)


@pytest.mark.parametrize("edges", [[[1, 1]], [[0, 1]], [[1, 4]], [[1, 2], [2, 1]], [[True, 2]]])
def test_invalid_graphs_rejected(edges):
    source = metadata()
    source["edges"] = edges
    with pytest.raises(ValueError):
        public_graph(source)


def test_invalid_private_reference_cannot_be_packaged():
    source = metadata()
    source["solution"] = {1: 0, 2: 0, 3: 0}
    with pytest.raises(ValueError, match="reference"):
        adapt_example(source)


def test_json_roundtrip_reference_keys_supported():
    source = metadata()
    assert adapt_example(json.loads(json.dumps(source))) == adapt_example(source)


def test_public_edge_order_does_not_create_another_instance():
    source = metadata()
    changed = metadata()
    changed["edges"] = [[3, 2], [2, 1]]
    assert adapt_example(source)[0] == adapt_example(changed)[0]
