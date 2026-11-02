"""New no-tool text interface for graph-coloring metadata, not official Prolog eval.

Only explicit public constraints enter the question. A reference assignment is
validated separately; alternative valid assignments must receive equal credit.
No generator, subprocess, model, or external environment is imported.
"""
from __future__ import annotations

import json
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256


def public_graph(metadata):
    n, k = metadata["num_vertices"], metadata["chromatic_number"]
    # The source's chromatic_number is a greedy-derived bound, not an optimum proof.
    if type(n) is not int or type(k) is not int or not 1 <= k <= n <= 128:
        raise ValueError("bounded integer vertex/color counts required")
    edges = metadata["edges"]
    if not isinstance(edges, list):
        raise ValueError("edge list required")
    canonical = set()
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2 or any(type(v) is not int or not 1 <= v <= n for v in edge) or edge[0] == edge[1]:
            raise ValueError("simple undirected graph endpoints required")
        normalized = tuple(sorted(edge))
        if normalized in canonical:
            raise ValueError("duplicate undirected edge")
        canonical.add(normalized)
    return {"vertex_count": n, "max_colors": k, "edges": [list(e) for e in sorted(canonical)]}


def valid_assignment(graph, colors):
    # Revalidate the public constraints, including when loaded from another file.
    graph = public_graph({"num_vertices": graph["vertex_count"], "chromatic_number": graph["max_colors"], "edges": graph["edges"]})
    if not isinstance(colors, list) or len(colors) != graph["vertex_count"] or any(type(c) is not int or not 0 <= c < graph["max_colors"] for c in colors):
        return False
    return all(colors[a-1] != colors[b-1] for a, b in graph["edges"])


def grade_answer(graph, answer):
    if not isinstance(answer, str):
        return False
    try:
        colors = json.loads(answer)
    except json.JSONDecodeError:
        return False
    return valid_assignment(graph, colors)


def adapt_example(metadata):
    graph = public_graph(metadata)
    source_solution = metadata["solution"]
    if not isinstance(source_solution, dict) or len(source_solution) != graph["vertex_count"]:
        raise ValueError("complete private reference assignment required")
    normalized = {}
    for key, color in source_solution.items():
        if type(key) is int:
            vertex = key
        elif isinstance(key, str) and key.isascii() and key.isdecimal() and str(int(key)) == key:
            vertex = int(key)
        else:
            raise ValueError("invalid reference vertex key")
        if vertex in normalized or not 1 <= vertex <= graph["vertex_count"]:
            raise ValueError("duplicate/out-of-range reference vertex")
        normalized[vertex] = color
    reference = [normalized[v] for v in range(1, graph["vertex_count"]+1)]
    if not valid_assignment(graph, reference):
        raise ValueError("private reference violates public constraints")
    prompt = (
        f"Color an undirected graph with vertices 1 through {graph['vertex_count']}. "
        f"Use integer colors 0 through {graph['max_colors']-1}; adjacent vertices must have different colors. "
        "Return only a JSON array with one color per vertex, in vertex order. "
        "You may use fewer than the allowed colors; any valid coloring is accepted.\n"
        "Edges: " + json.dumps(graph["edges"], separators=(",", ":"))
    )
    fingerprint = content_sha256({"task": "graph_coloring", **graph})
    question = {"schema": "sch.vpd-graph-coloring-text-question.v1", "family": "graph_coloring",
                "public_instance_sha256": fingerprint, "constraints": graph, "prompt": prompt}
    grader = {"schema": "sch.vpd-graph-coloring-text-grader.v1", "public_instance_sha256": fingerprint,
              "reference_assignment": reference, "scoring": "constraint_validity_not_reference_exact_match"}
    return question, grader
