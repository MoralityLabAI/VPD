import sys
import pytest
from experiments.authz_mechinterp.check_graph_coloring_generator_canary_v1 import checked_source_bytes, load_generator, NAMESPACE


def test_changed_source_is_rejected_before_any_execution(tmp_path):
    (tmp_path/"base.py").write_text("raise RuntimeError('must not execute')")
    (tmp_path/"graph_coloring.py").write_text("raise RuntimeError('must not execute')")
    before = set(sys.modules)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_generator(tmp_path)
    assert set(sys.modules) == before


def test_missing_source_is_not_reconstructed_implicitly(tmp_path):
    with pytest.raises(FileNotFoundError):
        checked_source_bytes(tmp_path)
