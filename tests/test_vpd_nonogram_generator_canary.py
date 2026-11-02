import sys
import pytest

from experiments.authz_mechinterp.check_nonogram_generator_canary_v1 import load_generator


def test_changed_source_rejected_before_execution(tmp_path):
    for name in ("base.py", "nonogram.py"):
        (tmp_path/name).write_text("raise RuntimeError('must not execute')")
    before = set(sys.modules)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_generator(tmp_path)
    assert set(sys.modules) == before


def test_missing_source_not_reconstructed(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_generator(tmp_path)
