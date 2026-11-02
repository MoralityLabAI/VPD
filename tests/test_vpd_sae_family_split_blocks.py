import pytest

from experiments.authz_mechinterp.audit_sae_family_split_blocks_v1 import question_digest, split_block, validate_role_blocks


def test_source_json_string_hash_recipe():
    import hashlib
    assert question_digest("a\nb") == hashlib.sha256(b'"a\\nb"').hexdigest()
    assert question_digest("a\nb") != hashlib.sha256(b'a\nb').hexdigest()
    assert question_digest("é") == hashlib.sha256('"é"'.encode()).hexdigest()


@pytest.mark.parametrize("pair", [("dyck_language", "dyck_language_errors"), ("word_sorting", "word_sorting_mistake")])
def test_related_variants_cannot_cross_roles(pair):
    assert split_block(pair[0]) == split_block(pair[1])
    with pytest.raises(ValueError, match="crosses roles"):
        validate_role_blocks([dict(family=pair[0], split="train"), dict(family=pair[1], split="eval")])


def test_coassignment_and_order_invariance():
    rows = [dict(family=f, split="train") for f in ("dyck_language", "dyck_language_errors")]
    assert validate_role_blocks(rows) == validate_role_blocks(rows[::-1])
    assert len(validate_role_blocks(rows)) == 1


def test_unknown_family_remains_explicitly_unreviewed():
    assert split_block("graph_coloring") == "unreviewed:graph_coloring"


def test_same_family_cannot_cross_roles():
    with pytest.raises(ValueError):
        validate_role_blocks([dict(family="math", split="train"), dict(family="math", split="eval")])


def test_invalid_role_rejected():
    with pytest.raises(ValueError):
        validate_role_blocks([dict(family="math", split="quarantine")])
