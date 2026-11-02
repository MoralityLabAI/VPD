import random
import numpy as np
from experiments.nanbeige_reasoning.build_campsite_support import construct,symmetry_key
from experiments.nanbeige_reasoning.audit_hermes import public_rule
from experiments.nanbeige_reasoning.verify_hermes_controller_control import validate


def test_rectangular_symmetry_constraints_follow_the_solution():
    rng=random.Random(42)
    public,solution=next(candidate for _ in range(100)
        if (candidate:=construct(rng,5,4)) is not None)
    expected=symmetry_key(public['grid'],public['row_constraints'],public['col_constraints'])
    for flip in (False,True):
        for turns in range(4):
            grid=np.rot90(np.fliplr(public['grid']) if flip else public['grid'],turns)
            solved=np.rot90(np.fliplr(solution) if flip else solution,turns)
            rows=(solved=='C').sum(axis=1).tolist()
            columns=(solved=='C').sum(axis=0).tolist()
            assert symmetry_key(grid.tolist(),rows,columns)==expected


def test_constructed_solution_is_valid_and_seeded():
    first,second=random.Random(204),random.Random(204)
    for _ in range(30):
        result=construct(first,6,5)
        assert result==construct(second,6,5)
        if result is not None:
            public,solution=result
            assert public_rule(solution,public,True)
            assert not public_rule(public['grid'],public,True)


def test_matching_rejects_shared_tree_despite_local_adjacency():
    grid=[list('CTCXX'),list('XXXXT'),list('XXXTC')]
    public=dict(grid=[[('X' if v=='C' else v) for v in row] for row in grid],
        row_constraints=[2,0,1],col_constraints=[1,0,1,0,1],n=3,m=5)
    assert public_rule(grid,public,False)
    assert validate(public,grid)==(False,False)


def test_literal_per_tent_rule_does_not_imply_perfect_matching():
    grid=[list('XXTXC'),list('XXXXT'),list('XCTCX')]
    public=dict(grid=[[('X' if v=='C' else v) for v in row] for row in grid],
        row_constraints=[1,0,2],col_constraints=[0,1,0,1,1],n=3,m=5)
    assert public_rule(grid,public,True)
    assert validate(public,grid)==(False,True)
