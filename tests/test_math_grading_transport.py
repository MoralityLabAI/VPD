import subprocess
import sys
from experiments.nanbeige_reasoning.scoring import score,ENV


def test_stdin_transport_matches_published_verifier():
    for prediction in (r'\boxed{2}',r'\boxed{3}',r'<think>unfinished \boxed{2}'):
        direct=subprocess.run([sys.executable,'-X','utf8',str(ENV/'math/i3_math/i3_math/verify.py'),
            '2',prediction,'0'],capture_output=True,text=True,encoding='utf-8',check=True,timeout=30)
        actual=score(dict(scorer='intellect3_math',gold='2'),prediction)
        assert actual==dict(status='graded',score=float(direct.stdout.strip().splitlines()[-1]))


def test_long_math_response_preserves_answer_and_thinking_boundary():
    reasoning='Long reasoning sentence. '*2500
    grader=dict(scorer='intellect3_math',gold='2')
    assert score(grader,'<think>'+reasoning+r'</think>\boxed{2}')==dict(status='graded',score=1.)
    assert score(grader,'<think>'+reasoning+r'</think>\boxed{3}')==dict(status='graded',score=0.)
    assert score(grader,'<think>'+reasoning+r'\boxed{2}')==dict(status='graded',score=0.)
