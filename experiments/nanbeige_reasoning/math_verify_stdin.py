"""Feed the unchanged published math verifier without OS command-line limits."""
import json
from pathlib import Path
import runpy
import sys


def main():
    payload=json.load(sys.stdin)
    if set(payload)!={'gold','prediction','timeout'}:
        raise ValueError('Unexpected math-verifier input fields')
    if not isinstance(payload['gold'],str) or not isinstance(payload['prediction'],str):
        raise TypeError('Gold and prediction must be text')
    if type(payload['timeout']) is not int or payload['timeout'] not in (0,5):
        raise ValueError('Unsupported inner timeout')
    script=Path('artifacts/nanbeige_reasoning/sources/research_environments/environments/math/i3_math/i3_math/verify.py')
    # Set Python argv after process creation. These values never become shell
    # code or part of the operating system's bounded command line.
    sys.argv=[str(script),payload['gold'],payload['prediction'],str(payload['timeout'])]
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':main()
