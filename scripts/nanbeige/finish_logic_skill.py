"""Grade the backed-up experiment after its bounded controller terminates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    run=args.run;started=time.monotonic()
    while time.monotonic()-started<22200:
        end=run/'termination.json'
        if end.exists():
            term=json.loads(end.read_text())
            if not term['termination_command_succeeded']:
                raise RuntimeError('Controller reports termination failure')
            if not (run/'completion_verification.json').exists():
                raise RuntimeError('Controller stopped without a verified completed experiment')
            output=run/'retrieved/artifacts/nanbeige_reasoning/logic_skill_comparison_outputs_v1'
            subprocess.run([sys.executable,'-m','experiments.nanbeige_reasoning.grade_logic_skill_comparison',
                            '--run',str(output),'--output',str(run/'grading')],check=True)
            (run/'postprocess_complete.json').write_text(json.dumps(dict(status='graded',updated_unix=time.time()))+'\n')
            return
        time.sleep(30)
    raise TimeoutError('Postprocessing wait limit')


if __name__=='__main__':main()
