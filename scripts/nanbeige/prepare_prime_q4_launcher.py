"""Specialize the tested bounded launcher for one Massed Compute Ada offer."""
from pathlib import Path


def replace(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def main():
    root=Path(__file__).resolve().parents[2]
    text=(root/'scripts/nanbeige/launch_reasoning.sh').read_text()
    text=replace(text,'--image cuda_12_4_pytorch_2_4','--image ubuntu_22_cuda_12')
    text=replace(text,'REMOTE=/workspace/nanbeige-reasoning','REMOTE=/home/ubuntu/nanbeige-reasoning')
    text=replace(text,'MODE="${NANBEIGE_REASONING_MODE:-capture}"','MODE="prime-q4-answers"')
    start=text.index('case "$MODE" in');end=text.index('\nesac',start)+len('\nesac')
    text=text[:start]+'''BUNDLE=prime_q4_answers_bundle
REMOTE_ENTRY="scripts/nanbeige/run_prime_q4_answers_remote.sh"'''+text[end:]
    text=replace(text,'${NANBEIGE_COMPUTE_PROFILE:-lambda_a100}','${NANBEIGE_COMPUTE_PROFILE:-massed_ada}') if text.count('${NANBEIGE_COMPUTE_PROFILE:-lambda_a100}')==1 else text.replace('${NANBEIGE_COMPUTE_PROFILE:-lambda_a100}','${NANBEIGE_COMPUTE_PROFILE:-massed_ada}')
    text=text.replace('${NANBEIGE_MAX_HOURLY:-2.00}','${NANBEIGE_MAX_HOURLY:-0.75}')
    text=replace(text,"if profile == 'lambda_a100' and region in ('US', 'IN'):",
        "if profile == 'massed_ada' and region == 'US':\n    provider, gpu, stock = 'massedcompute', 'RTX6000Ada 48GB', ('Available',)\nelif profile == 'lambda_a100' and region in ('US', 'IN'):")
    text=replace(text,'  if python3 - "$RUN/status.json" > "$RUN/ssh_target.txt"',
        '''  python3 - "$RUN/status.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if str(r.get('status','')).upper() in {'FAILED','ERROR','STOPPED','TERMINATED'}:
    raise SystemExit('Provisioning entered a terminal failure state')
PY
  if python3 - "$RUN/status.json" > "$RUN/ssh_target.txt"''')
    path=root/'scripts/nanbeige/launch_prime_q4_answers.sh'
    path.write_text(text,encoding='utf-8',newline='\n')
    test=(root/'scripts/nanbeige/test_launch_lifecycle.py').read_text()
    for old,new in [
        ('launch_reasoning.sh','launch_prime_q4_answers.sh'),
        ('reasoning_bundle.tar','prime_q4_answers_bundle.tar'),
        ("'provider':'lambdalabs'","'provider':'massedcompute'"),
        ("'gpu_memory':40","'gpu_memory':48"),
        ("'price_value':1.99","'price_value':0.75"),
        ("'gpu_type':'A100 40GB'","'gpu_type':'RTX6000Ada 48GB'"),
        ("'location':'IN'","'location':'US'"),
        ("else 'IN'","else 'US'"),
        ("else 'lambda_a100'","else 'massed_ada'"),
        ("else 'lambdalabs'","else 'massedcompute'"),
        ('run_reasoning_remote.sh cache-diagnostics','run_prime_q4_answers_remote.sh'),
    ]:
        assert old in test,old
        test=test.replace(old,new)
    (root/'scripts/nanbeige/test_prime_q4_launch_lifecycle.py').write_text(test,encoding='utf-8',newline='\n')
    print(path)


if __name__=='__main__':main()
