"""Derive an independent launcher with detached work and continuous retrieval."""
from pathlib import Path

root=Path(__file__).resolve().parents[2]
source=(root/'scripts/nanbeige/launch_prime_q4_answers.sh').read_text()
source=source.replace('--image ubuntu_22_cuda_12','--image "${NANBEIGE_IMAGE:-ubuntu_22_cuda_12}"')
source=source.replace("if profile == 'massed_ada' and region == 'US':",
    "if profile == 'massed_a6000' and region == 'US':\n    provider, gpu, stock = 'massedcompute', 'A6000 48GB', ('Available',)\nelif profile == 'massed_ada' and region == 'US':")
source=source.replace('CREATE_ATTEMPTED=0','CREATE_ATTEMPTED=0\nREMOTE_READY=0')
old='  if [[ -n "$POD_ID" ]]; then\n    local stopped=0'
assert source.count(old)==1
source=source.replace(old,'''  if [[ "$REMOTE_READY" == 1 ]]; then
    timeout 200s python3 "$RUN/monitor_prime_q4.py" "$RUN" --backup-only > "$RUN/final_backup.log" 2>&1
    backup_code=$?
    printf '{"final_backup_exit_code":%s}\\n' "$backup_code" > "$RUN/final_backup.json"
  fi
  if [[ -n "$POD_ID" ]]; then
    local stopped=0''')
source=source.replace("trap 'exit 143' TERM","trap 'exit 143' TERM\ntrap 'exit 129' HUP")
source=source.replace('cp "$0" "$RUN/launcher.sh"',
    'cp "$0" "$RUN/launcher.sh"\ncp "$ROOT/scripts/nanbeige/monitor_prime_q4.py" "$RUN/monitor_prime_q4.py"')
start=source.index('echo "Starting experiment mode: $MODE"')
source=source[:start]+'''REMOTE_READY=1
echo "Starting detached experiment with incremental backups"
ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "$SSH_HOST" \\
  "cd $REMOTE && tar --no-same-owner -xzf bundle.tar.gz && python3 scripts/nanbeige/start_prime_q4_worker.py"
python3 "$RUN/monitor_prime_q4.py" "$RUN"
echo "All 70 outputs retrieved and verified before termination"
'''
(root/'scripts/nanbeige/launch_prime_q4_durable.sh').write_text(source,encoding='utf-8',newline='\n')
print('Durable launcher generated')
