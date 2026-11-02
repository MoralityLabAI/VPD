#!/usr/bin/env bash
# One temporary pod, bounded from before creation through artifact retrieval.
set -Eeuo pipefail
if [[ "${NANBEIGE_LAUNCHER_FROZEN:-0}" != 1 ]]; then
  SNAPSHOT_DIR=/mnt/c/projects/VPD/artifacts/nanbeige_reasoning/launcher_snapshots
  mkdir -p "$SNAPSHOT_DIR"
  SNAPSHOT="$SNAPSHOT_DIR/$(date -u +%Y%m%dT%H%M%SZ)-$$.sh"
  cp "$0" "$SNAPSHOT"
  export NANBEIGE_LAUNCHER_FROZEN=1
  exec bash "$SNAPSHOT" "$@"
fi
if [[ "${1:-}" != --bounded ]]; then
  LAUNCH_BOUND="${NANBEIGE_LAUNCH_BOUND_SECONDS:-28800}"
  [[ "$LAUNCH_BOUND" =~ ^[0-9]+$ && "$LAUNCH_BOUND" -ge 600 && "$LAUNCH_BOUND" -le 28800 ]] || { echo 'Invalid launch time bound'; exit 2; }
  exec timeout --signal=TERM --kill-after=120s "${LAUNCH_BOUND}s" bash "$0" --bounded
fi
ROOT=/mnt/c/projects/VPD
PRIME=/home/duganist/.local/bin/prime
PRIME_PYTHON=/home/duganist/.local/share/uv/tools/prime/bin/python3
KEY=/home/duganist/.ssh/id_rsa
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
NAME="nanbeige-reasoning-${STAMP}"
RUN="${ROOT}/artifacts/nanbeige_reasoning/prime-${STAMP}"
REMOTE=/home/ubuntu/nanbeige-reasoning
MODE="prime-q4-answers"
BUNDLE=prime_q4_answers_bundle
REMOTE_ENTRY="scripts/nanbeige/run_prime_q4_answers_remote.sh"
POD_ID=""
CREATE_ATTEMPTED=0
REMOTE_READY=0
mkdir -p "$RUN"
echo "Multipass launch directory: $RUN"
cd "$ROOT"
cp "$ROOT/artifacts/nanbeige_reasoning/$BUNDLE.tar.gz" "$RUN/bundle.tar.gz"
cp "$ROOT/artifacts/nanbeige_reasoning/$BUNDLE.tar.receipt.json" "$RUN/bundle.receipt.json"
cp "$0" "$RUN/launcher.sh"
cp "$ROOT/scripts/nanbeige/monitor_prime_q4.py" "$RUN/monitor_prime_q4.py"
python3 - "$RUN" <<'PY'
import hashlib, json, pathlib, sys
run = pathlib.Path(sys.argv[1])
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
receipt = json.loads((run/'bundle.receipt.json').read_text())
if digest(run/'bundle.tar.gz') != receipt['sha256']:
    raise SystemExit('Frozen upload bundle hash differs from receipt')
(run/'launch_inputs.json').write_text(json.dumps({
    'bundle_sha256': receipt['sha256'], 'launcher_sha256': digest(run/'launcher.sh'),
    'launcher_immutable_snapshot': True}, indent=2))
PY
export PRIME_DISABLE_VERSION_CHECK=1

[[ -s "$KEY" ]] || { echo 'Existing WSL SSH key is missing'; exit 1; }

identify_pod() {
  "$PRIME" --plain pods list --output json > "$RUN/pods.json"
  python3 - "$RUN/pods.json" "$NAME" <<'PY'
import json, sys
rows = [r for r in json.load(open(sys.argv[1]))['pods'] if r.get('name') == sys.argv[2]]
if len(rows) > 1:
    raise SystemExit('Ambiguous newly-created pod name')
print(rows[0]['id'] if rows else '')
PY
}
cleanup() {
  local original=$?
  trap - EXIT INT TERM
  set +e
  if [[ -s "$RUN/ssh_selection.json" ]]; then
    "$PRIME_PYTHON" -I "$ROOT/scripts/nanbeige/prime_ssh_selection.py" restore "$RUN/ssh_selection.json" > "$RUN/ssh_restore.log" 2>&1
  fi
  if [[ -z "$POD_ID" && "$CREATE_ATTEMPTED" == 1 ]]; then
    POD_ID="$(identify_pod)"
  fi
  if [[ "$REMOTE_READY" == 1 ]]; then
    timeout 200s python3 "$RUN/monitor_prime_q4.py" "$RUN" --backup-only > "$RUN/final_backup.log" 2>&1
    backup_code=$?
    printf '{"final_backup_exit_code":%s}\n' "$backup_code" > "$RUN/final_backup.json"
  fi
  if [[ -n "$POD_ID" ]]; then
    local stopped=0
    for attempt in 1 2 3; do
      if "$PRIME" --plain pods terminate "$POD_ID" --yes > "$RUN/termination.log" 2>&1; then
        stopped=1
        break
      fi
      sleep 3
    done
    printf '{"pod_id":"%s","termination_command_succeeded":%s,"run_exit_code":%s}\n' \
      "$POD_ID" "$([[ "$stopped" == 1 ]] && echo true || echo false)" "$original" > "$RUN/termination.json"
    echo "Pod termination receipt: $RUN/termination.json"
    [[ "$stopped" == 1 ]] || { echo 'ATTENTION: pod termination failed'; exit 3; }
  fi
  exit "$original"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

"$PRIME" --plain availability list --gpu-count 1 --no-group-similar --output json > "$RUN/availability.json"
CHOICE_ID="$(python3 - "$RUN/availability.json" "$RUN/choice.json" "${NANBEIGE_REASONING_REGION:-US}" "${NANBEIGE_COMPUTE_PROFILE:-massed_ada}" "${NANBEIGE_MAX_HOURLY:-0.75}" <<'PY'
import json, math, sys
region = sys.argv[3]
profile = sys.argv[4]
ceiling = float(sys.argv[5])
if not math.isfinite(ceiling) or not 0 < ceiling <= 2.00:
    raise SystemExit('Invalid approved hourly ceiling')
if profile == 'massed_a6000' and region == 'US':
    provider, gpu, stock = 'massedcompute', 'A6000 48GB', ('Available',)
elif profile == 'massed_ada' and region == 'US':
    provider, gpu, stock = 'massedcompute', 'RTX6000Ada 48GB', ('Available',)
elif profile == 'lambda_a100' and region in ('US', 'IN'):
    provider, gpu, stock = 'lambdalabs', 'A100 40GB', ('Available',)
elif profile == 'runpod_a40' and region in ('CA', 'SE'):
    provider, gpu, stock = 'runpod', 'A40 48GB', ('Available', 'Low', 'High')
elif profile == 'runpod_a100' and region == 'US':
    provider, gpu, stock = 'runpod', 'A100 80GB', ('Available', 'Low', 'High')
elif profile == 'runpod_ada' and region == 'US':
    provider, gpu, stock = 'runpod', 'RTX6000Ada 48GB', ('Available', 'Low', 'High')
else:
    raise SystemExit('Unsupported experimental compute profile/region')
data = json.load(open(sys.argv[1], encoding='utf-8-sig'))
rows = [r for r in data['gpu_resources'] if r.get('stock_status') in stock
        and int(r['gpu_count']) == 1 and float(r['gpu_memory']) >= 24
        and float(r['price_value']) <= ceiling and not r.get('is_spot')
        and r.get('provider') == provider and r['gpu_type'] == gpu and r['location'] == region]
if not rows:
    raise SystemExit(f'No available {region} {provider} {gpu} below the experimental hourly ceiling')
choice = min(rows, key=lambda r: float(r['price_value']))
json.dump(choice, open(sys.argv[2], 'w'), indent=2)
print(choice['id'])
PY
)"
# Serialize the account-wide primary SSH-key selection during provisioning.
# Release after restoration so separate authorized experiments can run together.
exec 9>"$ROOT/artifacts/nanbeige_reasoning/provisioning.lock"
flock -x 9
"$PRIME_PYTHON" -I "$ROOT/scripts/nanbeige/prime_ssh_selection.py" select "$RUN/ssh_selection.json"
CREATE_ATTEMPTED=1
CREATE_OPTIONS=()
if [[ "${NANBEIGE_COMPUTE_PROFILE:-massed_ada}" == runpod_* ]]; then
  read -r CREATE_VCPUS CREATE_MEMORY < <(python3 - "$RUN/choice.json" <<'PY'
import json, re, sys
choice = json.load(open(sys.argv[1]))
def minimum_resource(value):
    match = re.fullmatch(r'(\d+)(?:-(\d+))?', str(value).strip())
    if not match:
        raise SystemExit('Unsupported advertised resource quantity')
    lower = int(match[1])
    if match[2] and int(match[2]) < lower:
        raise SystemExit('Invalid advertised resource range')
    return lower
cpu, memory = minimum_resource(choice['vcpus']), minimum_resource(choice['memory_gb'])
if cpu < 1 or memory < 16:
    raise SystemExit('Invalid advertised CPU/memory resources')
print(cpu, memory)
PY
)
  CREATE_OPTIONS+=(--disk-size 120 --vcpus "$CREATE_VCPUS" --memory "$CREATE_MEMORY")
fi
"$PRIME" --plain pods create --id "$CHOICE_ID" --name "$NAME" \
  --image "${NANBEIGE_IMAGE:-ubuntu_22_cuda_12}" "${CREATE_OPTIONS[@]}" --yes > "$RUN/create.log" 2>&1
for attempt in {1..12}; do
  POD_ID="$(identify_pod)"
  [[ -n "$POD_ID" ]] && break
  sleep 5
done
[[ -n "$POD_ID" ]] || { echo 'New pod could not be identified'; exit 1; }
printf '%s\n' "$POD_ID" > "$RUN/pod_id.txt"
echo "Created multipass pod: $POD_ID"

READINESS_STARTED=$SECONDS
READINESS_LIMIT=2400
SSH_READY=0
while (( SECONDS - READINESS_STARTED < READINESS_LIMIT )); do
  "$PRIME" --plain pods status "$POD_ID" --output json > "$RUN/status.json"
  python3 - "$RUN/status.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if str(r.get('status','')).upper() in {'FAILED','ERROR','STOPPED','TERMINATED'}:
    raise SystemExit('Provisioning entered a terminal failure state')
PY
  if python3 - "$RUN/status.json" > "$RUN/ssh_target.txt" <<'PY'
import json, re, shlex, sys
parts = shlex.split(json.load(open(sys.argv[1])).get('ssh', ''))
hosts = [s for s in parts if re.fullmatch(r'[A-Za-z0-9._-]+@[A-Za-z0-9.-]+', s)]
if len(hosts) != 1:
    raise SystemExit(1)
port = parts[parts.index('-p') + 1] if '-p' in parts else '22'
if not port.isdigit() or not 1 <= int(port) <= 65535:
    raise SystemExit('Invalid SSH port')
print(hosts[0], port)
PY
  then SSH_READY=1; break; fi
  sleep 10
done
if [[ "$SSH_READY" != 1 ]]; then
  printf '{"status":"ssh_readiness_timeout","waited_seconds":%s,"limit_seconds":%s}\n' \
    "$((SECONDS - READINESS_STARTED))" "$READINESS_LIMIT" > "$RUN/readiness_timeout.json"
  echo 'SSH target absent after the explicit 40-minute readiness bound'
  exit 1
fi
read -r SSH_HOST SSH_PORT < "$RUN/ssh_target.txt"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new
          -o "UserKnownHostsFile=$RUN/known_hosts" -i "$KEY")
for attempt in {1..24}; do
  if ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "$SSH_HOST" "mkdir -p $REMOTE" 2> "$RUN/ssh_connect.log"; then
    break
  fi
  [[ "$attempt" != 24 ]] || { echo 'SSH did not become ready'; exit 1; }
  sleep 5
done
"$PRIME_PYTHON" -I "$ROOT/scripts/nanbeige/prime_ssh_selection.py" restore "$RUN/ssh_selection.json"
flock -u 9
scp "${SSH_OPTS[@]}" -P "$SSH_PORT" "$RUN/bundle.tar.gz" "$SSH_HOST:$REMOTE/bundle.tar.gz"
REMOTE_READY=1
echo "Starting detached experiment with incremental backups"
ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "$SSH_HOST" \
  "cd $REMOTE && tar --no-same-owner -xzf bundle.tar.gz && python3 scripts/nanbeige/start_prime_q4_worker.py"
python3 "$RUN/monitor_prime_q4.py" "$RUN"
echo "All 70 outputs retrieved and verified before termination"
