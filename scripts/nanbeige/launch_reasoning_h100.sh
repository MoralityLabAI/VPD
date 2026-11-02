#!/usr/bin/env bash
# One temporary pod, bounded from before creation through artifact retrieval.
set -Eeuo pipefail
if [[ "${1:-}" != --bounded ]]; then
  exec timeout --signal=TERM --kill-after=120s 28800s bash "$0" --bounded
fi
ROOT=/mnt/c/projects/VPD
PRIME=/home/duganist/.local/bin/prime
PRIME_PYTHON=/home/duganist/.local/share/uv/tools/prime/bin/python3
KEY=/home/duganist/.ssh/id_rsa
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
NAME="nanbeige-reasoning-${STAMP}"
RUN="${ROOT}/artifacts/nanbeige_reasoning/prime-${STAMP}"
REMOTE=/workspace/nanbeige-reasoning
POD_ID=""
CREATE_ATTEMPTED=0
mkdir -p "$RUN"
cd "$ROOT"
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

"$PRIME" --plain availability list --gpu-count 1 --output json > "$RUN/availability.json"
CHOICE_ID="$(python3 - "$RUN/availability.json" "$RUN/choice.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding='utf-8-sig'))
rows = [r for r in data['gpu_resources'] if r.get('stock_status') == 'Available'
        and int(r['gpu_count']) == 1 and float(r['gpu_memory']) >= 24
        and float(r['price_value']) <= 3.50 and not r.get('is_spot')
        and r['gpu_type'] == 'H100 80GB' and r['location'] == 'US']
if not rows:
    raise SystemExit('No available US H100 below the retry hourly ceiling')
choice = min(rows, key=lambda r: float(r['price_value']))
json.dump(choice, open(sys.argv[2], 'w'), indent=2)
print(choice['id'])
PY
)"
echo "Multipass launch directory: $RUN"
"$PRIME_PYTHON" -I "$ROOT/scripts/nanbeige/prime_ssh_selection.py" select "$RUN/ssh_selection.json"
CREATE_ATTEMPTED=1
"$PRIME" --plain pods create --id "$CHOICE_ID" --name "$NAME" \
  --image cuda_12_4_pytorch_2_4 --yes > "$RUN/create.log" 2>&1
for attempt in {1..12}; do
  POD_ID="$(identify_pod)"
  [[ -n "$POD_ID" ]] && break
  sleep 5
done
[[ -n "$POD_ID" ]] || { echo 'New pod could not be identified'; exit 1; }
printf '%s\n' "$POD_ID" > "$RUN/pod_id.txt"
echo "Created multipass pod: $POD_ID"

for attempt in {1..120}; do
  "$PRIME" --plain pods status "$POD_ID" --output json > "$RUN/status.json"
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
  then break; fi
  sleep 10
done
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
scp "${SSH_OPTS[@]}" -P "$SSH_PORT" "$ROOT/artifacts/nanbeige_reasoning/reasoning_bundle.tar.gz" "$SSH_HOST:$REMOTE/bundle.tar.gz"
echo 'Starting reasoning generation, all-layer capture, and VPD/SAE measurements'
set +e
ssh "${SSH_OPTS[@]}" -o ServerAliveInterval=20 -p "$SSH_PORT" "$SSH_HOST" \
  "cd $REMOTE && tar --no-same-owner -xzf bundle.tar.gz && bash scripts/nanbeige/run_reasoning_remote.sh" \
  > "$RUN/remote.log" 2>&1
REMOTE_CODE=$?
set -e
printf '%s\n' "$REMOTE_CODE" > "$RUN/remote_exit_code.txt"
ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "$SSH_HOST" \
  "cd $REMOTE && tar -czf results.tar.gz artifacts" >> "$RUN/retrieval.log" 2>&1
scp "${SSH_OPTS[@]}" -P "$SSH_PORT" "$SSH_HOST:$REMOTE/results.tar.gz" "$RUN/results.tar.gz" >> "$RUN/retrieval.log" 2>&1
echo "Retrieved preflight results: $RUN/results.tar.gz"
exit "$REMOTE_CODE"
