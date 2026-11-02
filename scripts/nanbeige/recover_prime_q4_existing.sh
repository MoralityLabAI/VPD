#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/mnt/c/projects/VPD
RUN=$ROOT/artifacts/nanbeige_reasoning/prime-20260908T140712Z-6341
POD=8a6f2ebc1a01453caabaac4358cf7eb7
CONTROLLER=6341
PRIME=/home/duganist/.local/bin/prime
if [[ "${1:-}" != --bounded ]]; then
  REMAINING=$(python3 - <<'PY'
import datetime,time
start=datetime.datetime(2026,9,8,14,7,12,tzinfo=datetime.timezone.utc).timestamp()
remaining=int(start+28800-time.time()-180)
assert 600<remaining<28800
print(remaining)
PY
)
  exec timeout --signal=TERM --kill-after=120s "${REMAINING}s" bash "$0" --bounded
fi
CMD=$(tr '\0' ' ' < /proc/$CONTROLLER/cmdline)
[[ "$CMD" == 'bash /mnt/c/projects/VPD/artifacts/nanbeige_reasoning/launcher_snapshots/20260908T140712Z-6337.sh --bounded ' ]] || exit 2
OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$RUN/known_hosts" -i /home/duganist/.ssh/id_rsa)
cleanup() {
  CODE=$?
  trap - EXIT INT TERM
  set +e
  printf '{"recovery_exit_code":%s,"original_deadline_preserved":true}\n' "$CODE" > "$RUN/recovery_exit.json"
  kill -CONT "$CONTROLLER"
  for attempt in {1..60}; do
    [[ -e "$RUN/termination.json" ]] && break
    sleep 2
  done
  if [[ ! -e "$RUN/termination.json" ]]; then
    "$PRIME" --plain pods terminate "$POD" --yes > "$RUN/recovery_termination.log" 2>&1
  fi
  exit "$CODE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
kill -STOP "$CONTROLLER"
cp "$ROOT/scripts/nanbeige/recover_prime_q4_remote.sh" "$RUN/recover_prime_q4_remote.sh"
scp "${OPTS[@]}" "$RUN/recover_prime_q4_remote.sh" ubuntu@216.81.248.31:/home/ubuntu/nanbeige-reasoning/recover_prime_q4_remote.sh
ssh "${OPTS[@]}" -o ServerAliveInterval=20 ubuntu@216.81.248.31 \
  'bash /home/ubuntu/nanbeige-reasoning/recover_prime_q4_remote.sh' > "$RUN/recovery_remote.log" 2>&1
