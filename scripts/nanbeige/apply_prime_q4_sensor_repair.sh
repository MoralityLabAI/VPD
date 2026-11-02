#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/c/projects/VPD
RUN=$ROOT/artifacts/nanbeige_reasoning/prime-20260908T140712Z-6341
REMOTE=/home/ubuntu/nanbeige-reasoning
OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$RUN/known_hosts" -i /home/duganist/.ssh/id_rsa)
cp "$ROOT/artifacts/nanbeige_reasoning/prime_q4_sensor_patch.tar.gz" "$RUN/sensor_repair.tar.gz"
scp "${OPTS[@]}" "$RUN/sensor_repair.tar.gz" "$ROOT/scripts/nanbeige/repair_prime_q4_sensor.py" ubuntu@216.81.248.31:$REMOTE/
ssh "${OPTS[@]}" ubuntu@216.81.248.31 "python3 $REMOTE/repair_prime_q4_sensor.py" > "$RUN/sensor_repair.log" 2>&1
