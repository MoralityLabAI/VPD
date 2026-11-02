#!/usr/bin/env bash
set -euo pipefail
RUN=/mnt/c/projects/VPD/artifacts/nanbeige_reasoning/prime-20260908T140712Z-6341
REMOTE=/home/ubuntu/nanbeige-reasoning/artifacts/nanbeige_reasoning
SNAPSHOT=$RUN/startup_snapshot
mkdir -p "$SNAPSHOT/run" "$SNAPSHOT/context"
OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$RUN/known_hosts" -i /home/duganist/.ssh/id_rsa)
for name in contract.json runtime.json 00_baseline.json; do
  scp "${OPTS[@]}" "ubuntu@216.81.248.31:$REMOTE/prime_q4_feature_answers_v1/$name" "$SNAPSHOT/run/$name"
done
scp "${OPTS[@]}" "ubuntu@216.81.248.31:$REMOTE/q4_answer_context_v1/verification.json" "$SNAPSHOT/context/verification.json"
