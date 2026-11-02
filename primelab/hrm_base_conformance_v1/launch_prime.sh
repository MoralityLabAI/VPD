#!/usr/bin/env bash
set -Eeuo pipefail

PRIME_BIN="${PRIME_BIN:-/home/duganist/.local/bin/prime}"
LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/projects/VPD/primelab/hrm_base_conformance_v1}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="${RUN_ID:-PL-HRM-BASE-CONFORMANCE-V1-${RUN_STAMP}}"
POD_NAME="${POD_NAME:-hrm-base-conf-${RUN_STAMP}}"
POD_IMAGE="${POD_IMAGE:-cuda_12_4_pytorch_2_4}"
POD_DISK_GB="${POD_DISK_GB:-40}"
PEM_PATH="${PEM_PATH:-/home/duganist/.ssh/id_rsa}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/hrm-base-conformance-v1}"
MAX_PRICE_PER_HOUR="${MAX_PRICE_PER_HOUR:-1.30}"
MAX_TOTAL_SECONDS="${MAX_TOTAL_SECONDS:-3600}"
LAUNCH_DIR="${LOCAL_ROOT}/launches/${RUN_ID}"
RESULT_DIR="${LOCAL_ROOT}/results/${RUN_ID}"
BUNDLE="${LAUNCH_DIR}/bundle.tar.gz"
POD_ID=""
TERMINATED=0
export RUN_ID

mkdir -p "${LAUNCH_DIR}" "${RESULT_DIR}"

cleanup() {
  local original_exit=$?
  if [[ -n "${POD_ID}" && "${TERMINATED}" -eq 0 ]]; then
    set +e
    "${PRIME_BIN}" --plain pods terminate "${POD_ID}" --yes \
      > "${LAUNCH_DIR}/termination_stdout.log" \
      2> "${LAUNCH_DIR}/termination_stderr.log"
    local termination_exit=$?
    set -e
    if [[ "${termination_exit}" -eq 0 ]]; then
      TERMINATED=1
    fi
    python3 -c 'import json,sys; json.dump({"pod_id":sys.argv[1],"termination_exit_code":int(sys.argv[2]),"terminated":bool(int(sys.argv[3])),"launcher_exit_before_cleanup":int(sys.argv[4])},open(sys.argv[5],"w"),indent=2,sort_keys=True)' \
      "${POD_ID}" "${termination_exit}" "${TERMINATED}" "${original_exit}" \
      "${LAUNCH_DIR}/pod_termination_receipt.json"
  fi
  return "${original_exit}"
}
trap cleanup EXIT INT TERM

if [[ ! -x "${PRIME_BIN}" ]]; then
  echo "Prime CLI missing: ${PRIME_BIN}" >&2
  exit 1
fi
if [[ ! -f "${PEM_PATH}" ]]; then
  echo "Prime SSH key missing: ${PEM_PATH}" >&2
  exit 1
fi

"${PRIME_BIN}" --plain whoami > "${LAUNCH_DIR}/whoami.txt"
"${PRIME_BIN}" --plain wallet --limit 5 --output json > "${LAUNCH_DIR}/wallet_before.json"
"${PRIME_BIN}" --plain pods list --output json > "${LAUNCH_DIR}/pods_before.json"
"${PRIME_BIN}" --plain availability list --gpu-type A10_24GB --gpu-count 1 --output json \
  > "${LAUNCH_DIR}/availability.json"

GPU_CHOICE_ID="$(
  MAX_PRICE="${MAX_PRICE_PER_HOUR}" python3 -c 'import json,os,sys; rows=json.load(open(sys.argv[1]))["gpu_resources"]; rows=[r for r in rows if r.get("stock_status")=="Available" and float(r["price_value"])<=float(os.environ["MAX_PRICE"])]; rows.sort(key=lambda r:(float(r["price_value"]),r["id"])); print(rows[0]["id"] if rows else "")' \
    "${LAUNCH_DIR}/availability.json"
)"
if [[ -z "${GPU_CHOICE_ID}" ]]; then
  echo "No eligible A10 24 GB resource within price cap" >&2
  exit 1
fi

python3 -c 'import hashlib,json,os,sys; from pathlib import Path; root=Path(sys.argv[1]); names={"remote_conformance.py","run_remote.sh","requirements.txt","probe_pack.json"}; paths=[p for p in root.rglob("*") if p.is_file() and (p.name in names or ("fused_repaired_composite" in p.parts and p.suffix==".py"))]; files={str(p.relative_to(root)):{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)}; json.dump({"run_id":os.environ["RUN_ID"],"files":files},open(sys.argv[2],"w"),indent=2,sort_keys=True)' \
  "${LOCAL_ROOT}" "${LAUNCH_DIR}/bundle_manifest.json"

tar -C "${LOCAL_ROOT}" -czf "${BUNDLE}" \
  remote_conformance.py run_remote.sh requirements.txt probe_pack.json \
  fused_repaired_composite/__init__.py \
  fused_repaired_composite/configuration_hrm_text.py \
  fused_repaired_composite/modeling_hrm_text.py
sha256sum "${BUNDLE}" > "${LAUNCH_DIR}/bundle.sha256"

"${PRIME_BIN}" --plain pods create \
  --id "${GPU_CHOICE_ID}" \
  --name "${POD_NAME}" \
  --disk-size "${POD_DISK_GB}" \
  --image "${POD_IMAGE}" \
  --yes \
  > "${LAUNCH_DIR}/create_stdout.log" \
  2> "${LAUNCH_DIR}/create_stderr.log"

for _ in $(seq 1 18); do
  "${PRIME_BIN}" --plain pods list --output json > "${LAUNCH_DIR}/pods_after.json"
  POD_ID="$(POD_NAME_VALUE="${POD_NAME}" python3 -c 'import json,os,sys; rows=[r for r in json.load(open(sys.argv[1]))["pods"] if r.get("name")==os.environ["POD_NAME_VALUE"]]; print(rows[0]["id"] if len(rows)==1 else "")' "${LAUNCH_DIR}/pods_after.json")"
  if [[ -n "${POD_ID}" ]]; then
    break
  fi
  sleep 5
done
if [[ -z "${POD_ID}" ]]; then
  echo "Could not identify newly created pod" >&2
  exit 1
fi
export POD_ID
printf '%s\n' "${POD_ID}" > "${LAUNCH_DIR}/pod_id.txt"

SSH_CONN=""
for _ in $(seq 1 90); do
  "${PRIME_BIN}" --plain pods status "${POD_ID}" --output json > "${LAUNCH_DIR}/pod_status.json"
  SSH_CONN="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("ssh",""))' "${LAUNCH_DIR}/pod_status.json")"
  if [[ -n "${SSH_CONN}" && "${SSH_CONN}" != "N/A" ]]; then
    break
  fi
  sleep 10
done
if [[ -z "${SSH_CONN}" || "${SSH_CONN}" == "N/A" ]]; then
  echo "SSH connection was not published" >&2
  exit 1
fi

SSH_HOST="${SSH_CONN%% -p *}"
SSH_PORT="${SSH_CONN##* -p }"
if [[ "${SSH_PORT}" == "${SSH_CONN}" ]]; then
  SSH_PORT=22
fi
chmod 600 "${PEM_PATH}"

ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
  -i "${PEM_PATH}" -p "${SSH_PORT}" "${SSH_HOST}" \
  "mkdir -p '${REMOTE_ROOT}'"
scp -o StrictHostKeyChecking=no -i "${PEM_PATH}" -P "${SSH_PORT}" \
  "${BUNDLE}" "${SSH_HOST}:${REMOTE_ROOT}/bundle.tar.gz"
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
  -i "${PEM_PATH}" -p "${SSH_PORT}" "${SSH_HOST}" \
  "cd '${REMOTE_ROOT}' && tar --no-same-owner -xzf bundle.tar.gz && chmod +x run_remote.sh"

set +e
timeout --signal=TERM --kill-after=90s "${MAX_TOTAL_SECONDS}s" \
  ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    -i "${PEM_PATH}" -p "${SSH_PORT}" "${SSH_HOST}" \
    "cd '${REMOTE_ROOT}' && RUN_ID='${RUN_ID}' REMOTE_ROOT='${REMOTE_ROOT}' bash run_remote.sh"
REMOTE_EXIT=$?
set -e
printf '%s\n' "${REMOTE_EXIT}" > "${LAUNCH_DIR}/remote_exit_code.txt"

ssh -o StrictHostKeyChecking=no -i "${PEM_PATH}" -p "${SSH_PORT}" "${SSH_HOST}" \
  "cd '${REMOTE_ROOT}' && tar -czf results.tar.gz results"
scp -o StrictHostKeyChecking=no -i "${PEM_PATH}" -P "${SSH_PORT}" \
  "${SSH_HOST}:${REMOTE_ROOT}/results.tar.gz" "${RESULT_DIR}/results.tar.gz"
tar -C "${RESULT_DIR}" --strip-components=1 -xzf "${RESULT_DIR}/results.tar.gz"

set +e
"${PRIME_BIN}" --plain pods terminate "${POD_ID}" --yes \
  > "${LAUNCH_DIR}/termination_stdout.log" \
  2> "${LAUNCH_DIR}/termination_stderr.log"
TERMINATION_EXIT=$?
set -e
if [[ "${TERMINATION_EXIT}" -eq 0 ]]; then
  TERMINATED=1
fi
python3 -c 'import json,sys; json.dump({"pod_id":sys.argv[1],"termination_exit_code":int(sys.argv[2]),"terminated":bool(int(sys.argv[3]))},open(sys.argv[4],"w"),indent=2,sort_keys=True)' \
  "${POD_ID}" "${TERMINATION_EXIT}" "${TERMINATED}" \
  "${LAUNCH_DIR}/pod_termination_receipt.json"

"${PRIME_BIN}" --plain wallet --limit 10 --output json > "${LAUNCH_DIR}/wallet_after.json"

python3 -c 'import json,os,sys; from pathlib import Path; payload={"run_id":os.environ["RUN_ID"],"pod_id":os.environ["POD_ID"],"remote_exit_code":int(sys.argv[1]),"result_dir":sys.argv[2]}; Path(sys.argv[3]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")' \
  "${REMOTE_EXIT}" "${RESULT_DIR}" "${LAUNCH_DIR}/launch_receipt.json"

if [[ "${TERMINATION_EXIT}" -ne 0 ]]; then
  echo "Pod termination failed" >&2
  exit 3
fi
if [[ "${REMOTE_EXIT}" -ne 0 ]]; then
  exit 2
fi
echo "RESULT_DIR=${RESULT_DIR}"
