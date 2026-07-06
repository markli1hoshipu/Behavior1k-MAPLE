#!/bin/bash
#SBATCH --job-name=b1k_eval_recovery
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --gres=gpu:rtx5080:2
#SBATCH --time=2:00:00
#SBATCH --output=/shared_work/logs/%j_%x.out
#SBATCH --error=/shared_work/logs/%j_%x.err

# Recovery eval: restore the sim to a recorded failure frame, then run our
# PI05 policy from there. Same dual-GPU layout as eval_openpi.sbatch:
#  GPU 0 = OmniGibson sim, GPU 1 = PI05 inference server.
#
# Usage:
#   PI_CKPT=<absolute_ckpt_path>  RESTORE_STATE=<absolute_json>  \
#   sbatch eval_openpi_recovery.sbatch [task_name]

set -uo pipefail

TASK_NAME="${1:-putting_shoes_on_rack}"
PI_CKPT="${PI_CKPT:?must set PI_CKPT to ckpt dir (e.g. .../ckpt-200000)}"
RESTORE_STATE="${RESTORE_STATE:?must set RESTORE_STATE to absolute JSON path}"
PI_CONFIG_NAME="${PI_CONFIG_NAME:-pi05_b1k_mp_radio}"
OPENPI_ROOT="${OPENPI_ROOT:-/shared_work/openpi}"

LOG_BASE="${LOG_BASE:-/shared_work/logs/behavior_recovery/openpi_${SLURM_JOB_ID}}"
VLA_PORT="${VLA_PORT:-$((8765 + SLURM_JOB_ID % 1000))}"
mkdir -p "${LOG_BASE}"

echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Task:          ${TASK_NAME}"
echo "pi config:     ${PI_CONFIG_NAME}"
echo "pi ckpt:       ${PI_CKPT}"
echo "Restore JSON:  ${RESTORE_STATE}"
echo "WS port:       ${VLA_PORT}"
echo "Log base:      ${LOG_BASE}"
nvidia-smi -L
echo "============================================================"

# Sanity-check restore state existence and source
if [ ! -f "${RESTORE_STATE}" ]; then
  echo "ERROR: restore JSON not found: ${RESTORE_STATE}"; exit 1
fi
python3 -c "import json; d=json.load(open('${RESTORE_STATE}')); src=d.get('source',{}); print(f\"restore: task={src.get('task_name')} inst={src.get('instance_id')} demo={src.get('demo_id')} step={src.get('rollout_step')}\")"

# ── 1) Launch pi05 inference server on GPU 1 ──────────────────────────────
SERVER_LOG="${LOG_BASE}/pi_server.log"
(
    cd "${OPENPI_ROOT}"
    export PATH="${HOME}/.local/bin:${PATH}"
    export CUDA_VISIBLE_DEVICES=1
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
    export JAX_PLATFORMS=cuda
    exec "${OPENPI_ROOT}/.venv/bin/python" -u scripts/deploy_pi05_b1k.py \
        --config-name "${PI_CONFIG_NAME}" \
        --ckpt-dir "${PI_CKPT}" \
        --host 127.0.0.1 \
        --port "${VLA_PORT}"
) > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap 'echo "stopping pi server $SERVER_PID"; kill $SERVER_PID 2>/dev/null || true; rm -rf "/tmp/og_appdata_${SLURM_JOB_ID}" 2>/dev/null || true' EXIT
echo "pi05 server launched (PID=${SERVER_PID})"

sleep 5
if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    echo "ERROR: pi05 server died within 5 s. Last 60 lines:"
    tail -60 "${SERVER_LOG}"
    exit 1
fi

# ── 2) Run recovery eval on GPU 0 ──────────────────────────────────────────
(
    source /shared_work/miniconda3/etc/profile.d/conda.sh
    conda activate behavior
    export OMNIGIBSON_GPU_ID=0
    export OMNIGIBSON_HEADLESS=1
    export OMNI_KIT_ACCEPT_EULA=YES
    export OG_DATA_PATH=/shared_work/BEHAVIOR-1K/datasets
    export OMNIGIBSON_APPDATA_PATH="/tmp/og_appdata_${SLURM_JOB_ID}"
    mkdir -p "${OMNIGIBSON_APPDATA_PATH}"
    cd /shared_work/BEHAVIOR-1K

    MAX_STEPS_ARG=""
    if [ -n "${MAX_STEPS:-}" ]; then MAX_STEPS_ARG="max_steps=${MAX_STEPS}"; fi
    python /shared_work/behavior1k-mp/eval_recovery/eval_recovery.py \
        task.name="${TASK_NAME}" \
        log_path="${LOG_BASE}" \
        policy=websocket \
        model.port=${VLA_PORT} \
        +restore_state="${RESTORE_STATE}" \
        ${MAX_STEPS_ARG}
)
RC=$?

echo "Recovery eval finished (rc=${RC})"
exit $RC
