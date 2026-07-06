#!/bin/bash
#SBATCH --job-name=b2026_eval_openpi
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --gres=gpu:rtx5080:2
#SBATCH --time=2:00:00
#SBATCH --exclude=trt-eai-oldlab-5
#SBATCH --output=/shared_work/logs/%j_%x.out
#SBATCH --error=/shared_work/logs/%j_%x.err

# BEHAVIOR-2026 (v3.9.0) eval with pi05 policy over websocket.
# Uses the NEW omnigibson.eval CLI (was omnigibson.learning.eval in v3.8).
#
# Usage:
#   PI_CKPT=<abs_ckpt>  sbatch eval_openpi_b2026.sbatch \
#       <task_name> <instance_index> [mode=public_test|train]
set -uo pipefail

TASK_NAME="${1:?task_name required}"
INSTANCE_IDX="${2:?instance index required}"
MODE="${3:-public_test}"
PI_CKPT="${PI_CKPT:?set PI_CKPT}"
PI_CONFIG_NAME="${PI_CONFIG_NAME:-pi05_b1k_mp_radio}"
OPENPI_ROOT="${OPENPI_ROOT:-/shared_work/openpi}"

LOG_BASE="${LOG_BASE:-/shared_work/logs/b2026_eval/openpi_${SLURM_JOB_ID}}"
VLA_PORT="${VLA_PORT:-$((8765 + SLURM_JOB_ID % 1000))}"
mkdir -p "${LOG_BASE}"

echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        $(hostname)"
echo "Task:        ${TASK_NAME}   Instance: ${INSTANCE_IDX}   Mode: ${MODE}"
echo "pi config:   ${PI_CONFIG_NAME}"
echo "pi ckpt:     ${PI_CKPT}"
echo "WS port:     ${VLA_PORT}"
echo "Log base:    ${LOG_BASE}"
nvidia-smi -L
echo "============================================================"

# ── 1) pi05 server on GPU 1 ──────────────────────────────────────────
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
    echo "ERROR: pi05 server died within 5 s"
    tail -60 "${SERVER_LOG}"
    exit 1
fi

# ── 2) eval on GPU 0 via v3.9 CLI ────────────────────────────────────
(
    source /shared_work/miniconda3/etc/profile.d/conda.sh
    conda activate behavior2026
    export OMNIGIBSON_GPU_ID=0
    export OMNIGIBSON_HEADLESS=1
    export OMNI_KIT_ACCEPT_EULA=YES
    export OG_DATA_PATH=/shared_work/BEHAVIOR-2026/datasets
    export OMNIGIBSON_DATA_PATH=/shared_work/BEHAVIOR-2026/datasets
    export OMNIGIBSON_APPDATA_PATH="/tmp/og_appdata_${SLURM_JOB_ID}"
    mkdir -p "${OMNIGIBSON_APPDATA_PATH}"
    cd /shared_work/BEHAVIOR-2026

    python -m omnigibson.eval.eval \
        --task-name "${TASK_NAME}" \
        --host 127.0.0.1 --port ${VLA_PORT} \
        --instance-indices "${INSTANCE_IDX}" \
        --mode "${MODE}" \
        --num-rollouts 1 \
        --headless \
        --write-video \
        --output-dir "${LOG_BASE}"
)
RC=$?

echo "Eval finished (rc=${RC})"
exit $RC
