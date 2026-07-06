#!/bin/bash
#SBATCH --job-name=b1k_eval_openpi
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

# Pure pi05/openpi evaluation (no motion-planner orchestrator, no data collection).
# Drop-in equivalent of `eval_xvla.sbatch`, but the inference server is
# openpi's `deploy_pi05_b1k.py` instead of X-VLA's `deploy_b1k.py`.
#
# Usage:
#   PI_CKPT=/path/to/orbax/ckpt sbatch /shared_work/behavior1k-mp/slurm/eval_openpi.sbatch \
#          turning_on_radio 242

set -uo pipefail

TASK_NAME="${1:-turning_on_radio}"
INSTANCE_ID="${2:-242}"
PI_CKPT="${3:-${PI_CKPT:-/shared_work/behavior1k-xvla/checkpoints/openpi/pi05_b1k_mp_radio/ckpt-60000}}"
PI_CONFIG_NAME="${PI_CONFIG_NAME:-pi05_b1k_mp_radio}"
OPENPI_ROOT="${OPENPI_ROOT:-/shared_work/openpi}"

LOG_BASE="${LOG_BASE:-/shared_work/logs/behavior_eval/openpi_${SLURM_JOB_ID}}"
VLA_PORT="${VLA_PORT:-$((8765 + SLURM_JOB_ID % 1000))}"
mkdir -p "${LOG_BASE}"

echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        $(hostname)"
echo "Task:        ${TASK_NAME}   Instance: ${INSTANCE_ID}"
echo "pi config:   ${PI_CONFIG_NAME}"
echo "pi ckpt:     ${PI_CKPT}"
echo "WS port:     ${VLA_PORT}"
echo "Log base:    ${LOG_BASE}"
nvidia-smi -L
echo "============================================================"

# Translate literal instance ID → CSV index
CSV_PATH=/shared_work/BEHAVIOR-1K/datasets/2025-challenge-task-instances/metadata/test_instances.csv
INSTANCE_INDEX=$(awk -F',' -v task="${TASK_NAME}" -v inst="${INSTANCE_ID}" '
    NR==1 { next }
    {
        if ($2 != task) next
        for (i = 3; i <= NF; i++) {
            gsub(/[ "\r]/, "", $i)
            if ($i == inst) { print i - 3; exit 0 }
        }
        print "NOT_FOUND"; exit 1
    }
' "${CSV_PATH}")
if [ "${INSTANCE_INDEX}" = "NOT_FOUND" ] || [ -z "${INSTANCE_INDEX}" ]; then
    echo "ERROR: instance ${INSTANCE_ID} not found in test_instances.csv for task ${TASK_NAME}"
    exit 1
fi
echo "Resolved instance ID ${INSTANCE_ID} → CSV index ${INSTANCE_INDEX}"

# ─── 1) Launch pi05 inference server on GPU 1 ──────────────────────────────
SERVER_LOG="${LOG_BASE}/pi_server.log"
(
    cd "${OPENPI_ROOT}"
    export PATH="${HOME}/.local/bin:${PATH}"
    export CUDA_VISIBLE_DEVICES=1
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
    export JAX_PLATFORMS=cuda
    # Use the venv python directly (uv run hides early stdout from sbatch).
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

# ─── 2) Run eval on GPU 0 — eval.py + policy=websocket ──────────────────────
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

    python OmniGibson/omnigibson/learning/eval.py \
        task.name="${TASK_NAME}" \
        "eval_instance_ids=[${INSTANCE_INDEX}]" \
        log_path="${LOG_BASE}" \
        policy=websocket \
        model.port=${VLA_PORT}
)
RC=$?

echo "Eval finished (rc=${RC})"
exit $RC
