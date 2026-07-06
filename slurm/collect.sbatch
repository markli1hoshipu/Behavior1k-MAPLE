#!/bin/bash
#SBATCH --job-name=b1k_mp_collect
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

# Hybrid MP + X-VLA data collection for one (task, instance) pair.
# Allocates 2× 5080 — GPU 0 runs OmniGibson sim, GPU 1 runs the X-VLA inference server.
#
# Usage:
#   sbatch /shared_work/behavior1k-mp/slurm/collect.sbatch \
#          <task_name> <instance_id> [<vla_ckpt>]
#
# Example:
#   sbatch collect.sbatch turning_on_radio 301
#
# Required env (set externally or as defaults below):
#   LOG_BASE          (default: /shared_work/logs/behavior_datacollect/mp_${SLURM_JOB_ID})
#   VLA_PORT          (default: 8765 + offset by job id to avoid collisions)

set -uo pipefail

TASK_NAME="${1:-turning_on_radio}"
INSTANCE_ID="${2:-301}"
VLA_CKPT="${3:-/shared_work/behavior1k-xvla/checkpoints/v20/task0-60k}"
NUM_EPISODES="${NUM_EPISODES:-1}"

# Logs (vla_server.log, metrics, slurm out/err) stay under /shared_work/logs/.
# Demo dataset output (parquet, hdf5, videos, annotations, meta) goes to
# /shared_work/DATASETS/behavior-1k-mp-collected/ — pooled across all jobs so
# downstream tools see one coherent dataset tree (mirrors the HF layout).
LOG_BASE="${LOG_BASE:-/shared_work/logs/behavior_datacollect/mp_${SLURM_JOB_ID}}"
DATASET_ROOT="${DATASET_ROOT:-/shared_work/DATASETS/behavior-1k-mp-collected}"
VLA_PORT="${VLA_PORT:-$((8765 + SLURM_JOB_ID % 1000))}"
mkdir -p "${LOG_BASE}"
mkdir -p "${DATASET_ROOT}"
# Per-job staging dir for the in-flight HDF5 recorder (kept under /logs since
# the file is temporary and gets split into per-episode files at finalize).
STAGING_DIR="${LOG_BASE}/staging"
mkdir -p "${STAGING_DIR}"

echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        $(hostname)"
echo "Task:        ${TASK_NAME}   Instance: ${INSTANCE_ID}"
echo "VLA ckpt:    ${VLA_CKPT}"
echo "VLA port:    ${VLA_PORT}"
echo "Log base:    ${LOG_BASE}"
nvidia-smi -L
echo "============================================================"

# ───────────────────────── 1) Translate literal instance ID → CSV index ───────
# Fail fast: do this before launching anything so a bad instance ID exits
# without burning a VLA-server boot or an OmniGibson boot.
# `eval_data_gen.py` treats `eval_instance_ids` as INDICES into the per-task
# row of `test_instances.csv`, NOT as literal instance IDs.
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
echo "Resolved instance ID ${INSTANCE_ID} → CSV index ${INSTANCE_INDEX} (task ${TASK_NAME})"

# ───────────────────────── 2) Launch BOTH servers in parallel ────────────────
# VLA model load takes ~60 s; OmniGibson Kit + USD load takes ~3-5 min. So
# launching them simultaneously saves the 60 s that the old "sequential boot"
# spent in a foreground wait-for-port loop. The first websocket call from
# OmniGibson doesn't happen until `HybridPolicy.attach_env(env)` fires — well
# after VLA is ready — and `WebsocketClientPolicy._wait_for_server` does its
# own health check + retry, so no Python-side timing is required.
#
# Safety check: if VLA process exits within the first 5 s (e.g. model file
# missing, CUDA OOM at load), we want to fail loudly rather than let
# OmniGibson run for 5 min and then crash on a connect error.

SERVER_LOG="${LOG_BASE}/vla_server.log"
(
    source /shared_work/miniconda3/etc/profile.d/conda.sh
    conda activate xvla
    export CUDA_VISIBLE_DEVICES=1
    export PYTHONPATH="/shared_work/behavior1k-xvla/X-VLA:/shared_work/behavior1k-xvla/behavior1k_training:${PYTHONPATH:-}"
    cd /shared_work/behavior1k-xvla/behavior1k_training
    exec python -u deploy_b1k.py \
        --model_path "${VLA_CKPT}" \
        --port "${VLA_PORT}" \
        --host 127.0.0.1 \
        --device cuda
) > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap 'echo "stopping VLA server $SERVER_PID"; kill $SERVER_PID 2>/dev/null || true; rm -rf "/tmp/og_appdata_${SLURM_JOB_ID}" 2>/dev/null || true' EXIT
echo "VLA server launched in background (PID=${SERVER_PID}); model load runs in parallel with OmniGibson boot."

# Brief sanity check: if VLA crashes within 5 s, abort.
sleep 5
if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    echo "ERROR: VLA server died within 5 s of launch. Last 60 lines of ${SERVER_LOG}:"
    tail -60 "${SERVER_LOG}"
    exit 1
fi
echo "VLA server still alive after 5 s — proceeding to launch OmniGibson immediately."

# ───────────────────────── 3) Run data collection on GPU 0 ─────────────────────────
(
    source /shared_work/miniconda3/etc/profile.d/conda.sh
    conda activate behavior
    pip install -q -e /shared_work/behavior1k-mp >/dev/null 2>&1 || true
    export OMNIGIBSON_GPU_ID=0
    export OMNIGIBSON_HEADLESS=1
    export OMNI_KIT_ACCEPT_EULA=YES
    export OG_DATA_PATH=/shared_work/BEHAVIOR-1K/datasets
    export OMNIGIBSON_APPDATA_PATH="/tmp/og_appdata_${SLURM_JOB_ID}"
    mkdir -p "${OMNIGIBSON_APPDATA_PATH}"
    cd /shared_work/BEHAVIOR-1K

    # Ensure our hybrid_mp config is reachable via Hydra's policy/ search path.
    POL_DIR=/shared_work/BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/policy
    if [ ! -e "${POL_DIR}/hybrid_mp.yaml" ]; then
        ln -sf /shared_work/behavior1k-mp/configs/policy/hybrid_mp.yaml "${POL_DIR}/hybrid_mp.yaml"
    fi

    # Use eval_data_gen_par_save_all.py: saves BOTH successful and failed
    # episodes (good for debugging hybrid MP), writes parquet + videos directly
    # (no separate replay step), and already handles 420-entry CSV rows.
    python OmniGibson/omnigibson/learning/eval_data_gen_par_save_all.py \
        task.name="${TASK_NAME}" \
        instance_id="${INSTANCE_ID}" \
        "eval_instance_ids=[${INSTANCE_INDEX}]" \
        log_path="${LOG_BASE}" \
        "++output_folder=${DATASET_ROOT}" \
        "++recording_path=${STAGING_DIR}/${TASK_NAME}.hdf5" \
        "++num_eval_episodes=${NUM_EPISODES}" \
        policy=hybrid_mp \
        only_successes=false \
        model.vla_port=${VLA_PORT} \
        ++external_camera_enabled=true \
        "++external_camera_position=[1.5, 2.0, 1.5]" \
        "++external_camera_orientation=[0.5, -0.5, 0.5, -0.5]"
)
RC=$?

echo "Collection finished (rc=${RC})"
exit $RC
