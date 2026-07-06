# shared_env.sh — sourced by SLURM sbatch scripts in this directory.
#
# Provides:
#   • common env defaults (BEHAVIOR-1K clone, conda env, port allocation)
#   • helper functions:  wait_for_port,  log_banner
#
# Usage in an sbatch script:
#   source "$(dirname "${BASH_SOURCE[0]}")/shared_env.sh"

# ─── env defaults ────────────────────────────────────────────────────
: "${MAPLE_ROOT:=/shared_work/behavior1k-mp}"
: "${BEHAVIOR1K_ROOT:=/shared_work/BEHAVIOR-1K}"
: "${OPENPI_ROOT:=/shared_work/openpi}"
: "${CONDA_ENV:=behavior}"
: "${LOG_ROOT:=/shared_work/logs}"

# Deterministic-but-unique port per SLURM job (avoids collisions on shared nodes).
: "${VLA_PORT:=$((8765 + SLURM_JOB_ID % 1000))}"

export MAPLE_ROOT BEHAVIOR1K_ROOT OPENPI_ROOT CONDA_ENV LOG_ROOT VLA_PORT

# ─── helpers ─────────────────────────────────────────────────────────
# wait_for_port <port> [<timeout_s=90>]
# Returns 0 as soon as the port is accepting TCP connections on localhost.
wait_for_port() {
    local port="$1"
    local timeout="${2:-90}"
    local start
    start=$(date +%s)
    while ! (echo > /dev/tcp/127.0.0.1/"$port") 2>/dev/null; do
        if [ $(( $(date +%s) - start )) -gt "$timeout" ]; then
            echo "wait_for_port: timeout after ${timeout}s waiting for :${port}" >&2
            return 1
        fi
        sleep 1
    done
}

log_banner() {
    echo "============================================================"
    echo "Job ID:      ${SLURM_JOB_ID:-<not-in-slurm>}"
    echo "Node:        $(hostname)"
    for kv in "$@"; do
        echo "$kv"
    done
    echo "============================================================"
}
