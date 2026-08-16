#!/bin/bash
#SBATCH --job-name=eval_lora_v3_base
#SBATCH --partition=all
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5080:1
#SBATCH --time=2:00:00
#SBATCH --output=/shared_work/logs/%A_%a_%x.out
#SBATCH --error=/shared_work/logs/%A_%a_%x.err

# Baseline: raw untrained Qwen3.5-0.8B (no LoRA) on the same full eval set, same
# prompt/label list as the fine-tuned runs - to measure how much training helped.
SHARD_SIZE=10
START=$((SLURM_ARRAY_TASK_ID * SHARD_SIZE))
END=$((START + SHARD_SIZE - 1))

echo "Job ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}   Node: $(hostname)   tasks[${START}:${END}]"
nvidia-smi -L

cd /shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export HF_HOME=/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker/hf_cache
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NO_ADAPTER=1
export RESULT_TAG=baseeval

/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker/.venv/bin/python3 -u eval_lora_v3_fulleval.py "$START" "$END"
RC=$?
echo "Shard finished (rc=${RC})"
exit $RC
