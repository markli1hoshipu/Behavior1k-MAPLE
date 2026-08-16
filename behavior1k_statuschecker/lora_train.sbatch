#!/bin/bash
#SBATCH --job-name=lora_qwen35_08b
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5080:1
#SBATCH --exclude=trt-EAI-OLDLAB-2
#SBATCH --time=2:00:00
#SBATCH --output=/shared_work/logs/%j_%x.out
#SBATCH --error=/shared_work/logs/%j_%x.err

# LoRA fine-tune Qwen3.5-0.8B on Opus-generated CoT annotations across 10
# BEHAVIOR-1K tasks. Dedicated compute node so it doesn't contend with
# interactive-node jobs (this OOM'd once already on trt-eai-oldlab-1 due to
# a concurrent rd-collect job sharing the GPU).

echo "============================================================"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
nvidia-smi -L
echo "============================================================"

cd /shared_work/markhsp/behavior1k-statuschecker
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV  # avoid inheriting stale venv state from the submitting shell
export HF_HOME=/shared_work/markhsp/behavior1k-statuschecker/hf_cache
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/shared_work/markhsp/behavior1k-statuschecker/.venv/bin/python3 -u lora_train.py
RC=$?
echo "Training finished (rc=${RC})"
exit $RC
