#!/bin/bash
#SBATCH --job-name=eval_lora_v3_fulleval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5080:1
#SBATCH --time=4:00:00
#SBATCH --output=/shared_work/logs/%j_%x.out
#SBATCH --error=/shared_work/logs/%j_%x.err

echo "Job ID: ${SLURM_JOB_ID}   Node: $(hostname)"
nvidia-smi -L

cd /shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export HF_HOME=/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker/hf_cache
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker/.venv/bin/python3 -u eval_lora_v3_fulleval.py
RC=$?
echo "Eval finished (rc=${RC})"
exit $RC
