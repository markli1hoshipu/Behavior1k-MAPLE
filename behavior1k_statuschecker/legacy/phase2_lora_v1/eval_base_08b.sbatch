#!/bin/bash
#SBATCH --job-name=eval_base_08b
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

echo "Job ID: ${SLURM_JOB_ID}   Node: $(hostname)"
nvidia-smi -L

cd /shared_work/markhsp/behavior1k-statuschecker
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export HF_HOME=/shared_work/markhsp/behavior1k-statuschecker/hf_cache
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/shared_work/markhsp/behavior1k-statuschecker/.venv/bin/python3 -u eval_base_08b.py
RC=$?
echo "Base-model eval finished (rc=${RC})"
exit $RC
