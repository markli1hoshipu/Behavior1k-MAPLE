#!/bin/bash
#SBATCH --job-name=lora_train_v3_2gpu
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx5080:2
#SBATCH --exclude=trt-EAI-OLDLAB-2
#SBATCH --time=8:00:00
#SBATCH --output=/shared_work/logs/%j_%x.out
#SBATCH --error=/shared_work/logs/%j_%x.err

echo "Job ID: ${SLURM_JOB_ID}   Node: $(hostname)"
nvidia-smi -L

cd /shared_work/markhsp/behavior1k-statuschecker
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export HF_HOME=/shared_work/markhsp/behavior1k-statuschecker/hf_cache
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/shared_work/markhsp/behavior1k-statuschecker/.venv/bin/accelerate launch \
    --multi_gpu --num_processes=2 --mixed_precision=bf16 \
    lora_train_v3.py
RC=$?
echo "Training finished (rc=${RC})"
exit $RC
