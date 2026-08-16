#!/bin/bash
#SBATCH --job-name=lora_train_v3_10task_1gpu
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5080:1
#SBATCH --exclude=trt-EAI-OLDLAB-2,TRT-EAI-OLDLAB-1,trt-eai-oldlab-6,aaron-workstation
#SBATCH --time=8:00:00
#SBATCH --output=/shared_work/logs/%j_%x.out
#SBATCH --error=/shared_work/logs/%j_%x.err

# Single-GPU (no DDP/NCCL at all) - the 2-GPU version hit a reproducible, deterministic
# collective timeout twice in a row at the exact same point (same NCCL seq number both
# times, even on a fully idle node), almost certainly one specific training example
# stalling one rank's data loading. Single-GPU sidesteps the whole issue: slower per
# epoch but guaranteed to just work.

echo "Job ID: ${SLURM_JOB_ID}   Node: $(hostname)"
nvidia-smi -L

cd /shared_work/markhsp/behavior1k-statuschecker
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export HF_HOME=/shared_work/markhsp/behavior1k-statuschecker/hf_cache
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TASK_SUBSET=task-0004,task-0010,task-0016,task-0029,task-0037,task-0046,task-0055,task-0064,task-0072,task-0089
export ADAPTER_OUT=/shared_work/markhsp/behavior1k-statuschecker/lora_adapter_v3_10task
export HELDOUT_OUT=/shared_work/markhsp/behavior1k-statuschecker/results/lora_v3_10task_heldout_episodes.json
export EPOCHS_TO_RUN=4

/shared_work/markhsp/behavior1k-statuschecker/.venv/bin/python3 -u lora_train_v3.py
RC=$?
echo "Training finished (rc=${RC})"
exit $RC
