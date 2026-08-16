#!/bin/bash
#SBATCH --job-name=lora_train_v3_10task
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx5080:2
#SBATCH --exclude=trt-EAI-OLDLAB-2,TRT-EAI-OLDLAB-1,trt-eai-oldlab-6,aaron-workstation
#SBATCH --time=6:00:00
#SBATCH --output=/shared_work/logs/%j_%x.out
#SBATCH --error=/shared_work/logs/%j_%x.err

# 10-task subset (4,10,16,29,37,46,55,64,72,89), 4 epochs, single-node 2-GPU (avoids
# the cross-node NCCL issues hit on the full 100-task run). Saves a checkpoint after
# every epoch (lora_adapter_v3_10task_epoch0..3) so we can eval at 2 epochs (epoch1)
# and 4 epochs (epoch3/final) from this one run.

echo "Job ID: ${SLURM_JOB_ID}   Node: $(hostname)"
nvidia-smi -L

cd /shared_work/markhsp/behavior1k-statuschecker
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export HF_HOME=/shared_work/markhsp/behavior1k-statuschecker/hf_cache
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TASK_SUBSET=task-0004,task-0010,task-0016,task-0029,task-0037,task-0046,task-0055,task-0064,task-0072,task-0089
export ADAPTER_OUT=/shared_work/markhsp/behavior1k-statuschecker/lora_adapter_v3_10task
export HELDOUT_OUT=/shared_work/markhsp/behavior1k-statuschecker/results/lora_v3_10task_heldout_episodes.json
export EPOCHS_TO_RUN=4

/shared_work/markhsp/behavior1k-statuschecker/.venv/bin/accelerate launch \
    --multi_gpu --num_processes=2 --mixed_precision=bf16 \
    lora_train_v3.py
RC=$?
echo "Training finished (rc=${RC})"
exit $RC
