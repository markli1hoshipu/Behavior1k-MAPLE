#!/bin/bash
#SBATCH --job-name=eval_lora_v3_ep0
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

# Evaluates the COMPLETE epoch-0 checkpoint (1 full epoch over all 100 tasks) against
# the full disjoint eval set, as an immediate parallel safety net while the epoch-1
# resume (job 32810) is still being sorted out. 10 shards of 10 tasks each.
SHARD_SIZE=10
START=$((SLURM_ARRAY_TASK_ID * SHARD_SIZE))
END=$((START + SHARD_SIZE - 1))

echo "Job ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}   Node: $(hostname)   tasks[${START}:${END}]"
nvidia-smi -L

cd /shared_work/markhsp/behavior1k-statuschecker
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export HF_HOME=/shared_work/markhsp/behavior1k-statuschecker/hf_cache
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ADAPTER_PATH=/shared_work/markhsp/behavior1k-statuschecker/lora_adapter_v3_epoch0
export RESULT_TAG=epoch0eval

/shared_work/markhsp/behavior1k-statuschecker/.venv/bin/python3 -u eval_lora_v3_fulleval.py "$START" "$END"
RC=$?
echo "Shard finished (rc=${RC})"
exit $RC
