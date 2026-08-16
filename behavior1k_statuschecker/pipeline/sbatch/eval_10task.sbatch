#!/bin/bash
#SBATCH --job-name=eval_10task
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5080:1
#SBATCH --time=1:00:00
#SBATCH --output=/shared_work/logs/%j_%x.out
#SBATCH --error=/shared_work/logs/%j_%x.err

# Reusable single-GPU eval for the 10-task experiment (only 10 tasks -> no need for
# a job array). Set ADAPTER_PATH/NO_ADAPTER/RESULT_TAG before sbatch submission, e.g.:
#   sbatch --export=ALL,NO_ADAPTER=1,RESULT_TAG=10task_base eval_10task.sbatch
#   sbatch --export=ALL,ADAPTER_PATH=.../lora_adapter_v3_10task_epoch1,RESULT_TAG=10task_2ep eval_10task.sbatch
#   sbatch --export=ALL,ADAPTER_PATH=.../lora_adapter_v3_10task,RESULT_TAG=10task_4ep eval_10task.sbatch

echo "Job ID: ${SLURM_JOB_ID}   Node: $(hostname)   ADAPTER_PATH=${ADAPTER_PATH}  NO_ADAPTER=${NO_ADAPTER}  RESULT_TAG=${RESULT_TAG}"
nvidia-smi -L

cd /shared_work/markhsp/behavior1k-statuschecker
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export HF_HOME=/shared_work/markhsp/behavior1k-statuschecker/hf_cache
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TASK_SUBSET=task-0004,task-0010,task-0016,task-0029,task-0037,task-0046,task-0055,task-0064,task-0072,task-0089

/shared_work/markhsp/behavior1k-statuschecker/.venv/bin/python3 -u eval_lora_v3_fulleval.py
RC=$?
echo "Eval finished (rc=${RC})"
exit $RC
