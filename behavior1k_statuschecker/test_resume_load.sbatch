#!/bin/bash
#SBATCH --job-name=test_resume_load
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5080:1
#SBATCH --time=0:15:00
#SBATCH --output=/shared_work/logs/%j_%x.out
#SBATCH --error=/shared_work/logs/%j_%x.err

echo "Job ID: ${SLURM_JOB_ID}   Node: $(hostname)"
cd /shared_work/markhsp/behavior1k-statuschecker
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export HF_HOME=/shared_work/markhsp/behavior1k-statuschecker/hf_cache
export CUDA_VISIBLE_DEVICES=0

/shared_work/markhsp/behavior1k-statuschecker/.venv/bin/python3 -u test_resume_load.py
RC=$?
echo "Test finished (rc=${RC})"
exit $RC
