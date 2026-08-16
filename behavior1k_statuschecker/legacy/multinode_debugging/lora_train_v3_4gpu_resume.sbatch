#!/bin/bash
#SBATCH --job-name=lora_train_v3_4gpu_resume
#SBATCH --partition=all
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx5080:2
#SBATCH --nodelist=trt-eai-oldlab-3,trt-eai-oldlab-5
#SBATCH --time=7:00:00
#SBATCH --output=/shared_work/logs/%j_%x.out
#SBATCH --error=/shared_work/logs/%j_%x.err

# Resumes lora_train_v3.py from the epoch-0 checkpoint (job 32785 hit the 8h time
# limit ~74% through epoch 1) and runs just the remaining 1 epoch, instead of
# redoing the ~7.5h already spent on epoch 0 + partial epoch 1.

echo "Job ID: ${SLURM_JOB_ID}   Nodes: ${SLURM_JOB_NODELIST}   NNODES: ${SLURM_NNODES}"

cd /shared_work/markhsp/behavior1k-statuschecker
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export HF_HOME=/shared_work/markhsp/behavior1k-statuschecker/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=^lo,docker0,virbr0,veth
export GLOO_SOCKET_IFNAME=^lo,docker0,virbr0,veth
# fail fast (~10min) instead of hanging silently for hours if cross-node comms break again
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"

srun --kill-on-bad-exit=1 bash -c "
    unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
    export HF_HOME=/shared_work/markhsp/behavior1k-statuschecker/hf_cache
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_IB_DISABLE=1
    export NCCL_DEBUG=WARN
    export NCCL_SOCKET_FAMILY=AF_INET
    export NCCL_SOCKET_IFNAME=^lo,docker0,virbr0,veth
    export GLOO_SOCKET_IFNAME=^lo,docker0,virbr0,veth
    export TORCH_NCCL_BLOCKING_WAIT=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export RESUME_ADAPTER=/shared_work/markhsp/behavior1k-statuschecker/lora_adapter_v3_epoch0
    export EPOCH_OFFSET=1
    export EPOCHS_TO_RUN=1
    echo \"[node \$(hostname), SLURM_NODEID=\$SLURM_NODEID] starting torchrun (resume)\"
    /shared_work/markhsp/behavior1k-statuschecker/.venv/bin/python3 -m torch.distributed.run \
        --nnodes=\${SLURM_NNODES} \
        --nproc_per_node=2 \
        --node_rank=\${SLURM_NODEID} \
        --rdzv_id=\${SLURM_JOB_ID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        /shared_work/markhsp/behavior1k-statuschecker/lora_train_v3.py
"
RC=$?
echo "Training finished (rc=${RC})"
exit $RC
