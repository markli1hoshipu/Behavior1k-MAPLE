#!/bin/bash
#SBATCH --job-name=lora_train_v3_4gpu
#SBATCH --partition=all
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx5080:2
#SBATCH --exclude=trt-EAI-OLDLAB-2,trt-eai-oldlab-6,aaron-workstation
#SBATCH --time=8:00:00
#SBATCH --output=/shared_work/logs/%j_%x.out
#SBATCH --error=/shared_work/logs/%j_%x.err

echo "Job ID: ${SLURM_JOB_ID}   Nodes: ${SLURM_JOB_NODELIST}   NNODES: ${SLURM_NNODES}"

cd /shared_work/markhsp/behavior1k-statuschecker
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export HF_HOME=/shared_work/markhsp/behavior1k-statuschecker/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# lab cluster nodes are plain ethernet, not InfiniBand - avoid NCCL hanging trying IB verbs
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
# nodes have multiple NICs incl. IPv6 link-local (fe80::...) that aren't routable
# cross-node - force NCCL to IPv4 only and skip loopback/virtual interfaces
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=^lo,docker0,virbr0,veth
export GLOO_SOCKET_IFNAME=^lo,docker0,virbr0,veth

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
    echo \"[node \$(hostname), SLURM_NODEID=\$SLURM_NODEID] starting torchrun\"
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
