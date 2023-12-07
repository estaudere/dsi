#!/bin/bash

#SBATCH -J clip-dsi
#SBATCH -o ./slurm_logs/%x_%j.out
#SBATCH -p gpu-a100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=3
#SBATCH -t 5:00:00
#SBATCH -A CCR23037
#SBATCH --mail-user=ndesaraju@utexas.edu
#SBATCH --mail-type=all

source /work/08609/nehades/ls6/miniconda3/bin/activate dsi

# debugging flags
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# network flags
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export NCCL_NET_GDR_LEVEL="SYS"
export NCCL_NET_GDR_READ=1

echo $MASTER_ADDR

cd /work/08609/nehades/ls6/dsi/clip-dsi

export CUDA_LAUNCH_BLOCKING=1

srun python train.py \
    --steps 1000000 \
    --batch_size 16 \
    --logger "csv" \
    --val_epochs 50 \
    --log_steps 10 \
    # --ckpt_path 