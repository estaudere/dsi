#!/bin/bash

#SBATCH -J dsi
#SBATCH -o ./slurm_logs/%x_%j.out
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=42
#SBATCH -t 48:00:00
#SBATCH -A MLL
#SBATCH --mail-user=ndesaraju@utexas.edu
#SBATCH --mail-type=all

source /work/08609/nehades/ls6/miniconda3/bin/activate dsi

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export NCCL_NET_GDR_LEVEL="SYS"
export NCCL_NET_GDR_READ=1

echo $MASTER_ADDR

cd /work/08609/nehades/ls6/dsi/dsi-naive

srun python train.py