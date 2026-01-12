#!/bin/bash
#SBATCH --job-name=train_stage2
#SBATCH --output=sbatch_output/%j_%x.out
#SBATCH --error=sbatch_output/%j_%x.err
#SBATCH --nodes=1                 # can be changed freely
#SBATCH --ntasks-per-node=1       # 1 task per GPU
#SBATCH --gpus-per-node=4         # 4 GPUs per node
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$CRSID@cam.ac.uk

# -------------------------------
#  Environment setup
# -------------------------------
set -e

hostname
nvidia-smi --list-gpus

if command -v module &> /dev/null; then
    module load cuda/12.6 gcc-native/13.2
fi
source $HOME/.bashrc
source $SCRATCH/miniforge3/bin/activate rae

# Config
STAGE_NAME="stage2"
MODEL_NAME="DiT-B"
IMAGE_SIZE=256
CONFIG_PATH=configs/${STAGE_NAME}/training/ImageNet${IMAGE_SIZE}/${MODEL_NAME}_DINOv2-B.yaml
DATA_PATH="data/imagenet-1k/ImageNet/train"

# Wandb
export PROJECT=imagerae_${STAGE_NAME}

# -------------------------------
#  Local setup
# -------------------------------
export NNODES=1
export GPUS_PER_NODE=4
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "================ LOCAL SETUP ================"
echo "NNODES:         $NNODES"
echo "GPUS_PER_NODE:  $GPUS_PER_NODE"
echo "WORLD_SIZE:     $WORLD_SIZE"
echo "=================================================="

# -------------------------------
#  Launch local training
# -------------------------------
torchrun --standalone \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    src/train_${STAGE_NAME}.py \
    --config $CONFIG_PATH \
    --data-path "${DATA_PATH}" \
    --results-dir results/${STAGE_NAME}/training \
    --precision bf16 \
    --image-size $IMAGE_SIZE \
    --global-batch-size 1024 \
    --micro-batch-size 256 \
    --log-every 50 \
    --ckpt-every 5000 \
    --sample-every 10000 \
    --wandb
