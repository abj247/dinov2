#!/bin/bash
#SBATCH --job-name=dinov2_1m
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160Gb
#SBATCH --gres=gpu:a100:1
#SBATCH --time=480:00:00
#SBATCH --partition=a100_long

# Load modules
module load python/3.10.10

# Activate conda environment
source /gpfs/data/shenlab/aj4718/miniconda3/etc/profile.d/conda.sh
conda activate dinov2

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export WANDB_API_KEY='YOUR_WANDB_API_KEY'

echo "Starting DINOv2 Pyramidal Training (1M Images)..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "========================================" 

# Run the training script
cd /gpfs/data/shenlab/aj4718/dinov2
CONFIG="dinov2/configs/train/vits14_pyramid_1m.yaml"
# Append Job ID to output directory to ensure a fresh run
OUTPUT_DIR="/gpfs/data/shenlab/aj4718/dinov2/logs/vits14_pyramid_1m_${SLURM_JOB_ID}"

# Create output directory
mkdir -p $OUTPUT_DIR

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Using torchrun for distributed training
# It will automatically use the detected number of GPUs via --nproc_per_node
torchrun --nproc_per_node=$NUM_GPUS dinov2/train/train_pyramid.py \
    --config-file $CONFIG \
    --output-dir $OUTPUT_DIR \
    --run_name pyramid_distillation_1m_1gpu_${SLURM_JOB_ID} \
    --wandb-project dinov2_pyramid_1m_test
