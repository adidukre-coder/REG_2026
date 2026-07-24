#!/usr/bin/env bash
#SBATCH --job-name=reg_trainv2
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclude=gpu-12,gpu-33,gpu-45
#SBATCH --output=/nfs-stor/adinath.dukre/adinath/reg2026_work/logs/trainv2_%j.out

# Step-4 retrain: per_task_attention=True + report-driving criticality weights + more epochs.
# Saves BEST val-report checkpoint to reg_attr_v2_best.pt each epoch (baseline untouched).
set -e
echo "node: $(hostname) | start: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh
conda activate reg2026
cd /home/adinath.dukre/adinath/reg_2026/REG2026
export PYTHONPATH=$PWD
python src/train_v2.py 1600 30
echo "end: $(date)"
