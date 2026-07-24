#!/bin/bash
#SBATCH --job-name=reg_train_treatment
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-12,gpu-22,gpu-45
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs-stor/adinath.dukre/adinath/reg2026_work/logs/train_treatment_%j.out
source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh
conda activate reg2026
cd /home/adinath.dukre/adinath/reg_2026/REG2026
export PYTHONPATH=/home/adinath.dukre/adinath/reg_2026/REG2026
export REG_FEAT_DIR=/nfs-stor/adinath.dukre/adinath/reg2026_work/features_otsu2000
export REG_SEED=42
export REG_LAB_PATH=/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json
export REG_LM_PATH=/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/label_maps.json
export REG_OUT_CKPT=/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints/reg_attr_treatment_seeded.pt
echo "node: $(hostname) | start: $(date) | TREATMENT (fixed labels) seed=$REG_SEED"
python src/analyze_performance_seeded.py 1600 20
echo "end: $(date)"
