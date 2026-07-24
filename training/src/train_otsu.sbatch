#!/bin/bash
#SBATCH --job-name=reg_trainotsu
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-12,gpu-22,gpu-45
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs-stor/adinath.dukre/adinath/reg2026_work/logs/trainotsu_%j.out
source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh
conda activate reg2026
cd /home/adinath.dukre/adinath/reg_2026/REG2026
export PYTHONPATH=/home/adinath.dukre/adinath/reg_2026/REG2026
# train the attribute model; save to a SEPARATE checkpoint (NEVER clobber the working model).
# FEAT_DIR / OUT_CKPT honor pre-set env (so the same script can train otsu OR hest for comparison).
export REG_FEAT_DIR=${REG_FEAT_DIR:-/nfs-stor/adinath.dukre/adinath/reg2026_work/features_otsu2000}
export REG_OUT_CKPT=${REG_OUT_CKPT:-/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints/reg_attr_otsu.pt}
PER_ORGAN=${PER_ORGAN:-1600}
EPOCHS=${EPOCHS:-20}
echo "node: $(hostname) | start: $(date) | FEAT=$REG_FEAT_DIR PER_ORGAN=$PER_ORGAN EPOCHS=$EPOCHS"
python -c "import torch;print('cuda',torch.cuda.is_available())"
python src/analyze_performance.py $PER_ORGAN $EPOCHS
echo "saved otsu model -> $REG_OUT_CKPT"
echo "end: $(date)"
