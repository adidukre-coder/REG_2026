#!/usr/bin/env bash
#SBATCH --job-name=reg_diagotsu
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=gpu-12,gpu-13,gpu-19,gpu-33,gpu-45
#SBATCH --array=0-6
#SBATCH --output=/nfs-stor/adinath.dukre/adinath/reg2026_work/logs/diagotsu_%a.out
set -e
source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh; conda activate reg2026
cd /home/adinath.dukre/adinath/reg_2026/REG2026
export PYTHONPATH=$PWD
export HF_HOME=/nfs-stor/adinath.dukre/adinath/reg2026_work/hf_cache
export TRIDENT_HOME=/nfs-stor/adinath.dukre/adinath/reg2026_work/trident_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export REG_SEGMENTER=otsu REG_SEG_WORKERS=0
export REG_MAX_PATCHES=${REG_MAX_PATCHES:-1000}   # default to the SUBMITTED config cap (1000)
echo "config: SEGMENTER=$REG_SEGMENTER MAX_PATCHES=$REG_MAX_PATCHES (true submitted config)"
python src/diag_otsu_reextract.py ${SLURM_ARRAY_TASK_ID} 7
