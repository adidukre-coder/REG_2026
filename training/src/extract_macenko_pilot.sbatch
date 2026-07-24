#!/usr/bin/env bash
#SBATCH --job-name=reg_macpilot
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu-12,gpu-17,gpu-22,gpu-33,gpu-45
#SBATCH --array=0-3
#SBATCH --output=/nfs-stor/adinath.dukre/adinath/reg2026_work/logs/macpilot_%a.out
source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh; conda activate reg2026
cd /home/adinath.dukre/adinath/reg_2026/REG2026
export PYTHONPATH=$PWD
export TMPDIR=/scratch/$(id -un)/regtmp; mkdir -p "$TMPDIR" 2>/dev/null || export TMPDIR=/nfs-stor/adinath.dukre/adinath/reg2026_work/tmp
export HF_HOME=/nfs-stor/adinath.dukre/adinath/reg2026_work/hf_cache
export TRIDENT_HOME=/nfs-stor/adinath.dukre/adinath/reg2026_work/trident_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "node: $(hostname) | $(date) | task ${SLURM_ARRAY_TASK_ID}"
python src/extract_macenko_pilot.py ${SLURM_ARRAY_TASK_ID} 4 60
echo "[task ${SLURM_ARRAY_TASK_ID}] python finished at $(date)"
