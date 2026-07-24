#!/usr/bin/env bash
#SBATCH --job-name=reg_hesttime
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=gpu-12,gpu-22,gpu-33,gpu-45
#SBATCH --output=/nfs-stor/adinath.dukre/adinath/reg2026_work/logs/hesttime_%j.out
set -e
source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh; conda activate reg2026
cd /home/adinath.dukre/adinath/reg_2026/REG2026
export PYTHONPATH=$PWD
# offline caches the hest path needs (mirror the submitted model bundle exactly)
export HF_HOME=/nfs-stor/adinath.dukre/adinath/reg2026_work/hf_cache
export TRIDENT_HOME=/nfs-stor/adinath.dukre/adinath/reg2026_work/trident_cache
export TORCH_HOME=/nfs-stor/adinath.dukre/adinath/reg2026_work/submission_model/torch_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# the hest config we want to PROVE (hest seg, no workers = shm-safe, 2000-patch cap)
export REG_SEGMENTER=hest REG_SEG_WORKERS=0 REG_MAX_PATCHES=2000
echo "node: $(hostname) | $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader | head -1

D=/nfs-stor/adinath.dukre/adinath/reg2026
# WORST CASE: 5 biggest test1 slides (up to 4GB) + 1GB debug villain + a 100k-patch train slide
python src/diag_hest_timing.py \
  "$D/test1/PIT_638570.tiff" \
  "$D/test1/PIT_931110.tiff" \
  "$D/test1/PIT_395033.tiff" \
  "$D/test1/PIT_502136.tiff" \
  "$D/test1/PIT_612763.tiff" \
  "$D/debug/PIT_03_00366_01.tiff" \
  "$D/train/PIT_02_00131_01.tiff"
echo "done: $(date)"
