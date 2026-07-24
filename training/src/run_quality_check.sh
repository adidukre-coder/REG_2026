#!/usr/bin/env bash
# Validate the convert-to-pyramidal step: extract features for 3 slides from BOTH the original
# (non-pyramidal, train/) and the converted (pyramidal, converted_slides/) versions, then compare.
# Run on a GPU node (gpu-16):  bash src/run_quality_check.sh
set -e
source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh
conda activate reg2026
DATA=/nfs-stor/adinath.dukre/adinath/reg2026
WORK=/nfs-stor/adinath.dukre/adinath/reg2026_work
QC="$WORK/qc"; mkdir -p "$QC"
cd "$WORK/TRIDENT"

# 3 known-good, OpenSlide-readable slides (exist as both original and converted)
SL="PIT_01_00019_01 PIT_01_00021_01 PIT_01_00025_01"
{ echo "wsi,mpp"; for s in $SL; do echo "$s.tiff,0.5"; done; } >| "$QC/list.csv"

echo "=== [1/2] features from ORIGINAL (non-pyramidal train/) ==="
python run_batch_of_slides.py --task all --wsi_dir "$DATA/train" \
  --custom_list_of_wsis "$QC/list.csv" --job_dir "$QC/feat_orig" \
  --patch_encoder hoptimus0 --mag 20 --patch_size 256 --gpus 0

echo "=== [2/2] features from CONVERTED (pyramidal converted_slides/) ==="
python run_batch_of_slides.py --task all --wsi_dir "$WORK/converted_slides" \
  --custom_list_of_wsis "$QC/list.csv" --job_dir "$QC/feat_conv" \
  --patch_encoder hoptimus0 --mag 20 --patch_size 256 --gpus 0

echo "=== compare features (cosine similarity at matching coords) ==="
cd /home/adinath.dukre/adinath/reg_2026/REG2026
python src/compare_features.py "$QC/feat_orig" "$QC/feat_conv" $SL
