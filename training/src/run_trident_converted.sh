#!/usr/bin/env bash
# Extract features for the 981 CONVERTED (now pyramidal, OpenSlide-readable) slides.
# Run on gpu-57 AFTER conversion finishes. Pyramidal -> fast. Writes to the SAME features dir.
set -e
source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh
conda activate reg2026
WORK=/nfs-stor/adinath.dukre/adinath/reg2026_work
cd "$WORK/TRIDENT"
python run_batch_of_slides.py \
  --task all \
  --wsi_dir "$WORK/converted_slides" \
  --custom_list_of_wsis "$WORK/converted_wsis.csv" \
  --job_dir "$WORK/features" \
  --patch_encoder hoptimus0 \
  --mag 20 --patch_size 256 \
  --gpus 0 1 2 3 \
  --max_workers 8 \
  --skip_errors
echo "DONE converted-slide extraction."
