#!/usr/bin/env bash
# FULL feature extraction — all 11,220 train slides, organ-interleaved, 4 GPUs.
# Launch DETACHED on gpu-57 so it survives logout:
#   tmux new -s extract        # then inside:  bash src/run_trident_full.sh
#   (or)  nohup bash src/run_trident_full.sh > work/logs/extract.out 2>&1 &
# It checkpoints per slide (skips finished) -> safe to restart / resume.
set -e
source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh
conda activate reg2026

DATA=/nfs-stor/adinath.dukre/adinath/reg2026
WORK=/nfs-stor/adinath.dukre/adinath/reg2026_work
cd "$WORK/TRIDENT"

python run_batch_of_slides.py \
  --task all \
  --wsi_dir "$DATA/train" \
  --custom_list_of_wsis "$WORK/train_interleaved.csv" \
  --job_dir "$WORK/features" \
  --patch_encoder hoptimus0 \
  --mag 20 --patch_size 256 \
  --gpus 0 1 2 3 \
  --max_workers 16 \
  --skip_errors

echo "DONE. Features in $WORK/features/20x_256px_0px_overlap/features_hoptimus0/"
