#!/usr/bin/env bash
# Smoke-test TRIDENT feature extraction on 5 slides. RUN THIS ON gpu-57 (needs GPU).
#   conda activate reg2026 && bash src/run_trident_test5.sh
set -e

source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh
conda activate reg2026

DATA=/nfs-stor/adinath.dukre/adinath/reg2026
WORK=/nfs-stor/adinath.dukre/adinath/reg2026_work
cd "$WORK/TRIDENT"

echo "=== GPU visible? ==="
python -c "import torch; print('cuda', torch.cuda.is_available(), '| devices', torch.cuda.device_count())"

echo "=== TRIDENT: seg -> coords -> feat on 5 slides (H-Optimus-0, 20x, 256px) ==="
python run_batch_of_slides.py \
  --task all \
  --wsi_dir "$DATA/train" \
  --custom_list_of_wsis "$WORK/test5_wsis.csv" \
  --job_dir "$WORK/features_test5" \
  --patch_encoder hoptimus0 \
  --mag 20 --patch_size 256 \
  --gpus 0

echo; echo "=== OUTPUT TREE ==="
find "$WORK/features_test5" -maxdepth 2 -type d
echo; echo "=== feature files produced ==="
find "$WORK/features_test5" -name "*.h5" | head
echo; echo "=== inspect one feature file shape ==="
python - <<'PY'
import h5py, glob, os
fs = glob.glob("/nfs-stor/adinath.dukre/adinath/reg2026_work/features_test5/**/*.h5", recursive=True)
feats = [f for f in fs if "feat" in f.lower() or "h_optimus" in f.lower() or "hoptimus" in f.lower()]
for f in (feats or fs)[:6]:
    try:
        with h5py.File(f,"r") as h:
            keys=list(h.keys())
            shapes={k: h[k].shape for k in keys}
            print(os.path.basename(f), "->", shapes)
    except Exception as e:
        print(os.path.basename(f), "ERR", e)
PY
