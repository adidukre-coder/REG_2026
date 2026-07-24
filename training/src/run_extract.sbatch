#!/bin/bash
# Unattended feature extraction as a SLURM BATCH job (survives logout; resumes via checkpoints).
# Submit:   sbatch src/run_extract.sbatch
# ADJUST --partition / --qos / --time to your cluster's limits (interactive srun was capped at 3h;
# a batch QoS usually allows longer). Re-submitting after a timeout just resumes.
#
#SBATCH --job-name=reg_extract
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs-stor/adinath.dukre/adinath/reg2026_work/logs/sbatch_extract_%j.out

set -e
source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh
conda activate reg2026
WORK=/nfs-stor/adinath.dukre/adinath/reg2026_work
DATA=/nfs-stor/adinath.dukre/adinath/reg2026
cd "$WORK/TRIDENT"

run_sharded() {   # $1=wsi_dir  $2=csv  $3=N_workers
  local WSIDIR="$1" CSV="$2" N="$3" SHARDS="$WORK/shards_$(basename $CSV .csv)"
  mkdir -p "$SHARDS"
  python - "$CSV" "$SHARDS" "$N" <<'PY'
import sys
csv,outdir,N=sys.argv[1],sys.argv[2],int(sys.argv[3])
lines=open(csv).read().splitlines(); hdr,rows=lines[0],lines[1:]
outs=[open(f"{outdir}/s{i}.csv","w") for i in range(N)]
[o.write(hdr+"\n") for o in outs]
[outs[j%N].write(r+"\n") for j,r in enumerate(rows)]
[o.close() for o in outs]
print(f"{len(rows)} slides -> {N} shards")
PY
  for i in $(seq 0 $((N-1))); do
    CUDA_VISIBLE_DEVICES=$((i%4)) nohup python run_batch_of_slides.py \
      --task all --wsi_dir "$WSIDIR" --custom_list_of_wsis "$SHARDS/s${i}.csv" \
      --job_dir "$WORK/features" --patch_encoder hoptimus0 \
      --mag 20 --patch_size 256 --gpus 0 --max_workers 2 --skip_errors \
      > "$WORK/logs/$(basename $CSV .csv)_s${i}.out" 2>&1 &
  done
  wait
}

# Build the slide list from what's ACTUALLY present in converted_slides (TRIDENT hard-errors
# on any missing slide), kept organ-interleaved so partial coverage stays balanced.
echo "=== building available (present-only) organ-interleaved CSV ==="
python - <<'PY'
import os, json, itertools
from collections import defaultdict
CONV="/nfs-stor/adinath.dukre/adinath/reg2026_work/converted_slides"
LAB ="/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json"
OUT ="/nfs-stor/adinath.dukre/adinath/reg2026_work/available_interleaved.csv"
labels={(r['id'] if str(r['id']).endswith('.tiff') else r['id']+'.tiff'):r['organ']
        for r in json.load(open(LAB))}
present=set(f for f in os.listdir(CONV) if f.endswith('.tiff'))
byorg=defaultdict(list)
for name,org in labels.items():
    if name in present: byorg[org].append(name)
for o in byorg: byorg[o].sort()
order=[x for tup in itertools.zip_longest(*byorg.values()) for x in tup if x]
with open(OUT,'w') as f:
    f.write('wsi,mpp\n'); [f.write(f'{w},0.5\n') for w in order]
print(f"available: {len(order)} slides present in converted_slides")
PY
echo "=== extracting (pyramidal, organ-interleaved, present-only) ==="
run_sharded "$WORK/converted_slides" "$WORK/available_interleaved.csv" 16

echo "CHUNK DONE. Features so far: $(ls $WORK/features/20x_256px_0px_overlap/features_hoptimus0 | wc -l) / 11209"
echo "If < 11209, re-submit (sbatch src/run_extract.sbatch) to resume."
