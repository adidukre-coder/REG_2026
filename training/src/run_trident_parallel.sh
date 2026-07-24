#!/usr/bin/env bash
# High-throughput extraction: shard the slide list into N parallel workers across 4 GPUs.
# The default 4-process run leaves GPUs at 0% (I/O-bound seg). This runs many readers at once.
#
# USAGE (on gpu-57, after KILLING the old run):
#   tmux new -s extract
#   bash src/run_trident_parallel.sh 16        # 16 parallel workers (4 per GPU)
#   # detach: Ctrl-b d   ; logs in work/logs/shard_*.out
#
# Safe to re-run: TRIDENT checkpoints per slide and skips finished ones.
set -e
N=${1:-16}                      # number of parallel workers
source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh
conda activate reg2026

DATA=/nfs-stor/adinath.dukre/adinath/reg2026
WORK=/nfs-stor/adinath.dukre/adinath/reg2026_work
SHARD_DIR="$WORK/shards"; mkdir -p "$SHARD_DIR" "$WORK/logs"
cd "$WORK/TRIDENT"

# 1) round-robin split of the organ-interleaved CSV into N shards (keeps each shard balanced)
python - "$N" <<'PY'
import sys
N=int(sys.argv[1])
src="/nfs-stor/adinath.dukre/adinath/reg2026_work/train_interleaved.csv"
lines=open(src).read().splitlines()
header,rows=lines[0],lines[1:]
outs=[open(f"/nfs-stor/adinath.dukre/adinath/reg2026_work/shards/shard_{i}.csv","w") for i in range(N)]
for o in outs: o.write(header+"\n")
for j,r in enumerate(rows): outs[j%N].write(r+"\n")
for o in outs: o.close()
print(f"split {len(rows)} slides into {N} shards (~{len(rows)//N} each)")
PY

# 2) launch N workers; worker i pinned to GPU (i % 4)
echo "launching $N workers across 4 GPUs ..."
for i in $(seq 0 $((N-1))); do
  GPU=$((i % 4))
  CUDA_VISIBLE_DEVICES=$GPU nohup python run_batch_of_slides.py \
      --task all \
      --wsi_dir "$DATA/train" \
      --custom_list_of_wsis "$SHARD_DIR/shard_${i}.csv" \
      --job_dir "$WORK/features" \
      --patch_encoder hoptimus0 \
      --mag 20 --patch_size 256 \
      --gpus 0 \
      --max_workers 2 \
      --skip_errors \
      > "$WORK/logs/shard_${i}.out" 2>&1 &
  echo "  worker $i -> GPU $GPU (log: work/logs/shard_${i}.out)"
  sleep 1
done
echo "all $N workers launched. Monitor with:"
echo "  ls $WORK/features/20x_256px_0px_overlap/features_hoptimus0/ | wc -l"
wait
