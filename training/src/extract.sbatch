#!/bin/bash
# SELF-RESUBMITTING feature extraction. Submit ONCE: sbatch src/extract.sbatch
# It runs ~3h, then automatically resubmits the next chunk before the timeout, and keeps
# going chunk-after-chunk until extraction is complete (a run that adds no new features),
# then stops on its own. Runs on SLURM -> survives logout / no babysitting needed.
#
#SBATCH --job-name=reg_extract
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu-45
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=/nfs-stor/adinath.dukre/adinath/reg2026_work/logs/extract_%j.out

REPO=/home/adinath.dukre/adinath/reg_2026/REG2026
W=/nfs-stor/adinath.dukre/adinath/reg2026_work
FEATDIR=$W/features/20x_256px_0px_overlap/features_hoptimus0
cnt(){ ls "$FEATDIR" 2>/dev/null | wc -l; }
START=$(cnt)

# STEP 1 = EXTRACTION ONLY (no auto-training). Training is a deliberate, separate next step.
# Resubmit the next chunk ONLY if this run produced new features (progress).
# A run that adds nothing => extraction is complete (or stuck) => stop the chain.
resubmit_if_progress(){
  if [ -f "$W/STOP_EXTRACTION" ]; then
    echo "[auto] STOP_EXTRACTION present -> not resubmitting."; return; fi
  local n; n=$(cnt)
  if [ "$n" -gt "$START" ]; then
    rm -f "$W/noprogress_count"
    echo "[auto] progress: $START -> $n. Resubmitting next chunk..."
    cd "$REPO" && sbatch src/extract.sbatch
  else
    # No progress THIS run could be a server kill (NOT necessarily 'done'). Retry a few times;
    # only stop after 3 CONSECUTIVE no-progress runs (= truly complete or genuinely stuck).
    local k; k=$(( $(cat "$W/noprogress_count" 2>/dev/null || echo 0) + 1 )); echo "$k" > "$W/noprogress_count"
    if [ "$k" -ge 3 ]; then
      echo "[auto] no progress x$k consecutive -> STOPPING (likely complete or stuck — check logs)."
    else
      echo "[auto] no progress (attempt $k/3) — likely a server kill; resubmitting to retry..."
      cd "$REPO" && sbatch src/extract.sbatch
    fi
  fi
}
# On SLURM timeout (TERM ~180s before kill): stop driver, resubmit, exit cleanly.
trap 'echo "[auto] timeout signal received"; kill $DRIVER_PID 2>/dev/null; resubmit_if_progress; exit 0' TERM

source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh
conda activate reg2026
# Durable model caches on /nfs-stor (survive /home wipes) + offline (no token dependency).
export HF_HOME="$W/hf_cache"
export TRIDENT_HOME="$W/trident_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$REPO"
echo "[auto] node $(hostname) | start $(date) | features at start: $START"
# pre-flight: H-Optimus must load (from /nfs-stor cache) or ABORT fast (don't churn empty jobs).
if ! python -c "import timm; timm.create_model('hf-hub:bioptimus/H-optimus-0',pretrained=True,init_values=1e-5,dynamic_img_size=False)" 2>/dev/null; then
  echo "[auto] H-Optimus FAILED to load -> halting loop. Fix the /nfs-stor HF cache."; touch "$W/STOP_EXTRACTION"; exit 1
fi
find "$W/features" -name "*.lock" -delete   # clear stale locks from any prior killed run
python src/extract_driver.py --shard_size 30 --parallel 8 &
DRIVER_PID=$!
wait $DRIVER_PID
echo "[auto] driver finished naturally at $(date)"
resubmit_if_progress
