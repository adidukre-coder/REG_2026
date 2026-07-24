#!/usr/bin/env bash
#SBATCH --job-name=reg_diagcap
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=gpu-12,gpu-18,gpu-22,gpu-33,gpu-45,gpu-48,gpu-58
#SBATCH --output=/nfs-stor/adinath.dukre/adinath/reg2026_work/logs/diagcap_%j.out

# DIAGNOSTIC scoring: score the hest cap-sweep (full/2000/1000) on the SAME 140 GT held-out
# slides, interf1-only (Metric A). Reuses the official Qwen judge container. No submission, no
# format change. Predictions were pre-generated on the login node (src/diag_capsweep.py).
set -e
echo "node: $(hostname) | start: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

EVAL=/home/adinath.dukre/adinath/reg_2026/REG2026/submission_evaluation_code
LOGDIR=/nfs-stor/adinath.dukre/adinath/reg2026_work/logs
mkdir -p "$LOGDIR"

# rootless podman runtime (image store node-local /scratch; extraction TMPDIR on NFS)
SCR=/scratch/$(id -un)
export TMPDIR=/nfs-stor/adinath.dukre/adinath/reg2026_work/tmp
export XDG_RUNTIME_DIR="$SCR/podman-run"
mkdir -p "$TMPDIR" "$XDG_RUNTIME_DIR" "$SCR"

# health check: a broken/contended podman on this node hangs forever -> bail fast (exit 75 = retry)
echo "=== podman health check on $(hostname) ==="
if ! timeout 90 podman version >/dev/null 2>&1; then
  echo "!!! podman unresponsive on $(hostname) -> exit 75 (retry another node)"; exit 75
fi

IMG_TAR=/nfs-stor/adinath.dukre/adinath/reg2026_work/images/reg26-evaluation-sample.tar
if timeout 60 podman image exists localhost/reg26-evaluation-sample:latest; then
  echo "image already cached on $(hostname) -> no load needed"
else
  # prune is the known hang culprit -> hard timeout, never block on it
  timeout 90 podman system prune -f >/dev/null 2>&1 || echo "(prune skipped/timed out -- harmless)"
  rm -rf "$TMPDIR"/container_images_* 2>/dev/null || true
  FREE_MB=$(($(df --output=avail "$SCR" 2>/dev/null | tail -1)/1024))
  echo "=== /scratch free: ${FREE_MB} MB on $(hostname) ==="
  if [ "$FREE_MB" -lt 18000 ]; then
    echo "!!! /scratch too full on $(hostname) (${FREE_MB}MB, need ~18GB) -> exit 75 (retry another node)"; exit 75
  fi
  echo "=== loading eval image (TMPDIR=$TMPDIR) ==="
  timeout 900 podman load -i "$IMG_TAR" || { echo "!!! load failed on $(hostname) -> exit 75"; exit 75; }
fi
timeout 60 podman image exists localhost/reg26-evaluation-sample:latest && echo "image present on $(hostname)"

cd "$EVAL"
# nonexistent interf0 paths -> interf1-only (Metric A) scoring
NOPE_MAP=/tmp/diag_nope_mapping_$$.txt
NOPE_PRED=/tmp/diag_nope_preds_$$.json

for cap in full 2000 1000; do
  echo; echo "============================================================"
  echo "=== SCORING cap=$cap  ($(date)) ==="
  echo "============================================================"
  /bin/cp -f "data/predictions/interf1/predictions_hest_cap${cap}.json" "data/predictions/interf1/predictions.json"
  rm -f test/output/metrics.json 2>/dev/null || true
  # rely on ABSOLUTE defaults for interf1 (the prepare scripts cd into data/, so relative paths break);
  # only override interf0 with absolute nonexistent paths -> interf1-only (Metric A) scoring
  bash prepare_test_ground_truth.sh --interf0-mapping "$NOPE_MAP"
  bash prepare_test_input.sh --interf0 "$NOPE_PRED"
  GPU_MODE=cdi bash run_eval_local.sh
  if [ -f test/output/metrics.json ]; then
    cp test/output/metrics.json "$LOGDIR/diag_metrics_hest_cap${cap}.json"
    echo "--- cap=$cap metrics saved -> $LOGDIR/diag_metrics_hest_cap${cap}.json"
  else
    echo "!!! cap=$cap produced NO metrics.json"
  fi
done

echo; echo "================ SUMMARY (hest cap sweep, n=140) ================"
python3 - <<'PY'
import json, glob, os
LOG="/nfs-stor/adinath.dukre/adinath/reg2026_work/logs"
def grab(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d: d=d[k]
        else: return None
    return d
print(f"{'config':>14} | {'MetricA':>8} | {'edgeF1':>7} | {'MESS':>7} | {'report':>7} | {'pathV':>7}")
for cap in ["full","2000","1000"]:
    f=f"{LOG}/diag_metrics_hest_cap{cap}.json"
    if not os.path.exists(f): print(f"hest_cap{cap:>5} | (missing)"); continue
    m=json.load(open(f)); agg=m.get("aggregates",m)
    A   = grab(agg,"workflow_final_ranking_score")
    e   = grab(agg,"edge_f1")
    s   = grab(agg,"mess")
    r   = grab(agg,"final_report_score")
    p   = grab(agg,"binary_path_validity")
    def fmt(x): return f"{x:.4f}" if isinstance(x,(int,float)) else str(x)
    print(f"hest_cap{cap:>5} | {fmt(A):>8} | {fmt(e):>7} | {fmt(s):>7} | {fmt(r):>7} | {fmt(p):>7}")
PY
echo "end: $(date)"
