#!/usr/bin/env bash
#SBATCH --job-name=reg_score1
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=gpu-12,gpu-18,gpu-22,gpu-33,gpu-45,gpu-48,gpu-58
#SBATCH --output=/nfs-stor/adinath.dukre/adinath/reg2026_work/logs/score1_%j.out

# Generic single-config Metric-A scorer (interf1-only). Scores the predictions file given by
# $PRED_TAG: data/predictions/interf1/predictions_${PRED_TAG}.json -> diag_metrics_${PRED_TAG}.json
# Reuses the official Qwen judge container. Hardened podman (timeout-guarded, bail-fast exit 75).
set -e
TAG="${PRED_TAG:?set PRED_TAG}"
echo "node: $(hostname) | start: $(date) | TAG=$TAG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

EVAL=/home/adinath.dukre/adinath/reg_2026/REG2026/submission_evaluation_code
LOGDIR=/nfs-stor/adinath.dukre/adinath/reg2026_work/logs
SCR=/scratch/$(id -un)
export TMPDIR=/nfs-stor/adinath.dukre/adinath/reg2026_work/tmp
export XDG_RUNTIME_DIR="$SCR/podman-run"
mkdir -p "$TMPDIR" "$XDG_RUNTIME_DIR" "$SCR"

echo "=== podman health check on $(hostname) ==="
timeout 90 podman version >/dev/null 2>&1 || { echo "podman dead on $(hostname) -> exit 75"; exit 75; }
IMG_TAR=/nfs-stor/adinath.dukre/adinath/reg2026_work/images/reg26-evaluation-sample.tar
if timeout 60 podman image exists localhost/reg26-evaluation-sample:latest; then
  echo "image cached on $(hostname)"
else
  timeout 90 podman system prune -f >/dev/null 2>&1 || echo "(prune skipped)"
  rm -rf "$TMPDIR"/container_images_* 2>/dev/null || true
  FREE_MB=$(($(df --output=avail "$SCR" 2>/dev/null | tail -1)/1024))
  echo "/scratch free: ${FREE_MB} MB"
  [ "$FREE_MB" -lt 18000 ] && { echo "scratch too full on $(hostname) -> exit 75"; exit 75; }
  timeout 900 podman load -i "$IMG_TAR" || { echo "load failed -> exit 75"; exit 75; }
fi

cd "$EVAL"
NOPE=/tmp/score1_nope_$$
/bin/cp -f "data/predictions/interf1/predictions_${TAG}.json" "data/predictions/interf1/predictions.json"
rm -f test/output/metrics.json 2>/dev/null || true
bash prepare_test_ground_truth.sh --interf0-mapping "${NOPE}.txt"
bash prepare_test_input.sh --interf0 "${NOPE}.json"
GPU_MODE=cdi bash run_eval_local.sh

if [ -f test/output/metrics.json ]; then
  cp test/output/metrics.json "$LOGDIR/diag_metrics_${TAG}.json"
  echo "--- $TAG metrics saved -> $LOGDIR/diag_metrics_${TAG}.json"
  python3 -c "import json;m=json.load(open('$LOGDIR/diag_metrics_${TAG}.json'))['aggregates'];print('$TAG: MetricA=%.4f | edge=%.4f mess=%.4f report=%.4f pathV=%.4f'%(m['workflow_final_ranking_score'],m['edge_f1'],m['mess'],m['final_report_score'],m['binary_path_validity']))"
else
  echo "!!! $TAG produced NO metrics.json"
fi
echo "end: $(date)"
