#!/usr/bin/env bash
#SBATCH --job-name=reg_otsufeat
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@150
#SBATCH --exclude=gpu-12,gpu-17,gpu-22,gpu-33,gpu-45
#SBATCH --array=0-6
#SBATCH --output=/nfs-stor/adinath.dukre/adinath/reg2026_work/logs/otsufeat_%a.out
source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh; conda activate reg2026
cd /home/adinath.dukre/adinath/reg_2026/REG2026
export PYTHONPATH=$PWD
export TMPDIR=/scratch/$(id -un)/regtmp; mkdir -p "$TMPDIR" 2>/dev/null || export TMPDIR=/nfs-stor/adinath.dukre/adinath/reg2026_work/tmp
export HF_HOME=/nfs-stor/adinath.dukre/adinath/reg2026_work/hf_cache
export TRIDENT_HOME=/nfs-stor/adinath.dukre/adinath/reg2026_work/trident_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export REG_SEGMENTER=otsu REG_SEG_WORKERS=0 REG_MAX_PATCHES=2000
export REG_FEAT_WORKERS=8

OTSU=/nfs-stor/adinath.dukre/adinath/reg2026_work/features_otsu2000
TARGET=10900   # ~all extractable slides (a few have no tissue and never produce)
# SELF-RESUBMIT on SLURM timeout (durable — survives any external watcher dying). Exactly ONE
# task per array-generation wins the lock and resubmits the next generation if work remains.
on_term() {
  LOCK=/nfs-stor/adinath.dukre/adinath/reg2026_work/otsu_resub_${SLURM_ARRAY_JOB_ID}.lock
  if mkdir "$LOCK" 2>/dev/null; then
    N=$(ls "$OTSU"/*.h5 2>/dev/null | wc -l)
    if [ "$N" -lt "$TARGET" ]; then
      (cd /home/adinath.dukre/adinath/reg_2026/REG2026 && sbatch src/extract_otsu_features.sbatch) \
        && echo "[task ${SLURM_ARRAY_TASK_ID}] SELF-RESUBMITTED next generation at N=$N"
    else
      echo "[task ${SLURM_ARRAY_TASK_ID}] extraction complete (N=$N) -> no resubmit"
    fi
  fi
  exit 0
}
trap on_term TERM

echo "node: $(hostname) | $(date) | task ${SLURM_ARRAY_TASK_ID} | features so far: $(ls "$OTSU"/*.h5 2>/dev/null | wc -l)"
python src/extract_otsu_features.py ${SLURM_ARRAY_TASK_ID} 7 &
wait $!
echo "[task ${SLURM_ARRAY_TASK_ID}] python finished normally at $(date)"
