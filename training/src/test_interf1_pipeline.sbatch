#!/usr/bin/env bash
#SBATCH --job-name=reg_if1pipe
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=gpu-12,gpu-33,gpu-45
#SBATCH --output=/nfs-stor/adinath.dukre/adinath/reg2026_work/logs/if1pipe_%j.out

# Phase-1 validation: run the single-slide interf1 pipeline on the template's sample WSI
# in the conda env (TRIDENT + H-Optimus already working) and print the CoT + timing.
set -e
echo "node: $(hostname) | start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh
conda activate reg2026
cd /home/adinath.dukre/adinath/reg_2026/REG2026
export PYTHONPATH=$PWD
export HF_HOME=/nfs-stor/adinath.dukre/adinath/reg2026_work/hf_cache
export TRIDENT_HOME=/nfs-stor/adinath.dukre/adinath/reg2026_work/trident_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python - <<'PY'
import time, json
from src.interf1_pipeline import predict_cot
SLIDE="algorithm_submission_template/test/input/interf1/images/whole-slide-image/d021e460-42d8-4a72-b83e-f07050d8468a.tiff"
CK="/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints/reg_attr_analysis.pt"
t0=time.time()
cot=predict_cot(SLIDE, CK, gpu=0)
dt=time.time()-t0
print(f"\n=== CoT produced in {dt:.1f}s | {len(cot)} steps ===")
for s in cot[:6]:
    print(f"  Q: {s['question']}")
    print(f"     A: {s['answer'][:70]}  -> next: {s['next_question'][:45]}")
print("  ...")
last=cot[-1]
print(f"  LAST: {last['question']} | next_q empty: {last['next_question']==''}")
print(f"     report: {last['answer'][:90]}")
print(f"\nPER-CASE TIME = {dt:.1f}s  (budget check)")
PY
echo "end: $(date)"
