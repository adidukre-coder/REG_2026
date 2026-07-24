#!/bin/bash
# STAGE 2 — TRAIN the Interface-1 attribute model (BATCH job; submit when features >= ~4000).
# Submit:  sbatch src/train.sbatch
# Watch:   tail -f /nfs-stor/adinath.dukre/adinath/reg2026_work/logs/train_<jobid>.out
#          (you'll see per-epoch: train_loss, val_acc, gates[abnorm/neopla/...])
#
#SBATCH --job-name=reg_train
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs-stor/adinath.dukre/adinath/reg2026_work/logs/train_%j.out

source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh
conda activate reg2026
cd /home/adinath.dukre/adinath/reg_2026/REG2026
export PYTHONPATH="/home/adinath.dukre/adinath/reg_2026/REG2026:${PYTHONPATH:-}"
echo "node: $(hostname) | start: $(date)"
python src/train.py \
  --feat_dir /nfs-stor/adinath.dukre/adinath/reg2026_work/features/20x_256px_0px_overlap/features_hoptimus0 \
  --epochs 40 --tag v1
echo "end: $(date)"
# checkpoint -> /nfs-stor/.../reg2026_work/checkpoints/reg_attr_v1_best.pt
