#!/bin/bash
#SBATCH --job-name=reg_perf
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-12,gpu-45
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs-stor/adinath.dukre/adinath/reg2026_work/logs/perf_%j.out
source /home/adinath.dukre/miniconda3/etc/profile.d/conda.sh
conda activate reg2026
cd /home/adinath.dukre/adinath/reg_2026/REG2026
export PYTHONPATH=/home/adinath.dukre/adinath/reg_2026/REG2026
echo "node: $(hostname) | start: $(date)"
python -c "import torch;print('cuda',torch.cuda.is_available())"
# balanced ~1600/organ (~10k slides), 20 epochs — real perf + saves checkpoint
python src/analyze_performance.py 1600 20
echo "end: $(date)"
