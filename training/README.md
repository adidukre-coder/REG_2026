# Training pipeline

Scripts used to extract features, train the attribute heads (both the shared-attention baseline
and the per-task attention architecture), and run the evaluations/ablations reported in the paper.

## Key scripts

| Script | What it does |
|---|---|
| `extract_driver.py`, `extract_otsu_features.py`, `extract_hoptimus1_pilot.py`, `extract_macenko_pilot.py` | Feature extraction via TRIDENT (tissue segmentation → tiling → H-Optimus-0 encoding), including the segmentation-method variants discussed in Sect. 3.1 of the paper (learned segmenter vs. Otsu thresholding). |
| `train.py`, `train_v2.py`, `train_v2_resumable.py` | Shared-attention baseline training (Eq. 1), including the segmentation-fixed retrain. |
| `train_pertask_pilot.py` | Per-task attention architecture training (Eq. 2) — the full-corpus 3-seed ablation and the production-recipe retrain both use this script with different `REG_PER_ORGAN`/epoch settings (see paper Sect. 3.2). |
| `train_pertask_diversity.py` | The diversity-regularizing-loss variant screened in the negative-results table (Sect. 3.5). |
| `train_shared_resumable.py`, `train_otsu.py`, `train_otsu_labelfix.py` | Additional baseline/ablation training variants. |
| `compose_report.py` | Rule-based report composer (shared with the submission container). |
| `interf1_pipeline.py`, `interf1_infer.py` | Single-slide inference pipeline used both for local evaluation and as the basis for the submission container's Interface 1. |
| `eval_only.py`, `eval_v2_vs_baseline.py`, `eval_v2full_vs_baseline.py`, `entropy_gated_eval.py`, `entropy_gated_eval_v2.py` | Evaluation scripts computing the official Metric A/B formula against held-out data. |
| `faithfulness_ablation.py`, `attn_entropy_probe.py` | The causal masking ablation and attention-entropy analysis (Sect. 3.3-3.4). |
| `analyze_performance.py`, `analyze_performance_seeded.py`, `novelty_analysis_full.py` | Per-category accuracy breakdowns and multi-seed aggregation. |
| `beam_decode.py`, `eval_decode_ab.py` | The confidence-weighted decoding variant screened in the negative-results table. |

`.sbatch` files alongside each script are the Slurm job scripts used to run them on our cluster;
adapt the resource requests and paths for your own environment.

## Data layout

Scripts expect the REG2026 challenge dataset (WSIs + CoT labels) and TRIDENT-extracted H-Optimus-0
features on local/network storage; paths are set via environment variables or constants at the top
of each script (`REG_FEAT_DIR`, `REG_OUT_CKPT`, etc. — see individual `.sbatch` files for the
variables each script expects). Model checkpoints referenced elsewhere in this repo (e.g. for
inference) are produced by these scripts, not included here — see
[`../submission/model/README.md`](../submission/model/README.md).
