# REG2026 Submission — Faithful Per-Task Attention for CoT Pathology Report Generation

Code submission for the [REG2026 Challenge](https://reg2026.grand-challenge.org) (Pathologist
REasoning-Guided REport Generation), provided per the challenge's code-submission requirements.

## Team

- **Team Name:** GenMI
- **Team Members:** Gagneet Singh, Adinath Dukre, Imran Razzaq
- **Grand Challenge Profile URL(s):** https://grand-challenge.org/users/adinath@dukre/
- **Grand Challenge Username(s):** adinath@dukre

## Method (brief)

We untie MIL attention pooling per question instead of sharing one attention pool across all 93
diagnostic-attribute heads: each head learns its own gated-attention pool (Ilse et al., 2018-style)
over the same frozen H-Optimus-0 patch features. Predicted answers drive a deterministic
decision-graph traversal and a rule-based report composer. Full method, ablations, and the complete
development trajectory (a segmentation train/inference mismatch fix, an independent evaluation-
formula audit, and our real Test Phase 1 result) are described in the accompanying paper.

## Repository layout

| Directory | What it is |
|---|---|
| [`training/`](training/) | Training pipeline: feature extraction, attribute-head training (shared and per-task attention), evaluation/ablation scripts. Mirrors the `src/` layout used during development. |
| [`submission/`](submission/) | The exact Docker container source submitted to Grand Challenge (Interface 0 + Interface 1), minus model weights and build artifacts (see below). |

## Reproducing inference (the submitted container)

1. `cd submission/`
2. Fetch the vendored [TRIDENT](https://github.com/mahmoodlab/TRIDENT) package into `submission/trident/`
   (the Dockerfile expects it there; our copy is an unmodified checkout, so a fresh clone/pip
   install of TRIDENT is equivalent — see `submission/README.md`).
3. Download and extract model weights into `submission/model/` (see
   [`submission/model/README.md`](submission/model/README.md)).
4. `./do_build.sh` to build the image, then `./do_test_run.sh` to run both interfaces against the
   sample cases in `submission/test/input/` (add your own test WSIs there first).

Full details, environment/dependency notes, and the exact base image are in
[`submission/README.md`](submission/README.md) and [`submission/Dockerfile`](submission/Dockerfile).

## Reproducing training

See [`training/README.md`](training/README.md) for the training pipeline layout, and
`training/src/train_v2_resumable.py` (segmentation-fixed baseline) /
`training/src/train_pertask_pilot.py` (per-task attention architecture) for the two model variants
discussed in the paper. Training expects the REG2026 challenge dataset and TRIDENT-extracted
H-Optimus-0 features; paths are configured via environment variables at the top of each script.

## Model weights

Not included in this repository (see [`submission/model/README.md`](submission/model/README.md)
for the download link and layout) — the full `model/` directory (trained attribute-head weights
plus offline-inference caches) is hosted externally as it exceeds GitHub's file-size limits.

## Environment / dependencies

- **Training:** Python 3.11.15. [`training/requirements.txt`](training/requirements.txt) is a full
  `pip freeze` of the exact validated conda environment (153 packages, includes `torch==2.5.1+cu121`,
  `timm==0.9.16`, `transformers==4.57.6`, `trident==0.3.0`, `einops`, `h5py`, `openslide-python`,
  `pyvips`, `geopandas`, `shapely`, `scikit-image`, etc.). Setup:
  ```bash
  conda create -n reg2026 python=3.11
  conda activate reg2026
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  pip install -r training/requirements.txt
  ```
  (`torch`/`torchvision` need the CUDA-specific index URL above; installing them from
  `requirements.txt` directly via plain `pip install -r` will fail to resolve the `+cu121` build.)
- **Inference (container):** see `submission/requirements.txt` and `submission/Dockerfile` — pinned
  to `torch==2.5.1+cu121` to match the same validated environment (H-Optimus-0 + `timm==0.9.16`);
  `torch`/`torchvision` come from the Docker base image, not `requirements.txt`, inside the
  container.
