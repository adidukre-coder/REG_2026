# `model/` — weights (download separately)

Not committed to this repository (exceeds GitHub's file-size limits). Download and extract here
before building the container.

## Download

Full `model/` directory, packaged exactly as used in the validated submission container:

**https://huggingface.co/adidukrembzuai/reg2026-algorithm-candidate-pertask/blob/main/model.tar.gz**

```bash
cd submission/
curl -L -o model.tar.gz "https://huggingface.co/adidukrembzuai/reg2026-algorithm-candidate-pertask/resolve/main/model.tar.gz"
mkdir -p model
tar -xzf model.tar.gz -C model
```

## Expected contents after extraction

| File/dir | What it is |
|---|---|
| `reg_attr_analysis.pt` | Trained per-task attention attribute-head checkpoint (loaded via `MODEL_PATH` in `core.py`). |
| `hf_cache/`, `torch_cache/`, `trident_cache/` | Pre-populated caches for the H-Optimus-0 foundation model and TRIDENT's segmentation model, so the container can run fully offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` in the Dockerfile — Grand Challenge runs containers with `--network none`). |

If you only need the trained checkpoint (not full offline reproduction), `reg_attr_analysis.pt`
alone is ~288MB; the caches can instead be repopulated by removing the offline env vars in the
Dockerfile and allowing network access on first run.
