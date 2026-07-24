"""
Interface-1 single-slide pipeline: raw WSI -> H-Optimus features -> CoT.
This is what runs INSIDE the submission container's predict_chain_of_thought.

Replicates the training feature extraction EXACTLY (TRIDENT hest-seg, hoptimus0 encoder,
20x / 256px / 0 overlap, mpp forced to 0.5 for the bogus-metadata challenge TIFFs) so the
inference features match what the attribute model was trained on. Uses the TRIDENT Python
API (load_wsi(mpp=0.5) -> segment_tissue -> extract_tissue_coords -> extract_patch_features),
not a subprocess, so it is container-friendly.

Weight/cache locations come from env (HF_HOME, TRIDENT_HOME); set them to /opt/ml/model in
the container. Falls back to a pyvips pyramidal conversion if openslide can't read the slide
(AdobeDeflate-compressed slides — same fix as training).
"""
from __future__ import annotations
import os, tempfile, shutil
from pathlib import Path
import h5py, numpy as np, torch
# Use file-based tensor sharing for DataLoader workers -> parallel loading WITHOUT /dev/shm
# (the small container shm crashes shm-based workers). This lets segmentation use many workers.
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

SEG_WORKERS = int(os.environ.get("REG_SEG_WORKERS", "0"))  # 0 = no DataLoader workers (shm-safe)
SEG_BATCH = 64

MPP = 0.5            # challenge TIFFs report bogus mpp -> force 20x = 0.5 um/px
MAG = 20
PATCH = 256
# Otsu tissue seg: CPU thresholding on the pyramidal low-res level -> fast AND no DataLoader
# workers (so the container's tiny /dev/shm can't crash it). Predicts the same organs as the
# training hest seg on all debug slides; worst slide ~170s on T4 (limit 300).
SEGMENTER = os.environ.get("REG_SEGMENTER", "otsu")
ENCODER = "hoptimus0"
MAX_PATCHES = 1000   # cap so feature extraction finishes well under the 5-min/case T4 limit (margin for slow nodes)
BATCH_LIMIT = 32


def _cap_coords(coords_h5: str, max_n: int = MAX_PATCHES) -> int:
    with h5py.File(coords_h5, "r") as f:
        coords = f["coords"][:]; attrs = dict(f["coords"].attrs)
    n = int(len(coords))
    if n <= max_n:
        return n
    idx = np.linspace(0, n - 1, max_n).astype(np.int64)
    coords = coords[idx]
    with h5py.File(coords_h5, "w") as f:
        d = f.create_dataset("coords", data=coords)
        for k, v in attrs.items(): d.attrs[k] = v
    print(f"[interf1] capped patches {n} -> {max_n}", flush=True)
    return max_n


def _ensure_readable(wsi_path: str, work: str) -> str:
    """Ensure the slide is PYRAMIDAL (multi-level) + openslide-readable.

    The challenge TIFFs are single-level full-res images; segmenting/patching them forces a
    downsample-from-full-res read of the whole (often >1GB) file -> ~250s I/O-bound (the real
    timeout cause). Converting to a pyramidal tiled JPEG (pre-built low-res levels) makes seg
    read a small level -> fast. Same fix as training. Already-pyramidal slides pass through.
    AdobeDeflate slides (openslide can't read) are also handled here.
    """
    try:
        import openslide
        s = openslide.OpenSlide(wsi_path)
        levels = s.level_count
        s.close()
        if levels > 1:
            return wsi_path                       # already pyramidal -> fast access
        print(f"[interf1] slide is single-level -> converting to pyramidal", flush=True)
    except Exception as e:
        print(f"[interf1] openslide can't read ({type(e).__name__}) -> converting to pyramidal", flush=True)
    import pyvips
    out = os.path.join(work, Path(wsi_path).stem + "_pyr.tiff")
    img = pyvips.Image.new_from_file(wsi_path, access="sequential")
    img.tiffsave(out, tile=True, pyramid=True, compression="jpeg", Q=90,
                 tile_width=256, tile_height=256, bigtiff=True)
    return out


def slide_to_features(wsi_path: str, gpu: int = 0, work_dir: str | None = None) -> torch.Tensor:
    """raw WSI -> [N, 1536] H-Optimus feature tensor (CPU)."""
    from trident import load_wsi
    from trident.segmentation_models import segmentation_model_factory
    from trident.patch_encoder_models import encoder_factory

    work = work_dir or tempfile.mkdtemp(prefix="interf1_")
    job = os.path.join(work, "job"); os.makedirs(job, exist_ok=True)
    dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    path = _ensure_readable(wsi_path, work)

    with load_wsi(slide_path=path, lazy_init=False, mpp=MPP) as slide:
        seg = segmentation_model_factory(SEGMENTER, confidence_thresh=0.5)
        seg_dev = "cpu" if SEGMENTER == "otsu" else dev
        slide.segment_tissue(segmentation_model=seg, target_mag=seg.target_mag,
                             job_dir=job, device=seg_dev, holes_are_tissue=True,
                             num_workers=SEG_WORKERS, batch_size=SEG_BATCH)
        mag_str = f"{float(MAG):g}"
        save_coords = os.path.join(job, f"{mag_str}x_{PATCH}px_0px_overlap")
        slide.extract_tissue_coords(target_mag=MAG, patch_size=PATCH, save_coords=save_coords)
        coords_h5 = os.path.join(save_coords, "patches", f"{slide.name}_patches.h5")
        _cap_coords(coords_h5)
        enc = encoder_factory(ENCODER); enc.eval().to(dev)
        feat_dir = os.path.join(save_coords, f"features_{ENCODER}")
        slide.extract_patch_features(
            patch_encoder=enc,
            coords_path=coords_h5,
            save_features=feat_dir, device=dev, batch_limit=BATCH_LIMIT,
        )
        h5p = os.path.join(feat_dir, f"{slide.name}.h5")

    with h5py.File(h5p) as h:
        feats = torch.from_numpy(np.asarray(h["features"], dtype=np.float32))
    print(f"[interf1] extracted {feats.shape[0]} patches x {feats.shape[1]}-d", flush=True)
    return feats


def predict_cot(wsi_path: str, ckpt_path: str, gpu: int = 0, work_dir: str | None = None):
    """Full Interface-1: raw WSI -> chain-of-thought list."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cot_artifacts"))
    from src.interf1_infer import load_model, build_organ_answer_to_key, predict_chain_of_thought_from_features
    from traversal_engine import load_graph

    feats = slide_to_features(wsi_path, gpu=gpu, work_dir=work_dir)
    if feats.shape[0] == 0:                          # no tissue detected -> caller falls back
        raise ValueError("no tissue patches extracted from slide")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, lm = load_model(ckpt_path, dev)
    feats = feats.to(dev)
    graph, roots = load_graph()
    omap = build_organ_answer_to_key()
    return predict_chain_of_thought_from_features(feats, model, lm, graph, roots, omap, threshold=0.25)
