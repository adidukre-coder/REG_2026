"""
Sanity check before backfilling patch coordinates for the evidence-grounding feature:
re-run slide_to_features() with a PERSISTENT work_dir (so coords_h5 survives) for a small
handful of slides already in features_hoptimus0_pilot, and confirm the freshly extracted
features match the already-saved ones (cosine sim per row) -- i.e. that the coords we'd
persist really do align row-for-row with the features already used in every experiment
this session (attn_entropy_probe.py, eval_decode_ab.py, the 3 seeded checkpoints).

Zero cost to existing artifacts -- only reads the already-converted pyramidal TIFFs and
writes into a fresh scratch dir. Small GPU cost: H-Optimus-0 forward pass on ~5 slides.
"""
import os, sys, glob
os.environ.setdefault("REG_SEGMENTER", "otsu")
os.environ.setdefault("REG_MAX_PATCHES", "2000")
sys.path.insert(0, "cot_artifacts")
from src.interf1_pipeline import slide_to_features
import h5py, numpy as np, torch

PILOT_FEAT = "/nfs-stor/adinath.dukre/adinath/reg2026_work/features_hoptimus0_pilot"
CONV = "/nfs-stor/adinath.dukre/adinath/reg2026_work/converted_slides"
WORK = "/nfs-stor/adinath.dukre/adinath/reg2026_work/coord_backfill_check"
os.makedirs(WORK, exist_ok=True)

stems = sorted(os.path.basename(p)[:-3] for p in glob.glob(PILOT_FEAT + "/*.h5"))[:5]
print(f"checking {len(stems)} slides: {stems}", flush=True)

for stem in stems:
    wsi = os.path.join(CONV, stem + ".tiff")
    work = os.path.join(WORK, stem)
    os.makedirs(work, exist_ok=True)
    feats_new = slide_to_features(wsi, gpu=0, work_dir=work)

    with h5py.File(os.path.join(PILOT_FEAT, stem + ".h5")) as h:
        feats_old = torch.from_numpy(np.asarray(h["features"], dtype=np.float32))

    if feats_new.shape != feats_old.shape:
        print(f"  {stem}: SHAPE MISMATCH new={feats_new.shape} old={feats_old.shape}", flush=True)
        continue

    cos = torch.nn.functional.cosine_similarity(feats_new, feats_old, dim=-1)
    coords_glob = glob.glob(os.path.join(work, "job", "*", "patches", "*_patches.h5"))
    coords_info = "NOT FOUND"
    if coords_glob:
        with h5py.File(coords_glob[0]) as h:
            coords_info = f"coords shape={h['coords'].shape}"
    print(f"  {stem}: n={feats_new.shape[0]} mean_cos={float(cos.mean()):.4f} min_cos={float(cos.min()):.4f} | {coords_info}", flush=True)
