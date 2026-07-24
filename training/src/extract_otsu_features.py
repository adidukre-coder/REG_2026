"""
Re-extract H-Optimus features using the OTSU segmenter (+ MAX_PATCHES cap), so we can RETRAIN
the attribute model on otsu features and eliminate the train(hest)/infer(otsu) MISMATCH that
cost ~0.12 Metric A. Saves one <stem>.h5 (key 'features' [N,1536]) per slide -> a new feature
dir that REGFeatureDataset/analyze_performance can read by stem. Resumable (skips done files).

Reads the already-pyramidal converted_slides (so _ensure_readable passes through -> fast, and
features match training: validated cosine 0.9985 pyramidal-vs-original). Config via env:
  REG_SEGMENTER=otsu REG_SEG_WORKERS=0 REG_MAX_PATCHES=2000
Usage: python src/extract_otsu_features.py <chunk_idx> <n_chunks>
"""
import os, sys, glob, time, traceback, json
from collections import defaultdict, OrderedDict
import h5py
sys.path.insert(0, "cot_artifacts")
from src.interf1_pipeline import slide_to_features, SEGMENTER, MAX_PATCHES, SEG_WORKERS
# interf1_pipeline sets the 'file_system' sharing strategy (needed for the tiny CONTAINER shm),
# but that requires torch_shm_manager to create a socket dir, which FAILS on some compute nodes
# ("could not generate a random directory for manager socket"). For bulk extraction on compute
# nodes (which have a LARGE /dev/shm), revert to the default 'file_descriptor' strategy -> no
# socket manager, no per-node failures.
import torch
try:
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
except Exception:
    pass

OUT = os.environ.get("REG_OTSU_FEAT_DIR", "/nfs-stor/adinath.dukre/adinath/reg2026_work/features_otsu2000")
CONV = "/nfs-stor/adinath.dukre/adinath/reg2026_work/converted_slides"
LAB = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json"
os.makedirs(OUT, exist_ok=True)
idx, n = int(sys.argv[1]), int(sys.argv[2])

def _stem(p): return os.path.basename(p)[:-5]
# ORGAN-INTERLEAVED order so any partial extraction stays organ-BALANCED (valid early signal).
have = {_stem(p): p for p in glob.glob(CONV + "/*.tiff")}
organ = {r["id"].replace(".tiff", ""): r["organ"] for r in json.load(open(LAB))}
by = defaultdict(list)
for st, p in sorted(have.items()):
    by[organ.get(st, "UNK")].append(p)
interleaved = []
buckets = [iter(v) for v in by.values()]
done = False
while not done:
    done = True
    for it in buckets:
        nxt = next(it, None)
        if nxt is not None:
            interleaved.append(nxt); done = False
slides = interleaved[idx::n]
print(f"chunk {idx}/{n}: {len(slides)} slides | SEG={SEGMENTER} WORKERS={SEG_WORKERS} CAP={MAX_PATCHES} -> {OUT}", flush=True)

ok = skip = fail = 0
for sp in slides:
    stem = os.path.basename(sp)[:-5]
    out = os.path.join(OUT, stem + ".h5")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        skip += 1; continue
    try:
        t0 = time.time(); feats = slide_to_features(sp, gpu=0); dt = time.time() - t0
        if feats.shape[0] == 0:
            print(f"EMPTY {stem}", flush=True); fail += 1; continue
        tmp = out + ".tmp"
        with h5py.File(tmp, "w") as h:
            h.create_dataset("features", data=feats.numpy())
        os.replace(tmp, out)
        ok += 1
        if ok % 25 == 0:
            print(f"  chunk{idx}: {ok} done (last {stem} {feats.shape[0]}p {dt:.0f}s)", flush=True)
    except Exception as e:
        fail += 1; print(f"FAIL {stem} {type(e).__name__}: {e}", flush=True); traceback.print_exc()
print(f"chunk {idx} DONE: ok={ok} skip={skip} fail={fail}", flush=True)
