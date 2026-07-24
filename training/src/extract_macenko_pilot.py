"""
Pilot Macenko stain-normalization extraction, on the SAME 420-slide pilot set used by
extract_hoptimus1_pilot.py (identical selection logic: same organ map, same eligible set,
same random.Random(0) shuffle, same PER_ORGAN) -- so this is directly paired against BOTH
features_hoptimus0_pilot (baseline) and features_hoptimus1_pilot (encoder swap). Isolates
stain normalization as the only variable changed (encoder=hoptimus0, same segmenter=otsu,
same MAX_PATCHES=2000 -- only REG_STAIN_NORM differs from extract_otsu_features.py).

Usage: python src/extract_macenko_pilot.py <chunk_idx> <n_chunks> [PER_ORGAN]
Writes to REG_MACENKO_PILOT_DIR (default /nfs-stor/adinath.dukre/adinath/reg2026_work/features_hoptimus0_macenko_pilot)
"""
import os, sys, glob, time, random, traceback, json
from collections import defaultdict
import h5py

os.environ["REG_ENCODER"] = "hoptimus0"   # baseline encoder -- only stain norm differs
os.environ["REG_STAIN_NORM"] = "macenko"  # the only variable that differs from extract_otsu_features.py
os.environ.setdefault("REG_SEGMENTER", "otsu")
os.environ.setdefault("REG_MAX_PATCHES", "2000")
os.environ.setdefault("REG_SEG_WORKERS", "0")

sys.path.insert(0, "cot_artifacts")
from src.interf1_pipeline import slide_to_features
import torch
try:
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
except Exception:
    pass

OUT = os.environ.get("REG_MACENKO_PILOT_DIR", "/nfs-stor/adinath.dukre/adinath/reg2026_work/features_hoptimus0_macenko_pilot")
CONV = "/nfs-stor/adinath.dukre/adinath/reg2026_work/converted_slides"
OTSU0 = "/nfs-stor/adinath.dukre/adinath/reg2026_work/features_otsu2000"
LAB = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json"
os.makedirs(OUT, exist_ok=True)

idx, n = int(sys.argv[1]), int(sys.argv[2])
PER_ORGAN = int(sys.argv[3]) if len(sys.argv) > 3 else 60

organ = {r["id"].replace(".tiff", ""): r["organ"] for r in json.load(open(LAB))}
have_h0 = {os.path.basename(p)[:-3] for p in glob.glob(OTSU0 + "/*.h5")}  # stems with existing H-Optimus-0 features
have_conv = {os.path.basename(p)[:-5] for p in glob.glob(CONV + "/*.tiff")}
eligible = sorted(have_h0 & have_conv)

by_organ = defaultdict(list)
for st in eligible:
    by_organ[organ.get(st, "UNK")].append(st)

rng = random.Random(0)  # identical to extract_hoptimus1_pilot.py -> same 420 stems
picked = []
for o, stems in sorted(by_organ.items()):
    if o == "UNK":
        continue
    s = sorted(stems)
    rng.shuffle(s)
    picked += s[:PER_ORGAN]
picked = sorted(picked)

slides = picked[idx::n]
print(f"chunk {idx}/{n}: {len(slides)}/{len(picked)} pilot slides (PER_ORGAN={PER_ORGAN}) "
      f"ENCODER={os.environ['REG_ENCODER']} STAIN_NORM={os.environ['REG_STAIN_NORM']} "
      f"SEG={os.environ['REG_SEGMENTER']} CAP={os.environ['REG_MAX_PATCHES']} -> {OUT}", flush=True)

ok = skip = fail = 0
for stem in slides:
    sp = os.path.join(CONV, stem + ".tiff")
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
        print(f"  chunk{idx}: {ok} done (last {stem} {feats.shape[0]}p {dt:.0f}s)", flush=True)
    except Exception as e:
        fail += 1; print(f"FAIL {stem} {type(e).__name__}: {e}", flush=True); traceback.print_exc()
print(f"chunk {idx} DONE: ok={ok} skip={skip} fail={fail}", flush=True)
