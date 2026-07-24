"""
DIAGNOSTIC B (no submission, no format change): run the TRUE submitted-container pipeline
(otsu seg + MAX_PATCHES=1000, via interf1_pipeline.predict_cot) on the SAME 140 GT held-out
val slides, to measure the EXACT submitted-config Metric A on data we have ground truth for.

Compared with the hest cap-sweep (diag_capsweep.py):
  - otsu1000 vs hest1000  -> isolates the otsu-vs-hest SEG effect
  - otsu1000 vs official Debug 0.705 -> tells us how much of the official gap is config (this
    matches) vs distribution/7-slide noise (this is higher).

Reproduces the split EXACTLY (seed 0, 1600/organ, 0.18 val, 20/organ held-out) so the slides
are IDENTICAL to the 0.850 baseline. Writes one CoT json per slide (resumable).
Usage: python src/diag_otsu_reextract.py <chunk_idx> <n_chunks>
"""
import json, os, random, sys, time, glob, traceback
from collections import defaultdict
sys.path.insert(0, "cot_artifacts")
from src.dataset import REGFeatureDataset
from src.interf1_pipeline import predict_cot

FEAT="/nfs-stor/adinath.dukre/adinath/reg2026_work/features/20x_256px_0px_overlap/features_hoptimus0"
LAB ="/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json"
LM  ="/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/label_maps.json"
CK  ="/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints/reg_attr_analysis.pt"
CONVERTED="/nfs-stor/adinath.dukre/adinath/reg2026_work/converted_slides"
TRAIN="/nfs-stor/adinath.dukre/adinath/reg2026/train"
CAP=os.environ.get("REG_MAX_PATCHES","2000")   # tag output dir by cap so 1000/2000 runs don't collide
OUTDIR=f"/nfs-stor/adinath.dukre/adinath/reg2026_work/diag_otsu/cots_otsu{CAP}"
os.makedirs(OUTDIR, exist_ok=True)
PER_EVAL=20
idx=int(sys.argv[1]); n=int(sys.argv[2])

lm=json.load(open(LM))
ds=REGFeatureDataset(FEAT, LAB, lm)
by=defaultdict(list)
for i,s in enumerate(ds.samples): by[s["organ"]].append(i)
rng=random.Random(0); idxs=[]
for o,ii in by.items():
    rng.shuffle(ii); idxs+=ii[:1600]
rng.shuffle(idxs)
nval=int(0.18*len(idxs))
held=defaultdict(list)
for i in idxs[:nval]:
    o=ds.samples[i]["organ"]
    if len(held[o])<PER_EVAL: held[o].append(i)
eval_idx=[i for v in held.values() for i in v]
stems=sorted(ds.samples[i]["id"] for i in eval_idx)   # deterministic order for chunking
mine=stems[idx::n]
print(f"chunk {idx}/{n}: {len(mine)} of {len(stems)} held-out slides", flush=True)

def find_slide(stem):
    for d in (CONVERTED, TRAIN):
        p=os.path.join(d, stem+".tiff")
        if os.path.exists(p): return p
    return None

ok=fail=skip=0
for stem in mine:
    out=os.path.join(OUTDIR, stem+".json")
    if os.path.exists(out): skip+=1; continue
    sp=find_slide(stem)
    if sp is None:
        print(f"  MISS {stem}: no slide file", flush=True); fail+=1; continue
    try:
        t0=time.time(); cot=predict_cot(sp, CK, gpu=0); dt=time.time()-t0
        json.dump({"id": stem+".tiff", "chain-of-thought": cot}, open(out,"w"), indent=2, ensure_ascii=False)
        ok+=1
        print(f"  OK {stem} {dt:.0f}s organ={cot[0]['answer']} steps={len(cot)}", flush=True)
    except Exception as e:
        fail+=1; print(f"  FAIL {stem} {type(e).__name__}: {e}", flush=True); traceback.print_exc()
print(f"chunk {idx} DONE: ok={ok} skip={skip} fail={fail}", flush=True)
