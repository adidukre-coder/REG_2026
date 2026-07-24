"""
DIAGNOSTIC (no submission, no format change): measure the effect of the patch-count CAP on
Metric-A, using the SAME 140 held-out val slides as the 0.850 baseline and the SAME cached
hest features. Isolates the patch-cap variable (full vs 2000 vs 1000) — directly tests our
top SAFE Test-Phase candidate (bump MAX_PATCHES 1000 -> 2000) WITHOUT re-extracting features.

Reproduces analyze_performance.py's split EXACTLY (seed 0, 1600/organ, 0.18 val), takes the
SAME PER_EVAL/organ held-out slides, applies the cap the SAME way as the container's
_cap_coords (np.linspace uniform subsample), runs the SAME model + engine + composer (thr 0.25).

Usage: python src/diag_capsweep.py <PER_EVAL=20> <CAP: int or 'full'>
Writes:
  submission_evaluation_code/data/predictions/interf1/predictions_hest_cap<CAP>.json
  submission_evaluation_code/data/cases/interf1/ground_truth_CoT.json   (shared across caps)
"""
import json, os, random, sys
from collections import defaultdict
import h5py, numpy as np, torch
sys.path.insert(0, "cot_artifacts")
from src.dataset import REGFeatureDataset
from src.model import REGAttributeModel  # noqa (used via interf1_infer)
from src.interf1_infer import load_model, build_organ_answer_to_key, predict_chain_of_thought_from_features
from traversal_engine import load_graph

FEAT=os.environ.get("REG_FEAT_DIR", "/nfs-stor/adinath.dukre/adinath/reg2026_work/features/20x_256px_0px_overlap/features_hoptimus0")
LAB ="/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json"
LM  ="/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/label_maps.json"
COT ="/nfs-stor/adinath.dukre/adinath/reg2026/train_CoT_v01.json"
CK  =os.environ.get("REG_CK", "/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints/reg_attr_analysis.pt")
EVAL="submission_evaluation_code/data"
PER_EVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 20
CAP_ARG  = sys.argv[2] if len(sys.argv) > 2 else "full"
CAP = None if CAP_ARG == "full" else int(CAP_ARG)
THRESHOLD = 0.25  # tuned optimum, same as submission

def cap_feats(feats: torch.Tensor, cap):
    """Uniform subsample to `cap` rows — IDENTICAL policy to interf1_pipeline._cap_coords."""
    n = feats.shape[0]
    if cap is None or n <= cap:
        return feats
    idx = np.linspace(0, n - 1, cap).astype(np.int64)
    return feats[idx]

lm=json.load(open(LM))
ds=REGFeatureDataset(FEAT, LAB, lm)
# === reproduce analyze_performance split EXACTLY (same seed/params) to get the VAL (unseen) set ===
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
print(f"[cap={CAP_ARG}] held-out eval slides (unseen val): {len(eval_idx)}  ({PER_EVAL}/organ)", flush=True)

model,lmck=load_model(CK,"cpu"); graph,roots=load_graph(); omap=build_organ_answer_to_key()
cot={c['id']:c for c in json.load(open(COT))}

preds=[]; cases=[]; npatch=[]
for i in eval_idx:
    s=ds.samples[i]; sid=s["id"]
    fid=sid if sid.endswith(".tiff") else sid+".tiff"
    with h5py.File(s["path"]) as h:
        feats=torch.from_numpy(np.asarray(h["features"],dtype=np.float32))
    feats=cap_feats(feats, CAP); npatch.append(feats.shape[0])
    steps=predict_chain_of_thought_from_features(feats,model,lmck,graph,roots,omap,threshold=THRESHOLD)
    preds.append({"id": fid, "chain-of-thought": steps})
    gt=cot.get(fid)
    if gt: cases.append({"id": fid, "chain-of-thought": gt["chain-of-thought"]})

os.makedirs(f"{EVAL}/predictions/interf1", exist_ok=True)
os.makedirs(f"{EVAL}/cases/interf1", exist_ok=True)
pred_path=f"{EVAL}/predictions/interf1/predictions_hest_cap{CAP_ARG}.json"
json.dump(preds, open(pred_path,"w"), indent=2, ensure_ascii=False)
json.dump(cases, open(f"{EVAL}/cases/interf1/ground_truth_CoT.json","w"), indent=2, ensure_ascii=False)
print(f"[cap={CAP_ARG}] wrote {len(preds)} preds + {len(cases)} GT cases; patches/slide min/med/max="
      f"{min(npatch)}/{int(np.median(npatch))}/{max(npatch)}", flush=True)
print("  ->", pred_path)
