"""
Generate Interface-1 predictions + matching ground-truth for a HELD-OUT set, in the format
the submission_evaluation_code folder expects, so we can score the official Metric A.

Reproduces analyze_performance.py's exact train/val split (same seed) -> uses VAL slides
(never trained on). Writes:
  data/cases/interf1/ground_truth_CoT.json   (GT for the held-out slides)
  data/predictions/interf1/predictions.json  (our model's CoT for the same slides)
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
PER_EVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 5   # held-out slides per organ
THRESHOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25  # traversal edge threshold (0.25 = tuned optimum)

lm=json.load(open(LM))
ds=REGFeatureDataset(FEAT, LAB, lm)
# === reproduce analyze_performance split EXACTLY (same seed/params) to get the VAL (unseen) set ===
by=defaultdict(list)
for i,s in enumerate(ds.samples): by[s["organ"]].append(i)
rng=random.Random(0); idxs=[]
for o,ii in by.items():
    rng.shuffle(ii); idxs+=ii[:1600]
rng.shuffle(idxs)
nval=int(0.18*len(idxs)); val_idx=set(idxs[:nval])
# pick PER_EVAL held-out slides per organ from val
held=defaultdict(list)
for i in idxs[:nval]:
    o=ds.samples[i]["organ"]
    if len(held[o])<PER_EVAL: held[o].append(i)
eval_idx=[i for v in held.values() for i in v]
print(f"held-out eval slides (unseen val): {len(eval_idx)}  ({PER_EVAL}/organ)", flush=True)

model,lmck=load_model(CK,"cpu"); graph,roots=load_graph(); omap=build_organ_answer_to_key()
cot={c['id']:c for c in json.load(open(COT))}

preds=[]; cases=[]
for i in eval_idx:
    s=ds.samples[i]; sid=s["id"]                      # stem, e.g. PIT_01_00019_01
    fid=sid if sid.endswith(".tiff") else sid+".tiff"
    with h5py.File(s["path"]) as h:
        feats=torch.from_numpy(np.asarray(h["features"],dtype=np.float32))
    steps=predict_chain_of_thought_from_features(feats,model,lmck,graph,roots,omap,threshold=THRESHOLD)
    preds.append({"id": fid, "chain-of-thought": steps})
    gt=cot.get(fid)
    if gt: cases.append({"id": fid, "chain-of-thought": gt["chain-of-thought"]})

os.makedirs(f"{EVAL}/predictions/interf1", exist_ok=True)
os.makedirs(f"{EVAL}/cases/interf1", exist_ok=True)
json.dump(preds, open(f"{EVAL}/predictions/interf1/predictions.json","w"), indent=2, ensure_ascii=False)
json.dump(cases, open(f"{EVAL}/cases/interf1/ground_truth_CoT.json","w"), indent=2, ensure_ascii=False)
print(f"wrote {len(preds)} predictions + {len(cases)} GT cases", flush=True)
print("  ->", f"{EVAL}/predictions/interf1/predictions.json")
print("  ->", f"{EVAL}/cases/interf1/ground_truth_CoT.json")
