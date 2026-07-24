"""
Re-sweep the entropy-gated traversal K constant, but anchored at the REAL production
threshold (0.25, from algorithm_submission_template/src/interf1_pipeline.py) instead of the
0.10 baseline the original research sweep (entropy_gated_eval.py) was tuned against, and using
the new production-recipe checkpoint (reg_attr_v2_full_best.pt, per_task_attention=True, best
epoch=11 of the complete 30-epoch retrain) instead of the earlier research-only seed checkpoints.

local_threshold(q) = min(MAX_THRESH, BASE + K * entropy_norm[q]), BASE=0.25.
K=0 exactly reproduces the fixed threshold=0.25 baseline used in eval_v2full_vs_baseline.py
(sanity check -- should match that script's edge_f1/report_tokF1 numbers).

Free, read-only: one cached forward pass over the full val set, then a cheap CPU sweep across K.
Loads checkpoints read-only; writes nothing back to any checkpoint or submission directory.
"""
import json, math, random, sys
from collections import Counter, defaultdict
import torch
torch.set_num_threads(8)
sys.path.insert(0, "cot_artifacts")
from src.dataset import REGFeatureDataset
from src.model import REGAttributeModel, load_label_maps
from src.compose_report import compose_report
from traversal_engine import load_graph, assemble_cot, edge_set

FEAT = "/nfs-stor/adinath.dukre/adinath/reg2026_work/features/20x_256px_0px_overlap/features_hoptimus0"
LAB = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json"
LM = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/label_maps.json"
COT = "/nfs-stor/adinath.dukre/adinath/reg2026/train_CoT_v01.json"
CKPT = "/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints/reg_attr_v2_full_best.pt"
PER_ORGAN = 1600  # matches the production split exactly (train_v2_resumable.py / eval_v2full_vs_baseline.py)
BASE, MAX_THRESH = 0.25, 0.60

KS = [float(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

device = "cuda" if torch.cuda.is_available() else "cpu"
lm = load_label_maps(LM)
ck = torch.load(CKPT, map_location=device, weights_only=False)
model = REGAttributeModel(lm, in_dim=1536, per_task_attention=True).to(device)
model.load_state_dict(ck["model"]); model.eval()
print(f"loaded {CKPT}  (epoch={ck.get('epoch')}, val_report_tokf1={ck.get('val_report_tokf1')})", flush=True)

ds = REGFeatureDataset(FEAT, LAB, lm)
by_org = defaultdict(list)
for i, s in enumerate(ds.samples):
    by_org[s["organ"]].append(i)
rng = random.Random(0)
idxs = []
for o, ii in by_org.items():
    rng.shuffle(ii)
    idxs += ii[:PER_ORGAN]
rng.shuffle(idxs)
nval = int(0.18 * len(idxs))
val = idxs[:nval]
print(f"FULL val slides: {len(val)}", flush=True)

cot = {c['id']: c for c in json.load(open(COT))}
graph, roots = load_graph()
def gt_edges(c): return {(s['question'], s['next_question']) for s in c['chain-of-thought'] if s['question']}
def gt_report(c):
    for s in c['chain-of-thought']:
        if s['question'] == "What is the final pathology report?": return s['answer']
def tok_f1(a, b):
    A, B = (a or "").split(), (b or "").split()
    if not A and not B: return 1.0
    i = sum((Counter(A) & Counter(B)).values()); p = i/len(A) if A else 0; r = i/len(B) if B else 0
    return 2*p*r/(p+r) if p+r else 0.0
def e_f1(P, T):
    if not P and not T: return 1.0
    tp = len(P & T); pr = tp/len(P) if P else 0; rc = tp/len(T) if T else 0
    return 2*pr*rc/(pr+rc) if pr+rc else 0.0

cache = []
with torch.no_grad():
    for n, i in enumerate(val, 1):
        f, t, o, sid = ds[i]
        if not t:
            continue
        f = f.to(device)
        qs = list(lm.keys())
        logits, attns = model(f, questions=qs, return_attn=True)
        pred = {q: lm[q][int(logits[q].argmax())] for q in qs}
        ent = {}
        for q in qs:
            a = attns[q]
            Np = a.numel()
            e = float(-(a * a.clamp_min(1e-12).log()).sum())
            ent[q] = e / math.log(Np) if Np > 1 else 0.0
        c = cot.get(sid if sid.endswith('.tiff') else sid + '.tiff')
        cache.append((o, pred, ent, c))
        if n % 200 == 0:
            print(f"  forward pass {n}/{len(val)}", flush=True)

print(f"\nBASE={BASE}  MAX_THRESH={MAX_THRESH}  n_report_cases={sum(1 for _,_,_,c in cache if c)}\n", flush=True)
for K in KS:
    ef = []; rf = []
    for o, pred, ent, c in cache:
        if not c:
            continue
        thr_fn = (lambda q, ent=ent, K=K: min(MAX_THRESH, BASE + K * ent.get(q, 0.0)))
        steps = assemble_cot(o, lambda q: pred.get(q), graph, roots, threshold=thr_fn)
        vis = {s["question"] for s in steps} | {s["next_question"] for s in steps}
        vis.discard(""); vis.discard("What is the final pathology report?")
        pa = {q: pred[q] for q in vis if q in pred}
        ef.append(e_f1(edge_set(steps), gt_edges(c)))
        rf.append(tok_f1(compose_report(pa), gt_report(c)))
    edge = sum(ef) / len(ef) if ef else 0
    rep = sum(rf) / len(rf) if rf else 0
    print(f"K={K:<5}  edge_f1={edge:.4f}  report_tokF1={rep:.4f}")
