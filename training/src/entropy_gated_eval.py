"""
Entropy-gated decision-graph traversal: sweep an entropy-weight K to see whether raising the
local graph-branching threshold for questions the model's per-task attention was diffuse
about (high attn entropy) improves edge_f1 / report tokF1 vs. the fixed threshold=0.10
baseline used everywhere else in this project.

Per question q on a slide: local_threshold(q) = min(MAX_THRESH, BASE + K * entropy_norm[q])
K=0 exactly reproduces the existing fixed-threshold baseline (sanity check).

Usage: python src/entropy_gated_eval.py <ckpt_name e.g. seed42> [K1 K2 ...]
"""
import json, math, random, sys
from collections import Counter, defaultdict
import torch
sys.path.insert(0, "cot_artifacts")
from src.dataset import REGFeatureDataset
from src.model import REGAttributeModel, load_label_maps
from src.compose_report import compose_report
from traversal_engine import load_graph, assemble_cot, edge_set

FEAT = "/nfs-stor/adinath.dukre/adinath/reg2026_work/features/20x_256px_0px_overlap/features_hoptimus0"
LAB = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json"
LM = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/label_maps.json"
COT = "/nfs-stor/adinath.dukre/adinath/reg2026/train_CoT_v01.json"
CKDIR = "/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints"
PER_ORGAN = 3000
BASE, MAX_THRESH = 0.10, 0.60

CKNAME = sys.argv[1] if len(sys.argv) > 1 else "seed42"
KS = [float(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]

device = "cuda" if torch.cuda.is_available() else "cpu"
lm = load_label_maps(LM)
ckpt_path = f"{CKDIR}/reg_attr_pertask_full_{CKNAME}.pt"
ck = torch.load(ckpt_path, map_location=device, weights_only=False)
model = REGAttributeModel(lm, in_dim=1536, per_task_attention=True).to(device)
model.load_state_dict(ck["model"]); model.eval()
print(f"loaded {ckpt_path}", flush=True)

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
print(f"val slides: {len(val)}", flush=True)

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

# precompute, per val slide: pred dict + entropy_norm dict (one forward pass reused for all K)
cache = []
with torch.no_grad():
    for n, i in enumerate(val, 1):
        f, t, o, sid = ds[i]
        if not t:
            continue
        f = f.to(device)
        qs = list(t.keys())
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
        if n % 500 == 0:
            print(f"  forward pass {n}/{len(val)}", flush=True)

for K in KS:
    edge_by = defaultdict(list); rep_by = defaultdict(list)
    for o, pred, ent, c in cache:
        if not c:
            continue
        thr_fn = (lambda q, ent=ent, K=K: min(MAX_THRESH, BASE + K * ent.get(q, 0.0)))
        steps = assemble_cot(o, lambda q: pred.get(q), graph, roots, threshold=thr_fn)
        edge_by[o].append(e_f1(edge_set(steps), gt_edges(c)))
        rep_by[o].append(tok_f1(compose_report(pred), gt_report(c)))
    ee = [sum(v)/len(v) for v in edge_by.values() if v]
    re = [sum(v)/len(v) for v in rep_by.values() if v]
    print(f"K={K:<5}  edge_f1={sum(ee)/len(ee):.4f}  report_tokF1={sum(re)/len(re):.4f}")
