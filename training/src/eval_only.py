"""Clean, isolated eval-only pass on an already-trained (done=True) full-corpus checkpoint --
reproduces the exact eval logic/split from train_shared_resumable.py / train_pertask_pilot.py,
for when a training log got corrupted (e.g. concurrent writes to the same sbatch output file)
and we need a trustworthy readout of a checkpoint's real metrics.
Usage: python src/eval_only.py <ckpt_path>
"""
import json, sys
from collections import Counter, defaultdict
import random
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
PER_ORGAN = 3000

ckpt_path = sys.argv[1]
device = "cuda" if torch.cuda.is_available() else "cpu"
lm = load_label_maps(LM)
ck = torch.load(ckpt_path, map_location=device, weights_only=False)
per_task = ck.get("per_task_attention", False)
model = REGAttributeModel(lm, in_dim=1536, per_task_attention=per_task).to(device)
model.load_state_dict(ck["model"]); model.eval()
print(f"loaded {ckpt_path} (per_task_attention={per_task}, epoch={ck.get('epoch')}, done={ck.get('done')})", flush=True)

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

attr_cor = Counter(); attr_tot = Counter()
org_cor = Counter(); org_tot = Counter()
edge_by = defaultdict(list); rep_by = defaultdict(list)
with torch.no_grad():
    for i in val:
        f, t, o, sid = ds[i]
        if not t: continue
        f = f.to(device)
        lg = model(f, questions=list(t.keys()))
        pred = {q: lm[q][int(lg[q].argmax())] for q in t}
        for q, y in t.items():
            ok = int(lg[q].argmax()) == y; attr_cor[q] += ok; attr_tot[q] += 1
            org_cor[o] += ok; org_tot[o] += 1
        c = cot.get(sid if sid.endswith('.tiff') else sid+'.tiff')
        if c:
            steps = assemble_cot(o, lambda q: pred.get(q), graph, roots, threshold=0.10)
            edge_by[o].append(e_f1(edge_set(steps), gt_edges(c)))
            rep_by[o].append(tok_f1(compose_report(pred), gt_report(c)))

print(f"\n[Overall attribute accuracy] {sum(attr_cor.values())/sum(attr_tot.values()):.3f}")
print("\n[Per-organ: attribute acc | REAL Edge-F1 | REAL report tokF1]")
ae=[];ee=[];re=[]
for o in sorted(org_tot):
    a = org_cor[o]/org_tot[o]; e = sum(edge_by[o])/len(edge_by[o]) if edge_by[o] else 0; r = sum(rep_by[o])/len(rep_by[o]) if rep_by[o] else 0
    ae.append(a);ee.append(e);re.append(r)
    print(f"  {o:10s}{a:>9.3f}{e:>9.3f}{r:>9.3f}")
print(f"  {'OVERALL':10s}{sum(ae)/len(ae):>9.3f}{sum(ee)/len(ee):>9.3f}{sum(re)/len(re):>9.3f}")
