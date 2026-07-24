"""
Free, read-only check: does reg_attr_v2_full_best.pt (per_task_attention=True, the COMPLETE
30-epoch production retrain, best-by-report_tokf1 = epoch 11) actually beat the deployed
reg_attr_analysis.pt (shared attention) under TODAY's corrected compose_report.py and the real
production threshold (0.25), on the FULL val set?

Same methodology as eval_v2_vs_baseline.py (which checked the forgotten Jun 27 epoch-4 partial
checkpoint) but pointed at the new complete-run best checkpoint. Loads checkpoints read-only;
writes nothing back to any checkpoint or submission directory.
"""
import json, sys
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
PER_ORGAN = 1600  # matches train_v2.py / train_v2_resumable.py's split exactly

lm = load_label_maps(LM)
ds = REGFeatureDataset(FEAT, LAB, lm)
print(f"total paired slides: {len(ds)}", flush=True)

by_org = defaultdict(list)
for i, s in enumerate(ds.samples): by_org[s["organ"]].append(i)
rng = __import__("random").Random(0); idxs = []
for o, ii in by_org.items():
    rng.shuffle(ii); idxs += ii[:PER_ORGAN]
rng.shuffle(idxs)
nval = int(0.18 * len(idxs)); val, train = idxs[:nval], idxs[nval:]
print(f"subset {len(idxs)} (train {len(train)} / FULL val {len(val)}) | {PER_ORGAN}/organ", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"device: {device}", flush=True)

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

def evaluate_full(ckpt_path, label):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    pta = ck.get("per_task_attention", False)
    model = REGAttributeModel(lm, in_dim=1536, per_task_attention=pta).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"\n=== {label}  (per_task_attention={pta}) ===", flush=True)
    attr_cor = Counter(); attr_tot = Counter(); ef = []; rf = []
    with torch.no_grad():
        for n, i in enumerate(val, 1):
            f, t, o, sid = ds[i]
            if not t: continue
            f = f.to(device)
            lg = model(f, questions=list(t.keys()))
            for q, y in t.items():
                attr_cor[q] += int(lg[q].argmax() == y); attr_tot[q] += 1
            c = cot.get(sid if sid.endswith('.tiff') else sid + '.tiff')
            if c:
                lg_all = model(f)
                pred_all = {q: lm[q][int(lg_all[q].argmax())] for q in lm}
                steps = assemble_cot(o, lambda q: pred_all.get(q), graph, roots, threshold=0.25)
                vis = {s["question"] for s in steps} | {s["next_question"] for s in steps}
                vis.discard(""); vis.discard("What is the final pathology report?")
                pa = {q: pred_all[q] for q in vis if q in pred_all}
                ef.append(e_f1(edge_set(steps), gt_edges(c)))
                rf.append(tok_f1(compose_report(pa), gt_report(c)))
            if n % 200 == 0: print(f"  {n}/{len(val)}", flush=True)
    acc = sum(attr_cor.values()) / max(sum(attr_tot.values()), 1)
    edge = sum(ef) / len(ef) if ef else 0
    rep = sum(rf) / len(rf) if rf else 0
    print(f"  FULL-VAL  attr_acc={acc:.4f}  edge_f1={edge:.4f}  report_tokF1={rep:.4f}  (n_report={len(rf)})", flush=True)
    return acc, edge, rep

base_ckpt = "/nfs-stor/adinath.dukre/adinath/reg2026_work/submission_model/reg_attr_analysis.pt"
v2_ckpt = "/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints/reg_attr_v2_full_best.pt"

a_acc, a_edge, a_rep = evaluate_full(base_ckpt, "DEPLOYED BASELINE (shared attention)")
b_acc, b_edge, b_rep = evaluate_full(v2_ckpt, "reg_attr_v2_full_best.pt (per_task_attention, epoch 11 of complete 30-epoch run)")

print("\n=== DELTA (v2-full minus deployed baseline), today's corrected compose_report.py, threshold=0.25 ===")
print(f"  d_attr_acc      = {b_acc - a_acc:+.4f}")
print(f"  d_edge_f1       = {b_edge - a_edge:+.4f}")
print(f"  d_report_tokF1  = {b_rep - a_rep:+.4f}")
