"""
Evaluate confidence-weighted decision-path decoding (src/beam_decode.py) against the
existing greedy traversal, reusing the ALREADY-TRAINED seeded checkpoints -- no
retraining, no new feature extraction, pure inference-time comparison. Same val-split
convention as analyze_performance_seeded.py (PER_ORGAN slides/organ, fixed
random.Random(0) split -- independent of which seed's checkpoint is loaded).

Usage: python src/eval_decode_ab.py <per_organ> <ckpt_path> [feat_dir]
"""
import json, os, random, sys
from collections import Counter, defaultdict
import torch
sys.path.insert(0, "cot_artifacts")
from src.dataset import REGFeatureDataset
from src.model import REGAttributeModel, load_label_maps
from src.compose_report import compose_report
from src.beam_decode import assemble_cot_confidence
from traversal_engine import load_graph, assemble_cot, edge_set

PER_ORGAN = int(sys.argv[1]) if len(sys.argv) > 1 else 100
CKPT = sys.argv[2]
FEAT = sys.argv[3] if len(sys.argv) > 3 else "/nfs-stor/adinath.dukre/adinath/reg2026_work/features_hoptimus0_pilot"
LAB = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json"
LM = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/label_maps.json"
COT = "/nfs-stor/adinath.dukre/adinath/reg2026/train_CoT_v01.json"

lm = load_label_maps(LM)
ds = REGFeatureDataset(FEAT, LAB, lm)
by_org = defaultdict(list)
for i, s in enumerate(ds.samples): by_org[s["organ"]].append(i)
rng = random.Random(0); idxs = []
for o, ii in by_org.items():
    rng.shuffle(ii); idxs += ii[:PER_ORGAN]
rng.shuffle(idxs)
nval = int(0.18 * len(idxs)); val, train = idxs[:nval], idxs[nval:]
print(f"CKPT={CKPT}  FEAT={FEAT}  val={len(val)}", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(CKPT, map_location=device, weights_only=False)
model = REGAttributeModel(lm, in_dim=1536, per_task_attention=ck.get("per_task_attention", False)).to(device)
model.load_state_dict(ck["model"]); model.eval()

cot = {c['id']: c for c in json.load(open(COT))}
graph, roots = load_graph()


def gt_edges(c): return {(s['question'], s['next_question']) for s in c['chain-of-thought'] if s['question']}


def gt_report(c):
    for s in c['chain-of-thought']:
        if s['question'] == "What is the final pathology report?": return s['answer']


def tok_f1(a, b):
    A, B = (a or "").split(), (b or "").split()
    if not A and not B: return 1.0
    i = sum((Counter(A) & Counter(B)).values()); p = i / len(A) if A else 0; r = i / len(B) if B else 0
    return 2 * p * r / (p + r) if p + r else 0.0


def e_f1(P, T):
    if not P and not T: return 1.0
    tp = len(P & T); pr = tp / len(P) if P else 0; rc = tp / len(T) if T else 0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


edge_g = defaultdict(list); rep_g = defaultdict(list)
edge_c = defaultdict(list); rep_c = defaultdict(list)
flips = 0; total = 0

with torch.no_grad():
    for i in val:
        f, t, o, sid = ds[i]
        if not t: continue
        f = f.to(device)
        lg = model(f, questions=list(t.keys()))
        pred_greedy = {q: lm[q][int(lg[q].argmax())] for q in t}
        c = cot.get(sid if sid.endswith('.tiff') else sid + '.tiff')
        if not c: continue
        steps_g = assemble_cot(o, lambda q: pred_greedy.get(q), graph, roots, threshold=0.10)
        steps_c, pred_conf = assemble_cot_confidence(o, lg, lm, graph, roots, threshold=0.10,
                                                      exclude_gates=os.environ.get("REG_EXCLUDE_GATES") == "1")
        total += 1
        if pred_conf != pred_greedy: flips += 1
        edge_g[o].append(e_f1(edge_set(steps_g), gt_edges(c)))
        rep_g[o].append(tok_f1(compose_report(pred_greedy), gt_report(c)))
        edge_c[o].append(e_f1(edge_set(steps_c), gt_edges(c)))
        rep_c[o].append(tok_f1(compose_report(pred_conf), gt_report(c)))

print(f"val slides scored={total}  slides w/ >=1 answer changed by decode={flips}")
print(f"{'organ':10s}{'edge_g':>8}{'edge_c':>8}{'d_edge':>8}{'rep_g':>8}{'rep_c':>8}{'d_rep':>8}")
Ae = []; Ce = []; Ar = []; Cr = []
for o in sorted(edge_g):
    eg = sum(edge_g[o]) / len(edge_g[o]); ec = sum(edge_c[o]) / len(edge_c[o])
    rg = sum(rep_g[o]) / len(rep_g[o]); rc = sum(rep_c[o]) / len(rep_c[o])
    Ae.append(eg); Ce.append(ec); Ar.append(rg); Cr.append(rc)
    print(f"{o:10s}{eg:>8.3f}{ec:>8.3f}{ec - eg:>+8.3f}{rg:>8.3f}{rc:>8.3f}{rc - rg:>+8.3f}")
oa_eg = sum(Ae) / len(Ae); oa_ec = sum(Ce) / len(Ce); oa_rg = sum(Ar) / len(Ar); oa_rc = sum(Cr) / len(Cr)
print(f"{'OVERALL':10s}{oa_eg:>8.3f}{oa_ec:>8.3f}{oa_ec - oa_eg:>+8.3f}{oa_rg:>8.3f}{oa_rc:>8.3f}{oa_rc - oa_rg:>+8.3f}")
