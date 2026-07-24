"""
Resumable version of train_v2.py: the actual production retrain (per_task_attention=True,
criticality weights upweighting report-driving diagnosis/grader heads, PER_ORGAN=1600,
EPOCHS=30, same split as the deployed baseline) that produced the forgotten Jun 27
reg_attr_v2_best.pt (which stopped at epoch 4 with no way to tell if it was cut short).

Same recipe as train_v2.py, unchanged, plus per-epoch atomic checkpoint+resume (.tmp +
os.replace, "done" flag, sys.exit(0) if already done) so it survives the 3h gpu-debug-qos
cap via --dependency=afterany chaining, same pattern used for the full-corpus per_task_attention
and diversity-regularization runs this session. Saves TWO files:
  REG_OUT_CKPT   -- full resume state (model+opt+rng), overwritten every epoch, not the final artifact
  REG_BEST_CKPT  -- best-by-val_report_tokf1 model-only checkpoint, the actual deployable artifact
Baseline reg_attr_analysis.pt is never touched.
"""
import json, random, sys, time, os
from collections import Counter, defaultdict
import numpy as np
import torch
torch.set_num_threads(8)
sys.path.insert(0, "cot_artifacts")
from src.dataset import REGFeatureDataset, compute_class_weights
from src.model import REGAttributeModel, masked_multitask_loss, load_label_maps
from src.compose_report import compose_report
from traversal_engine import load_graph, assemble_cot, edge_set

SEED = int(os.environ.get("REG_SEED", "42"))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

FEAT = "/nfs-stor/adinath.dukre/adinath/reg2026_work/features/20x_256px_0px_overlap/features_hoptimus0"
LAB = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json"
LM = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/label_maps.json"
COT = "/nfs-stor/adinath.dukre/adinath/reg2026/train_CoT_v01.json"
CKDIR = "/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints"
PER_ORGAN = int(os.environ.get("REG_PER_ORGAN", "1600"))
EPOCHS = int(os.environ.get("REG_EPOCHS", "30"))
OUT_CKPT = os.environ.get("REG_OUT_CKPT", CKDIR + "/reg_attr_v2_full_resumable.pt")
BEST = os.environ.get("REG_BEST_CKPT", CKDIR + "/reg_attr_v2_full_best.pt")

lm = load_label_maps(LM)
ds = REGFeatureDataset(FEAT, LAB, lm)
print(f"total paired slides: {len(ds)}", flush=True)

# SAME split as analyze_performance.py / train_v2.py (seed 0) -> identical val/held-out
by_org = defaultdict(list)
for i, s in enumerate(ds.samples): by_org[s["organ"]].append(i)
split_rng = random.Random(0); idxs = []
for o, ii in by_org.items():
    split_rng.shuffle(ii); idxs += ii[:PER_ORGAN]
split_rng.shuffle(idxs)
nval = int(0.18 * len(idxs)); val, train = idxs[:nval], idxs[nval:]
_vb = defaultdict(list)
for i in val: _vb[ds.samples[i]["organ"]].append(i)
val_eval = [i for v in _vb.values() for i in v[:50]]
print(f"subset {len(idxs)} (train {len(train)} / val {nval}, per-epoch eval {len(val_eval)}) | {PER_ORGAN}/organ | {EPOCHS} epochs | per_task_attention=True", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"device: {device}", flush=True)

crit = {q: 1.0 for q in lm}
crit.update({
    "Is there any abnormality present?": 6.0,
    "Is there any neoplasm present?": 5.0,
    "What is the histologic type of neoplasm?": 3.5,
    "What is the #1 diagnosis?": 4.0,
    "What is the #2 diagnosis?": 2.0,
    "What is the histologic subtype of neoplasm?": 2.5,
    "What is the histologic type of lesion?": 2.5,
    "What is the primary of neoplasm?": 1.5,
    "What is the Gleason score?": 3.0,
    "What is the grade group?": 3.0,
    "What is the percentage of Gleason pattern 4?": 2.0,
    "What is the tumor volume?": 2.0,
    "What is the grade of neoplasm?": 2.5,
    "What is the score for nuclear pleomorphism?": 2.5,
    "What is the score for mitotic rate?": 2.5,
    "What is the score for tubular differentiation?": 2.5,
    "What is the nuclear grade of lesion?": 2.0,
    "Is there any invasion present?": 2.0,
})
cw = {q: w.to(device) for q, w in compute_class_weights(LAB, lm, list(lm.keys())).items()}
model = REGAttributeModel(lm, in_dim=1536, per_task_attention=True).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

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

def evaluate():
    model.eval()
    attr_cor = Counter(); attr_tot = Counter(); ef = []; rf = []
    with torch.no_grad():
        for i in val_eval:
            f, t, o, sid = ds[i]
            if not t: continue
            f = f.to(device)
            lg = model(f, questions=list(t.keys()))
            for q, y in t.items():
                attr_cor[q] += int(lg[q].argmax() == y); attr_tot[q] += 1
            c = cot.get(sid if sid.endswith('.tiff') else sid + '.tiff')
            if c:
                lg_all = model(f); pred_all = {q: lm[q][int(lg_all[q].argmax())] for q in lm}
                steps = assemble_cot(o, lambda q: pred_all.get(q), graph, roots, threshold=0.25)
                vis = {s["question"] for s in steps} | {s["next_question"] for s in steps}
                vis.discard(""); vis.discard("What is the final pathology report?")
                pa = {q: pred_all[q] for q in vis if q in pred_all}
                ef.append(e_f1(edge_set(steps), gt_edges(c)))
                rf.append(tok_f1(compose_report(pa), gt_report(c)))
    acc = sum(attr_cor.values()) / max(sum(attr_tot.values()), 1)
    return acc, (sum(ef)/len(ef) if ef else 0), (sum(rf)/len(rf) if rf else 0), attr_cor, attr_tot

os.makedirs(CKDIR, exist_ok=True)
train_rng = random.Random(SEED + 1000)
start_ep = 1
best_rf = -1.0
if os.path.exists(OUT_CKPT):
    prev = torch.load(OUT_CKPT, map_location=device, weights_only=False)
    if prev.get("done"):
        print(f"checkpoint at {OUT_CKPT} already marked done -- nothing to do.", flush=True)
        sys.exit(0)
    model.load_state_dict(prev["model"])
    opt.load_state_dict(prev["opt"])
    train_rng.setstate(prev["train_rng_state"])
    start_ep = prev.get("epoch", 0) + 1
    best_rf = prev.get("best_rf", -1.0)
    print(f"RESUMING from {OUT_CKPT} at epoch {start_ep}, best_rf so far={best_rf:.4f}", flush=True)

t0 = time.time()
for ep in range(start_ep, EPOCHS + 1):
    model.train(); train_rng.shuffle(train); opt.zero_grad()
    for n, i in enumerate(train, 1):
        f, t, o, s = ds[i]
        if not t: continue
        f = f.to(device)
        lg = model(f, questions=list(t.keys()))
        loss = masked_multitask_loss(lg, t, crit, cw) / 8
        loss.backward()
        if n % 8 == 0: opt.step(); opt.zero_grad()
    opt.step()

    acc, ef, rf, ac, at = evaluate()
    flag = ""
    if rf > best_rf:
        best_rf = rf
        tmp = BEST + ".tmp"
        torch.save({"model": model.state_dict(), "label_maps": lm, "per_task_attention": True,
                    "epoch": ep, "val_report_tokf1": rf, "val_edge_f1": ef, "val_attr_acc": acc}, tmp)
        os.replace(tmp, BEST)
        flag = " *BEST -> saved"
    ht = ac["What is the histologic type of neoplasm?"] / max(at["What is the histologic type of neoplasm?"], 1)
    print(f"  ep {ep:2d}/{EPOCHS} ({time.time()-t0:.0f}s)  attr_acc={acc:.3f} edge_f1={ef:.3f} report_tokf1={rf:.3f} histol_dx={ht:.3f}{flag}", flush=True)

    tmp = OUT_CKPT + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "label_maps": lm,
                "per_task_attention": True, "epoch": ep, "best_rf": best_rf,
                "train_rng_state": train_rng.getstate(), "done": False}, tmp)
    os.replace(tmp, OUT_CKPT)
    print(f"  resume-checkpoint saved (epoch {ep}) -> {OUT_CKPT}", flush=True)

tmp = OUT_CKPT + ".tmp"
torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "label_maps": lm,
            "per_task_attention": True, "epoch": EPOCHS, "best_rf": best_rf,
            "train_rng_state": train_rng.getstate(), "done": True}, tmp)
os.replace(tmp, OUT_CKPT)
print(f"\nDONE. best_rf={best_rf:.4f} -> {BEST}", flush=True)
print("(baseline reg_attr_analysis.pt untouched; run eval_v2_vs_baseline.py-style full-val check before adopting)", flush=True)
