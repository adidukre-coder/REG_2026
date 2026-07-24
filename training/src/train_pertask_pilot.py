"""
Same recipe as analyze_performance_seeded.py (same pilot features/labels, same PER_ORGAN/
EPOCHS defaults, same fixed random.Random(0) split, same per-seed torch/numpy/random seeding)
but with per_task_attention=True -- each of the 93 questions gets its OWN independent ABMIL
attention pooling (model.py GatedAttention), instead of one attention module shared by all
questions (which is what every existing checkpoint in this project, including the deployed
submission checkpoints, actually uses -- confirmed by inspection, see
reg2026-attention-evidence-grounding memory).

Purpose: test whether per-question attention specialization (a) changes accuracy at all
(free side-check) and (b) produces genuinely distinct, question-appropriate attention
patterns -- the prerequisite for an honest "evidence-grounded CoT" claim (grading questions
attending to different regions than organ-ID questions, etc.), as opposed to the shared-
attention grounding which is identical for every question on a given slide.

Usage: REG_SEED=42 python src/train_pertask_pilot.py [per_organ] [epochs]
"""
import json, os, random, sys, time
from collections import Counter, defaultdict
import numpy as np
import torch
torch.set_num_threads(8)
sys.path.insert(0, "cot_artifacts")
from src.dataset import REGFeatureDataset, compute_class_weights
from src.model import REGAttributeModel, masked_multitask_loss, load_label_maps, default_criticality_weights
from src.compose_report import compose_report
from traversal_engine import load_graph, assemble_cot, edge_set

SEED = int(os.environ.get("REG_SEED", "42"))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

FEAT = os.environ.get("REG_FEAT_DIR", "/nfs-stor/adinath.dukre/adinath/reg2026_work/features_hoptimus0_pilot")
LAB = os.environ.get("REG_LAB_PATH", "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json")
LM = os.environ.get("REG_LM_PATH", "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/label_maps.json")
COT = "/nfs-stor/adinath.dukre/adinath/reg2026/train_CoT_v01.json"
PER_ORGAN = int(sys.argv[1]) if len(sys.argv) > 1 else 100   # match the h0base pilot recipe (not the 500 full-analysis default)
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 8

print(f"SEED={SEED}  per_task_attention=True  PER_ORGAN={PER_ORGAN}  EPOCHS={EPOCHS}", flush=True)

lm = load_label_maps(LM)
ds = REGFeatureDataset(FEAT, LAB, lm)
print(f"total paired slides available: {len(ds)}", flush=True)

by_org = defaultdict(list)
for i, s in enumerate(ds.samples): by_org[s["organ"]].append(i)
rng = random.Random(0); idxs = []
for o, ii in by_org.items():
    rng.shuffle(ii); idxs += ii[:PER_ORGAN]
rng.shuffle(idxs)
nval = int(0.18 * len(idxs)); val, train = idxs[:nval], idxs[nval:]
print(f"analysis subset: {len(idxs)} (train {len(train)} / val {nval}) | {PER_ORGAN}/organ | {EPOCHS} epochs", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"device: {device}", flush=True)
crit = default_criticality_weights(lm)
cw = {q: w.to(device) for q, w in compute_class_weights(LAB, lm, list(lm.keys())).items()}
model = REGAttributeModel(lm, in_dim=1536, per_task_attention=True).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

ck_dir = "/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints"; os.makedirs(ck_dir, exist_ok=True)
out_ckpt = os.environ.get("REG_OUT_CKPT", f"{ck_dir}/reg_attr_pertask_seed{SEED}.pt")

start_ep = 1
if os.path.exists(out_ckpt):
    prev = torch.load(out_ckpt, map_location=device, weights_only=False)
    if prev.get("done"):
        print(f"checkpoint at {out_ckpt} already marked done -- nothing to do.", flush=True)
        sys.exit(0)
    model.load_state_dict(prev["model"])
    opt.load_state_dict(prev["opt"])
    start_ep = prev.get("epoch", 0) + 1
    print(f"RESUMING from {out_ckpt} at epoch {start_ep} (found existing, incomplete checkpoint)", flush=True)

train_rng = random.Random(SEED + 1000 * start_ep)  # vary shuffle order a bit across resumes, harmless

t0 = time.time()
for ep in range(start_ep, EPOCHS + 1):
    model.train(); train_rng.shuffle(train); opt.zero_grad()
    for n, i in enumerate(train, 1):
        f, t, o, s = ds[i]
        if not t: continue
        f = f.to(device)
        lg = model(f, questions=list(t.keys()))
        loss = masked_multitask_loss(lg, t, crit, cw) / 8; loss.backward()
        if n % 8 == 0: opt.step(); opt.zero_grad()
    opt.step()
    print(f"  epoch {ep}/{EPOCHS} done ({time.time()-t0:.0f}s)", flush=True)
    tmp = out_ckpt + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "label_maps": lm,
                "per_task_attention": True, "epoch": ep, "done": False}, tmp)
    os.replace(tmp, out_ckpt)  # atomic -- never leaves a half-written checkpoint on disk
    print(f"  checkpoint saved (epoch {ep}) -> {out_ckpt}", flush=True)

# ===== EVAL (mirrors analyze_performance_seeded.py exactly) =====
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

model.eval()
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

tmp = out_ckpt + ".tmp"
torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "label_maps": lm,
            "per_task_attention": True, "epoch": EPOCHS, "done": True}, tmp)
os.replace(tmp, out_ckpt)
print("\nsaved FINAL checkpoint -> " + out_ckpt)
