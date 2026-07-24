"""
Full-corpus novelty-strengthening analysis for the per-task-attention contribution.
Inference-only (loads already-trained checkpoints, no gradient steps) against the SAME
val split used to train/eval reg_attr_{pertask,shared}_full_seed{S}.pt (organ-balanced
PER_ORGAN=3000 pool, random.Random(0) split, nval=0.18) -- so results are directly
comparable to the numbers already reported for those checkpoints.

Produces three things for the paper:
  1. Per-question-category (gate / grader / descriptive) accuracy: pertask vs shared,
     to show WHERE the specialization gain concentrates.
  2. Specialization Index: mean pairwise Jensen-Shannon divergence between different
     questions' attention distributions on the same slide. ~0 by construction for shared
     (sanity check), meaningfully >0 for pertask (quantifies "genuinely different attention"
     instead of relying on a visual anecdote).
  3. Dual-uncertainty check: correlation between softmax margin and attention entropy,
     plus the high-margin/high-entropy error-rate breakdown from attn_entropy_probe.py,
     rerun here at full-corpus scale (2010 val slides vs. the pilot's ~75).

Usage: python src/novelty_analysis_full.py
"""
import json, math, os, random, sys
from collections import defaultdict
import numpy as np
import torch
sys.path.insert(0, "cot_artifacts")
from src.dataset import REGFeatureDataset
from src.model import REGAttributeModel, load_label_maps

FEAT = "/nfs-stor/adinath.dukre/adinath/reg2026_work/features/20x_256px_0px_overlap/features_hoptimus0"
LAB = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json"
LM = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/label_maps.json"
CKDIR = "/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints"
PER_ORGAN = 3000

CHECKPOINTS = {
    "pertask_42": (f"{CKDIR}/reg_attr_pertask_full_seed42.pt", True),
    "pertask_7": (f"{CKDIR}/reg_attr_pertask_full_seed7.pt", True),
    "pertask_123": (f"{CKDIR}/reg_attr_pertask_full_seed123.pt", True),
    "shared_42": (f"{CKDIR}/reg_attr_shared_full_seed42.pt", False),
    "shared_7": (f"{CKDIR}/reg_attr_shared_full_seed7.pt", False),
    "shared_123": (f"{CKDIR}/reg_attr_shared_full_seed123.pt", False),
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}", flush=True)

lm = load_label_maps(LM)


def categorize(q):
    if q.strip().lower().startswith("is there any") or q.strip().lower().startswith("is any"):
        return "gate"
    keys = ["grade", "score", "gleason", "differentiation", "pleomorphism", "mitotic",
            "pattern", "extent of invasion", "grading system"]
    ql = q.lower()
    if any(k in ql for k in keys):
        return "grader"
    return "descriptive"


CATS = {q: categorize(q) for q in lm}
print("category counts:", {c: sum(1 for v in CATS.values() if v == c) for c in ("gate", "grader", "descriptive")}, flush=True)

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


def js_div_matrix(A):
    # A: [Q, N] each row a prob dist over N patches. Returns [Q,Q] pairwise JS divergence (nats).
    A = A.clamp_min(1e-12)
    P = A.unsqueeze(1)          # [Q,1,N]
    Qm = A.unsqueeze(0)         # [1,Q,N]
    M = 0.5 * (P + Qm)          # [Q,Q,N]
    kl_pm = (P * (P.log() - M.log())).sum(-1)   # [Q,Q]
    kl_qm = (Qm * (Qm.log() - M.log())).sum(-1) # [Q,Q]
    return 0.5 * (kl_pm + kl_qm)


results = {}
for name, (ckpt_path, per_task) in CHECKPOINTS.items():
    if not os.path.exists(ckpt_path):
        print(f"[{name}] checkpoint not found yet, skipping ({ckpt_path})", flush=True)
        continue
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not ck.get("done"):
        print(f"[{name}] checkpoint not marked done yet, skipping", flush=True)
        continue
    model = REGAttributeModel(lm, in_dim=1536, per_task_attention=per_task).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    cat_cor = defaultdict(int)
    cat_tot = defaultdict(int)
    margins, ents, corrects = [], [], []
    spec_idx_per_slide = []

    with torch.no_grad():
        for n, i in enumerate(val, 1):
            f, t, o, sid = ds[i]
            if not t:
                continue
            f = f.to(device)
            qs = list(t.keys())
            logits, attns = model(f, questions=qs, return_attn=True)

            attn_rows = []
            for q in qs:
                lg = logits[q]
                probs = torch.softmax(lg, dim=-1)
                k = min(2, probs.numel())
                top = torch.topk(probs, k=k)
                margin = float(top.values[0] - (top.values[1] if k > 1 else 0.0))
                pred_ok = int(lg.argmax()) == t[q]
                cat = CATS.get(q, "descriptive")
                cat_tot[cat] += 1
                cat_cor[cat] += int(pred_ok)

                a = attns[q]
                Np = a.numel()
                a_np = a.detach()
                ent = float(-(a_np * a_np.clamp_min(1e-12).log()).sum())
                ent_norm = ent / math.log(Np) if Np > 1 else 0.0
                margins.append(margin); ents.append(ent_norm); corrects.append(pred_ok)
                attn_rows.append(a_np)

            if len(attn_rows) >= 2:
                A = torch.stack(attn_rows, dim=0)  # [Q,N]
                jsm = js_div_matrix(A)
                Qn = jsm.shape[0]
                iu = torch.triu_indices(Qn, Qn, offset=1)
                mean_js = float(jsm[iu[0], iu[1]].mean()) if iu.shape[1] > 0 else 0.0
                spec_idx_per_slide.append(mean_js)

            if n % 500 == 0:
                print(f"  [{name}] {n}/{len(val)} slides processed", flush=True)

    acc_by_cat = {c: (cat_cor[c] / cat_tot[c] if cat_tot[c] else None) for c in ("gate", "grader", "descriptive")}
    spec_idx = sum(spec_idx_per_slide) / len(spec_idx_per_slide) if spec_idx_per_slide else 0.0

    margins_np = np.array(margins); ents_np = np.array(ents); corr_np = np.array(corrects)
    if margins_np.std() > 1e-9 and ents_np.std() > 1e-9:
        pearson = float(np.corrcoef(margins_np, ents_np)[0, 1])
    else:
        pearson = 0.0
    high_margin_mask = margins_np > 0.5
    hm_ents = ents_np[high_margin_mask]; hm_corr = corr_np[high_margin_mask]
    base_rate = 1 - hm_corr.mean() if len(hm_corr) else 0.0
    topk_report = {}
    if len(hm_ents) > 0:
        order = np.argsort(-hm_ents)
        for frac in (0.05, 0.10, 0.20):
            k = max(1, int(frac * len(order)))
            topk = order[:k]
            wrong_rate = 1 - hm_corr[topk].mean()
            topk_report[frac] = wrong_rate

    results[name] = dict(acc_by_cat=acc_by_cat, spec_idx=spec_idx, pearson_margin_entropy=pearson,
                          base_wrong_rate_high_margin=float(base_rate), topk_wrong_rate=topk_report,
                          n_high_margin=int(high_margin_mask.sum()))

    print(f"\n===== {name} (per_task_attention={per_task}) =====")
    print(f"  accuracy by category: {acc_by_cat}")
    print(f"  specialization index (mean pairwise JS div across questions, same slide): {spec_idx:.5f}")
    print(f"  corr(margin, attn_entropy): {pearson:.4f}")
    print(f"  high-margin(>0.5) base wrong-rate: {base_rate:.3%}  (n={int(high_margin_mask.sum())})")
    for frac, wr in topk_report.items():
        print(f"    top {frac:.0%} highest-entropy among high-margin: wrong-rate={wr:.3%}")

print("\n\n================ SUMMARY (all checkpoints) ================")
for name, r in results.items():
    print(name, json.dumps(r, indent=None))

out_path = "/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints/novelty_analysis_full_results.json"
with open(out_path, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nsaved -> {out_path}")
