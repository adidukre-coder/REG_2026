"""
Probe: does per-question ABMIL attention entropy carry error signal that softmax
margin does NOT already carry?

Motivation: the confidence-weighted decoding experiment (beam_decode.py) tried to use
softmax top1-top2 margin to flag "uncertain" answers for reconsideration, and it failed
(reg2026-confidence-decode-negative.md) -- among other reasons, because margin and the
scoring objective are both downstream of the SAME softmax. Attention entropy (how spread
out the ABMIL attention is over patches, model.py GatedAttention, already computed for
free via forward(..., return_attn=True), never previously used) is a structurally
DIFFERENT signal: it reflects whether the model found a coherent region of evidence,
independent of how peaked the final classifier softmax happens to be.

Test: for predictions where softmax margin is HIGH (model "confident"), does attention
entropy still separate correct from incorrect? If yes -> real, additional signal, worth
building the evidence-grounding + self-audit feature on top of. If no -> entropy is just
recapitulating margin, not worth pursuing further.

Free to run: reuses already-trained checkpoints (reg_attr_h0base_seed{42,7,123}.pt) and
already-extracted pilot features (features_hoptimus0_pilot) -- zero GPU-hours, inference
only, few seconds per seed.
"""
import math, random, sys
from collections import defaultdict
import torch
sys.path.insert(0, "cot_artifacts")
from src.dataset import REGFeatureDataset
from src.model import REGAttributeModel, load_label_maps

FEAT = "/nfs-stor/adinath.dukre/adinath/reg2026_work/features_hoptimus0_pilot"
LAB = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json"
LM = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/label_maps.json"
PER_ORGAN = 100
SEEDS = [42, 7, 123]

lm = load_label_maps(LM)
ds = REGFeatureDataset(FEAT, LAB, lm)
by_org = defaultdict(list)
for i, s in enumerate(ds.samples): by_org[s["organ"]].append(i)
rng = random.Random(0); idxs = []
for o, ii in by_org.items():
    rng.shuffle(ii); idxs += ii[:PER_ORGAN]
rng.shuffle(idxs)
nval = int(0.18 * len(idxs)); val = idxs[:nval]
print(f"val slides={len(val)}", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# pooled across all 3 seeds
rows = []  # (margin, ent_norm, correct)

for seed in SEEDS:
    ckpt = f"/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints/reg_attr_h0base_seed{seed}.pt"
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = REGAttributeModel(lm, in_dim=1536, per_task_attention=ck.get("per_task_attention", False)).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    seed_rows = []
    with torch.no_grad():
        for i in val:
            f, t, o, sid = ds[i]
            if not t: continue
            f = f.to(device)
            logits, attns = model(f, questions=list(t.keys()), return_attn=True)
            for q, tgt in t.items():
                lg = logits[q]
                probs = torch.softmax(lg, dim=-1)
                k = min(2, probs.numel())
                top = torch.topk(probs, k=k)
                margin = float(top.values[0] - (top.values[1] if k > 1 else 0.0))
                correct = int(lg.argmax()) == tgt
                a = attns[q]
                N = a.numel()
                ent = float(-(a * (a.clamp_min(1e-12)).log()).sum())
                ent_norm = ent / math.log(N) if N > 1 else 0.0
                seed_rows.append((margin, ent_norm, correct))
    rows += seed_rows

    # per-seed report
    def stats(subset):
        if not subset: return None
        ents = [r[1] for r in subset]
        return sum(ents) / len(ents)
    high_margin = [r for r in seed_rows if r[0] > 0.5]  # "confident" per softmax
    hm_correct = [r for r in high_margin if r[2]]
    hm_wrong = [r for r in high_margin if not r[2]]
    print(f"\n== seed {seed} == total preds={len(seed_rows)}  high-margin(>0.5)={len(high_margin)} "
          f"(wrong among them={len(hm_wrong)})")
    print(f"  high-margin subset: mean attn-entropy  correct={stats(hm_correct)}  wrong={stats(hm_wrong)}")

# pooled across all seeds
high_margin = [r for r in rows if r[0] > 0.5]
hm_correct = [r for r in high_margin if r[2]]
hm_wrong = [r for r in high_margin if not r[2]]
print(f"\n===== POOLED (3 seeds) ===== total={len(rows)}  high-margin={len(high_margin)}  wrong-among-high-margin={len(hm_wrong)}")
if hm_correct and hm_wrong:
    mc = sum(r[1] for r in hm_correct) / len(hm_correct)
    mw = sum(r[1] for r in hm_wrong) / len(hm_wrong)
    print(f"  mean attn-entropy | correct={mc:.4f}  wrong={mw:.4f}  diff(wrong-correct)={mw-mc:+.4f}")

    # does thresholding on entropy alone (within the high-margin/"confident" subset) catch errors
    # better than random? Rank by entropy desc, see what fraction of top-K are actually wrong.
    ranked = sorted(high_margin, key=lambda r: -r[1])
    base_rate = len(hm_wrong) / len(high_margin)
    for frac in (0.05, 0.10, 0.20):
        k = max(1, int(frac * len(ranked)))
        topk = ranked[:k]
        wrong_in_topk = sum(1 for r in topk if not r[2])
        print(f"  top {frac:.0%} highest-entropy (within confident subset): "
              f"{wrong_in_topk}/{k} wrong ({wrong_in_topk/k:.1%}) vs base rate {base_rate:.1%}")
else:
    print("  not enough high-margin wrong or correct samples to compare.")
