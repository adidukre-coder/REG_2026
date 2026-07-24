"""
Faithfulness check for per-task attention: does masking out the patches a question's OWN
attention module flags as important actually disrupt THAT question's prediction more than
masking out patches flagged as important by a DIFFERENT question, or masking random patches?

Motivation: "attention is not explanation" (Jain & Wallace 2019) is a standard critique --
attention weights can be high without the model's prediction actually depending on those
positions. Per-task attention only earns the "evidence grounding" claim if it survives this
causal check, not just a visual/entropy correlation.

Design, per probed (slide, question) pair:
  - baseline: full patch bag -> logits[q], record softmax prob of the ORIGINAL predicted class
  - own-mask:   remove top-K patches by attn[q]        -> recompute logits[q] on remainder
  - other-mask: remove top-K patches by attn[q'] (q' != q, randomly chosen from same slide's
                asked questions) -> recompute logits[q] on remainder
  - rand-mask:  remove K random patches                 -> recompute logits[q] on remainder
K = 15% of patches (or at least 5, whichever larger).

If per-task attention is faithful: own-mask should cause a bigger probability drop (and
higher flip rate) for q's prediction than other-mask or rand-mask, on average.

Cheap: inference-only, subsample of val slides, a handful of probed questions per slide.
Usage: python src/faithfulness_ablation.py [ckpt_name] [n_slides] [q_per_slide]
"""
import json, random, sys
from collections import defaultdict
import torch
sys.path.insert(0, "cot_artifacts")
from src.dataset import REGFeatureDataset
from src.model import REGAttributeModel, load_label_maps

FEAT = "/nfs-stor/adinath.dukre/adinath/reg2026_work/features/20x_256px_0px_overlap/features_hoptimus0"
LAB = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json"
LM = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/label_maps.json"
CKDIR = "/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints"
PER_ORGAN = 3000

CKNAME = sys.argv[1] if len(sys.argv) > 1 else "seed42"
N_SLIDES = int(sys.argv[2]) if len(sys.argv) > 2 else 200
Q_PER_SLIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 4
MASK_FRAC = 0.15
PROBE_SEED = 999

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

probe_rng = random.Random(PROBE_SEED)
sample_val = probe_rng.sample(val, min(N_SLIDES, len(val)))
print(f"probing {len(sample_val)} slides, {Q_PER_SLIDE} questions/slide, mask_frac={MASK_FRAC}", flush=True)

drop_own, drop_other, drop_rand = [], [], []
flip_own, flip_other, flip_rand = [], [], []

with torch.no_grad():
    for n, i in enumerate(sample_val, 1):
        f, t, o, sid = ds[i]
        if len(t) < 2:
            continue
        f = f.to(device)
        qs = list(t.keys())
        logits0, attns0 = model(f, questions=qs, return_attn=True)
        Np = f.shape[0]
        K = max(5, int(MASK_FRAC * Np))

        probe_qs = probe_rng.sample(qs, min(Q_PER_SLIDE, len(qs)))
        for q in probe_qs:
            others = [q2 for q2 in qs if q2 != q]
            if not others:
                continue
            q_other = probe_rng.choice(others)

            lg0 = logits0[q]
            p0 = torch.softmax(lg0, dim=-1)
            pred0 = int(lg0.argmax())
            base_prob = float(p0[pred0])

            def masked_prob(rank_source):
                if rank_source == "rand":
                    perm = torch.randperm(Np)
                    drop_idx = perm[:K]
                else:
                    a = attns0[rank_source]
                    drop_idx = torch.argsort(-a)[:K]
                keep_mask = torch.ones(Np, dtype=torch.bool)
                keep_mask[drop_idx] = False
                f_sub = f[keep_mask]
                lg = model(f_sub, questions=[q])[q]
                p = torch.softmax(lg, dim=-1)
                return float(p[pred0]), int(lg.argmax()) != pred0

            p_own, flip1 = masked_prob(q)
            p_other, flip2 = masked_prob(q_other)
            p_rand, flip3 = masked_prob("rand")

            drop_own.append(base_prob - p_own)
            drop_other.append(base_prob - p_other)
            drop_rand.append(base_prob - p_rand)
            flip_own.append(flip1); flip_other.append(flip2); flip_rand.append(flip3)

        if n % 50 == 0:
            print(f"  {n}/{len(sample_val)} slides done", flush=True)

def mean(x): return sum(x) / len(x) if x else 0.0
def rate(x): return sum(x) / len(x) if x else 0.0

print(f"\n===== FAITHFULNESS ABLATION ({CKNAME}, n_probes={len(drop_own)}) =====")
print(f"  mean prob-drop  own-mask={mean(drop_own):+.4f}  other-mask={mean(drop_other):+.4f}  rand-mask={mean(drop_rand):+.4f}")
print(f"  flip-rate       own-mask={rate(flip_own):.3%}     other-mask={rate(flip_other):.3%}     rand-mask={rate(flip_rand):.3%}")
print(f"  own vs other prob-drop delta: {mean(drop_own) - mean(drop_other):+.4f}")
print(f"  own vs rand  prob-drop delta: {mean(drop_own) - mean(drop_rand):+.4f}")

out = {
    "ckpt": CKNAME, "n_probes": len(drop_own),
    "mean_drop": {"own": mean(drop_own), "other": mean(drop_other), "rand": mean(drop_rand)},
    "flip_rate": {"own": rate(flip_own), "other": rate(flip_other), "rand": rate(flip_rand)},
}
outp = f"{CKDIR}/faithfulness_{CKNAME}.json"
with open(outp, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"saved -> {outp}")
