"""
Build the evidence-grounding demo payload: for each of the 5 coordinate-backfilled pilot
slides (see backfill_coords_check.py -- confirmed cosine_sim=1.0000, exact row-alignment
between coords_h5 and the already-saved features), run the greedy CoT traversal, and for
every visited question record:
  - predicted answer + GT answer (correctness)
  - attention-entropy confidence flag (validated signal, attn_entropy_probe.py)
  - the top-K attended patch coordinates in LEVEL-0 pixel space (the literal spatial
    "evidence" for that answer, straight from the model's own already-computed ABMIL
    attention -- no extra training, no extra inference cost beyond return_attn=True).

Output: one JSON per slide under evidence_demo/ for the artifact-rendering step to consume.
"""
import glob, json, math, os, sys
sys.path.insert(0, "cot_artifacts")
import h5py, numpy as np, torch
from src.dataset import REGFeatureDataset
from src.model import REGAttributeModel, load_label_maps
from traversal_engine import load_graph, assemble_cot

PILOT_FEAT = "/nfs-stor/adinath.dukre/adinath/reg2026_work/features_hoptimus0_pilot"
BACKFILL = "/nfs-stor/adinath.dukre/adinath/reg2026_work/coord_backfill_check"
LAB = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json"
LM = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/label_maps.json"
COT = "/nfs-stor/adinath.dukre/adinath/reg2026/train_CoT_v01.json"
CKPT = os.environ.get("REG_DEMO_CKPT", "/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints/reg_attr_pertask_seed42.pt")
OUT = os.environ.get("REG_DEMO_OUT", "/nfs-stor/adinath.dukre/adinath/reg2026_work/evidence_demo_pertask")
os.makedirs(OUT, exist_ok=True)
TOPK = 5

lm = load_label_maps(LM)
labels = {r["id"].replace(".tiff", ""): r for r in json.load(open(LAB))}
cot = {c["id"]: c for c in json.load(open(COT))}
graph, roots = load_graph()

device = "cpu"
ck = torch.load(CKPT, map_location=device, weights_only=False)
model = REGAttributeModel(lm, in_dim=1536, per_task_attention=ck.get("per_task_attention", False)).to(device)
model.load_state_dict(ck["model"]); model.eval()


def gt_map(sid):
    c = cot.get(sid if sid.endswith(".tiff") else sid + ".tiff")
    if not c:
        return {}
    return {s["question"]: s["answer"] for s in c["chain-of-thought"] if s["question"]}


stems = sorted(os.path.basename(p) for p in os.listdir(BACKFILL))
print(f"{len(stems)} backfilled slides: {stems}", flush=True)

for stem in stems:
    organ = labels.get(stem, {}).get("organ")
    if not organ:
        print(f"  {stem}: no organ label, skip"); continue

    with h5py.File(os.path.join(PILOT_FEAT, stem + ".h5")) as h:
        feats = torch.from_numpy(np.asarray(h["features"], dtype=np.float32))
    coords_glob = glob.glob(os.path.join(BACKFILL, stem, "job", "*", "patches", "*_patches.h5"))
    with h5py.File(coords_glob[0]) as h:
        coords = np.asarray(h["coords"])  # [N, 2] level-0 pixel (x, y) top-left per patch
    assert coords.shape[0] == feats.shape[0], f"{stem}: coord/feature count mismatch"

    gt = gt_map(stem)
    qs = list(lm.keys())  # real inference computes all heads in one pass -- same here
    with torch.no_grad():
        logits, attns = model(feats, questions=qs, return_attn=True)

    pred = {q: lm[q][int(logits[q].argmax())] for q in qs}
    steps = assemble_cot(organ, lambda q: pred.get(q), graph, roots, threshold=0.25)
    visited = sorted({s["question"] for s in steps} | {s["next_question"] for s in steps} - {""})

    out_qs = []
    for q in visited:
        a = attns[q].numpy()  # [N], softmax over patches, sums to 1
        N = len(a)
        ent = float(-(a * np.log(np.clip(a, 1e-12, None))).sum())
        ent_norm = ent / math.log(N) if N > 1 else 0.0
        top_idx = np.argsort(-a)[:TOPK]
        evidence = [{"x": int(coords[i][0]), "y": int(coords[i][1]), "weight": float(a[i])} for i in top_idx]
        out_qs.append({
            "question": q,
            "answer": pred[q],
            "gt_answer": gt.get(q),
            "correct": (gt.get(q) == pred[q]) if q in gt else None,
            "attn_entropy_norm": ent_norm,
            "n_patches": N,
            "evidence": evidence,
        })

    payload = {"stem": stem, "organ": organ, "questions": out_qs}
    with open(os.path.join(OUT, stem + ".json"), "w") as f:
        json.dump(payload, f, indent=1)
    print(f"  {stem} ({organ}): {len(out_qs)} visited questions -> {OUT}/{stem}.json", flush=True)

print("done.")
