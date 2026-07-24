"""
REG2026 Interface-1 — train the multi-task attribute model.

Run on gpu-57:
  conda activate reg2026
  python src/train.py --feat_dir <features_hoptimus0 dir> --epochs 30

Trains REGAttributeModel on (H-Optimus features, CoT-derived attribute labels) with a
masked, criticality-weighted, class-balanced loss. Evaluates per-attribute accuracy
(especially the 3 cascade-critical GATES) on a held-out split. Saves best checkpoint to
/nfs-stor/.../reg2026_work/checkpoints/.
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path
import torch
from src.dataset import REGFeatureDataset, compute_class_weights
from src.model import (REGAttributeModel, masked_multitask_loss,
                       load_label_maps, default_criticality_weights)

LAB = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json"
LM  = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/label_maps.json"
CKPT_DIR = Path("/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints")
GATES = ["Is there any abnormality present?", "Is there any neoplasm present?",
         "What is the histologic type of neoplasm?"]


@torch.no_grad()
def evaluate(model, ds, idxs, device):
    """Per-question accuracy over val slides; returns (overall, gate_acc dict)."""
    model.eval()
    correct = defaultdict(int); total = defaultdict(int)
    for i in idxs:
        feats, tgt, organ, sid = ds[i]
        feats = feats.to(device)
        if not tgt: continue
        logits = model(feats, questions=list(tgt.keys()))
        for q, y in tgt.items():
            pred = int(logits[q].argmax())
            correct[q] += (pred == y); total[q] += 1
    accs = {q: correct[q]/total[q] for q in total}
    overall = sum(correct.values())/max(sum(total.values()), 1)
    gate_acc = {q: accs.get(q) for q in GATES if q in accs}
    return overall, gate_acc, accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True, help="features_hoptimus0 directory")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--accum", type=int, default=8, help="grad-accumulation steps (effective batch)")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--per_task_attention", action="store_true",
                    help="per-attribute attention (better but ~30x slower); default off = fast shared attention")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = load_label_maps(LM)
    ds = REGFeatureDataset(args.feat_dir, LAB, lm)
    print(f"paired slides: {len(ds)} | device: {device}")
    if len(ds) < 10:
        print("WARNING: very few paired slides — extract more features before real training.")

    # split by slide
    idx = list(range(len(ds))); random.Random(0).shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    crit = default_criticality_weights(lm)
    print("computing class weights (imbalance handling)...")
    cls_w = compute_class_weights(LAB, lm, list(lm.keys()))
    cls_w = {q: w.to(device) for q, w in cls_w.items()}

    model = REGAttributeModel(lm, in_dim=1536, per_task_attention=args.per_task_attention).to(device)
    print(f"per_task_attention={args.per_task_attention} | params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train(); random.Random(ep).shuffle(train_idx)
        running = 0.0; opt.zero_grad()
        for step, i in enumerate(train_idx, 1):
            feats, tgt, organ, sid = ds[i]
            if not tgt: continue
            feats = feats.to(device)
            logits = model(feats, questions=list(tgt.keys()))
            loss = masked_multitask_loss(logits, tgt, crit, cls_w) / args.accum
            loss.backward(); running += loss.item() * args.accum
            if step % args.accum == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()

        overall, gate_acc, _ = evaluate(model, ds, val_idx, device)
        gates_str = " ".join(f"{q.split()[3][:6]}={v:.2f}" for q, v in gate_acc.items())
        print(f"epoch {ep:3d} | train_loss {running/max(len(train_idx),1):.3f} "
              f"| val_acc {overall:.3f} | gates[{gates_str}]")
        if overall > best:
            best = overall
            torch.save({"model": model.state_dict(), "label_maps": lm,
                        "val_acc": overall, "epoch": ep},
                       CKPT_DIR / f"reg_attr_{args.tag}_best.pt")
    print(f"done. best val_acc={best:.3f}  -> {CKPT_DIR}/reg_attr_{args.tag}_best.pt")


if __name__ == "__main__":
    main()
