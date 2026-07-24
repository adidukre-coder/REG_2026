"""
REG2026 Interface-1 dataset — pairs H-Optimus features with the CoT-derived labels.

Feature files (from TRIDENT):  <feat_dir>/<slide_stem>.h5  with keys 'features' [N,1536], 'coords' [N,2]
Labels (build_labels.py):      train_labels.json  [{id, organ, labels:{question:answer}}]
Label maps:                    label_maps.json    {question: [answer0, answer1, ...]}  (idx = class)

A sample = (features[N,1536], {question: class_idx for asked questions}, organ, slide_id).
Variable N per slide → use batch_size=1 (standard for MIL) or the list collate below.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _stem(s: str) -> str:
    return Path(str(s)).stem  # "PIT_01_00019_01.tiff" -> "PIT_01_00019_01"


class REGFeatureDataset(Dataset):
    def __init__(self, feat_dir, labels_json, label_maps, organs=None):
        self.feat_dir = Path(feat_dir)
        self.label_maps = label_maps
        recs = json.load(open(labels_json))
        # skip zero-byte / unwritten files up front (robust against interrupted writes)
        avail = {_stem(p.name): p for p in self.feat_dir.glob("*.h5") if p.stat().st_size > 0}
        self.samples = []
        for r in recs:
            st = _stem(r["id"])
            if st in avail and (organs is None or r["organ"] in organs):
                # convert answer strings -> class indices using label_maps
                tgt = {}
                for q, a in r["labels"].items():
                    if q in label_maps and a in label_maps[q]:
                        tgt[q] = label_maps[q].index(a)
                self.samples.append({"path": avail[st], "organ": r["organ"],
                                     "id": st, "targets": tgt})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        try:
            with h5py.File(s["path"], "r") as h:
                feats = torch.from_numpy(np.asarray(h["features"], dtype=np.float32))  # [N,1536]
        except Exception:
            # corrupt/unreadable file -> return empty so the training loop skips it (if not t)
            return torch.zeros(1, 1536), {}, s["organ"], s["id"]
        return feats, s["targets"], s["organ"], s["id"]


def list_collate(batch):
    """Keep variable-N bags as a list (no padding)."""
    return list(batch)


def compute_class_weights(labels_json, label_maps, questions):
    """Inverse-frequency class weights per question (for imbalance-aware gate training)."""
    recs = json.load(open(labels_json))
    weights = {}
    for q in questions:
        n_cls = len(label_maps[q])
        cnt = Counter()
        for r in recs:
            a = r["labels"].get(q)
            if a in label_maps[q]:
                cnt[label_maps[q].index(a)] += 1
        total = sum(cnt.values())
        if total == 0:
            continue
        w = torch.tensor([total / (n_cls * max(cnt.get(c, 0), 1)) for c in range(n_cls)],
                         dtype=torch.float32)
        weights[q] = w / w.mean()  # normalize around 1
    return weights
