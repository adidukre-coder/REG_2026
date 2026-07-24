"""
REG2026 Interface-1 vision model — multi-task attribute predictor.

Design recap (see cot_artifacts/vision_model_design.md + FINDINGS.md):
  - Input per slide: H-Optimus-0 patch features  x = [N, 1536]  (N varies per slide;
    1536 = H-Optimus-0 ViT-giant embed dim — confirmed from the real .h5 files).
  - The CoT is a decision tree; the model ONLY predicts the diagnostic ANSWERS
    (closed-set classification per question). The traversal_engine turns answers -> CoT.
  - 93 questions total; each is a classification head over its answer vocab (label_maps.json).
  - Per-ATTRIBUTE attention pooling (ABMIL): each question attends to the patches that
    matter FOR THAT QUESTION (mitoses=focal, architecture=global) and yields an evidence map.
  - Heads are organ-agnostic at the module level; organ-conditioning happens at INFERENCE
    because the engine only asks the questions reachable for the predicted organ.
  - Loss = masked (only asked questions) multi-task cross-entropy, weighted by Edge-F1
    criticality (gates >> graders > trivial) with class weights on the rare gate classes.

Training is single-bag (one slide at a time, variable N) — standard for MIL; use gradient
accumulation for an effective batch. H-Optimus stays FROZEN (features are precomputed).
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# ABMIL gated attention (Ilse et al., 2018) — learns which patches matter, pools to [D].
# --------------------------------------------------------------------------------------
class GatedAttention(nn.Module):
    def __init__(self, in_dim: int = 1536, hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden)        # tanh branch
        self.U = nn.Linear(in_dim, hidden)        # sigmoid gate branch
        self.w = nn.Linear(hidden, 1)             # attention logit per patch
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor):
        # h: [N, D]
        a = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))   # [N, hidden]
        a = self.w(self.drop(a))                                # [N, 1]
        attn = torch.softmax(a, dim=0)                          # [N, 1] sums to 1 over patches
        pooled = (attn * h).sum(dim=0)                          # [D]
        return pooled, attn.squeeze(-1)                         # [D], [N]


# --------------------------------------------------------------------------------------
# Multi-task attribute model.
# --------------------------------------------------------------------------------------
class REGAttributeModel(nn.Module):
    def __init__(
        self,
        label_maps: dict[str, list[str]],
        in_dim: int = 1536,
        hidden: int = 256,
        per_task_attention: bool = True,
        dropout: float = 0.25,
    ):
        super().__init__()
        # stable ordering of questions; map question <-> index
        self.questions = list(label_maps.keys())
        self.q2idx = {q: i for i, q in enumerate(self.questions)}
        self.n_classes = [len(label_maps[q]) for q in self.questions]
        self.per_task_attention = per_task_attention

        if per_task_attention:
            # one attention pooling PER question (design choice: each looks where it should)
            self.attn = nn.ModuleList(
                [GatedAttention(in_dim, hidden, dropout) for _ in self.questions]
            )
        else:
            # one shared attention pooling (cheaper baseline / ablation)
            self.attn = GatedAttention(in_dim, hidden, dropout)

        # one linear classifier per question, sized from its answer vocab
        self.heads = nn.ModuleList(
            [nn.Linear(in_dim, c) for c in self.n_classes]
        )

    def forward(self, x: torch.Tensor, questions: list[str] | None = None,
                return_attn: bool = False):
        """
        x: [N, in_dim] patch features for ONE slide.
        questions: subset to compute (default: all). At inference the engine passes only
                   the question currently being asked -> cheap.
        Returns: {question: logits[n_classes]}  (and {question: attn[N]} if return_attn).
        """
        qs = questions if questions is not None else self.questions
        logits, attns = {}, {}
        if not self.per_task_attention:
            pooled_shared, attn_shared = self.attn(x)
        for q in qs:
            i = self.q2idx[q]
            if self.per_task_attention:
                pooled, attn = self.attn[i](x)
            else:
                pooled, attn = pooled_shared, attn_shared
            logits[q] = self.heads[i](pooled)
            if return_attn:
                attns[q] = attn
        return (logits, attns) if return_attn else logits

    def predict_answer(self, x: torch.Tensor, question: str, label_maps: dict) -> str:
        """Inference helper: return the top answer STRING for one question (for the engine)."""
        with torch.no_grad():
            lg = self.forward(x, questions=[question])[question]
            idx = int(lg.argmax())
        return label_maps[question][idx]


# --------------------------------------------------------------------------------------
# Loss — masked, criticality-weighted, with optional per-class weights on the gates.
# --------------------------------------------------------------------------------------
def masked_multitask_loss(
    logits: dict[str, torch.Tensor],
    targets: dict[str, int],            # {question: class_idx} — only asked questions present
    crit_weight: dict[str, float],      # {question: Edge-F1 criticality weight}
    class_weights: dict[str, torch.Tensor] | None = None,  # {question: tensor[n_classes]}
):
    """Sum CE over the questions this slide actually asked, scaled by attribute criticality."""
    total = 0.0
    n = 0
    for q, tgt in targets.items():
        if q not in logits:
            continue
        w = class_weights.get(q) if class_weights else None
        ce = F.cross_entropy(
            logits[q].unsqueeze(0),
            torch.tensor([tgt], device=logits[q].device),
            weight=w,
        )
        total = total + crit_weight.get(q, 1.0) * ce
        n += 1
    return total / max(n, 1)


# --------------------------------------------------------------------------------------
# Helpers to build label maps / criticality weights from our artifacts.
# --------------------------------------------------------------------------------------
def load_label_maps(path: str | Path) -> dict[str, list[str]]:
    return json.load(open(path))


def default_criticality_weights(label_maps: dict[str, list[str]]) -> dict[str, float]:
    """
    Weight each question by its measured Edge-F1 impact (FINDINGS.md). Gates dominate.
    Unlisted questions default to 1.0. Tune later.
    """
    base = {q: 1.0 for q in label_maps}
    base.update({
        "Is there any abnormality present?": 6.0,
        "Is there any neoplasm present?": 5.0,
        "What is the histologic type of neoplasm?": 3.0,
    })
    return base
