"""
Confidence-weighted decision-path decoding over the CoT decision graph.

The standard traversal (traversal_engine.assemble_cot) takes the model's single argmax
answer per question and follows it deterministically. It already hedges on STRUCTURAL
ambiguity (multiple next_questions kept when several follow an answer often enough in
training data, via `threshold`), but never reconsiders an ANSWER the model itself is
unsure about -- even when a low-margin call at an early gate determines which entire
downstream branch gets asked, and one wrong gate call cascades into several wrong edges.

This module adds that second axis of hedging: for the handful of questions on the greedy
path where the model's top-2 answers are close, try both, re-run the traversal for each
resulting answer assignment, and keep whichever assignment's VISITED nodes have the
highest length-normalized log-likelihood under the model's own softmax. This is a direct
analogue of length-normalized beam-search scoring in seq2seq decoding, applied to a
symbolic decision-graph traversal instead of a token sequence. It uses no ground truth,
so it is deployable at real inference (not just an offline eval trick).
"""
from __future__ import annotations
import math
from itertools import product
import torch
from traversal_engine import assemble_cot


def _top2_from_logits(logits: torch.Tensor, labels: list[str]):
    probs = torch.softmax(logits, dim=-1)
    k = min(2, probs.numel())
    top = torch.topk(probs, k=k)
    return [(labels[i], float(top.values[j])) for j, i in enumerate(top.indices.tolist())]


GATES = {"Is there any abnormality present?", "Is there any neoplasm present?",
         "What is the histologic type of neoplasm?"}


def assemble_cot_confidence(organ, logits: dict[str, torch.Tensor], lm: dict[str, list[str]],
                             graph, roots, *, threshold: float = 0.10,
                             margin_thresh: float = 0.15, max_branch: int = 6,
                             exclude_gates: bool = False):
    """logits: {question: raw logit tensor}, for whatever subset of questions the caller
    has available (oracle-lite eval passes only GT-asked questions; real inference passes
    all 93 -- both work unchanged).

    Returns (steps, pred) for the highest-scoring candidate answer assignment.
    """
    top2 = {q: _top2_from_logits(lg, lm[q]) for q, lg in logits.items()}
    greedy_pred = {q: t[0][0] for q, t in top2.items()}
    logsoftmax = {q: torch.log_softmax(lg, dim=-1) for q, lg in logits.items()}
    logprob = {q: {a: float(logsoftmax[q][i]) for i, a in enumerate(lm[q])} for q in logits}

    greedy_steps = assemble_cot(organ, lambda q: greedy_pred.get(q), graph, roots, threshold=threshold)

    def visited_of(steps):
        v = {s["question"] for s in steps} | {s["next_question"] for s in steps}
        v.discard("")
        return v

    visited0 = visited_of(greedy_steps)

    # Candidate pool: visited questions with a real top-2 alternative and a close margin.
    pool = [q for q in visited0 if q in top2 and len(top2[q]) > 1
            and (top2[q][0][1] - top2[q][1][1]) < margin_thresh
            and not (exclude_gates and q in GATES)]
    pool = sorted(pool, key=lambda q: top2[q][0][1] - top2[q][1][1])[:max_branch]

    def score(pred, steps):
        vis = visited_of(steps)
        lp = [logprob[q][pred[q]] for q in vis if q in pred and q in logprob]
        return sum(lp) / len(lp) if lp else float("-inf")

    best_pred, best_steps, best_score = greedy_pred, greedy_steps, score(greedy_pred, greedy_steps)
    for bits in product((0, 1), repeat=len(pool)):
        if not any(bits):
            continue  # all-zero bits == greedy, already scored
        cand_pred = dict(greedy_pred)
        for q, bit in zip(pool, bits):
            if bit:
                cand_pred[q] = top2[q][1][0]
        cand_steps = assemble_cot(organ, lambda q: cand_pred.get(q), graph, roots, threshold=threshold)
        s = score(cand_pred, cand_steps)
        if s > best_score:
            best_score, best_pred, best_steps = s, cand_pred, cand_steps
    return best_steps, best_pred
