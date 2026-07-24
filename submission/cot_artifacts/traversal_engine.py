"""
REG2026 — CoT Traversal Engine.

Deterministic chain-of-thought assembler. Given:
  - the organ,
  - an answer function  answer(question) -> answer_string  (supplied by the vision model),
  - the canonical decision graph extracted from training data,
it emits a list of {question, answer, next_question} steps (the Interface-1 output).

The ML model ONLY supplies answers; this engine owns all the question/structure logic,
which is what makes Edge-F1 + path-validity nearly free when answers are correct.
"""
from __future__ import annotations
import json
from collections import deque
from pathlib import Path

ART = Path(__file__).resolve().parent


def load_graph(art_dir: Path = ART):
    graph = json.load(open(art_dir / "decision_graph.json"))
    roots = json.load(open(art_dir / "root_questions.json"))
    return graph, roots


def children_for(graph, organ, question, answer, *, threshold=0.0, asked_set=None):
    """
    Return the list of next_questions that follow (organ, question, answer).

    Two selection modes:
      - asked_set given  -> ORACLE: keep children whose question is in asked_set (upper bound).
      - else THRESHOLD   -> INFERENCE: keep children with P(next|organ,q,a) >= threshold,
                            always keeping the most likely child so the chain never dies.
    """
    nq_counts = graph.get(organ, {}).get(question, {}).get(answer, {})
    if not nq_counts:
        return []
    total = sum(nq_counts.values())
    ranked = sorted(nq_counts.items(), key=lambda kv: -kv[1])
    if asked_set is not None:
        return [nq for nq, _ in ranked if nq == "" or nq in asked_set]
    keep = [nq for nq, c in ranked if c / total >= threshold]
    if not keep:
        keep = [ranked[0][0]]  # never let the chain die
    return keep


def assemble_cot(organ, answer_fn, graph, roots, *, threshold=0.0, asked_set=None, max_steps=200):
    """BFS over the decision graph from the organ's root questions; emit ordered steps."""
    root_qs = [q for q in roots.get(organ, []) if q]  # drop empty-string roots
    queue = deque(root_qs)
    enqueued = set(root_qs)
    steps = []
    while queue and len(steps) < max_steps:
        q = queue.popleft()
        a = answer_fn(q)
        if a is None:
            continue
        for nq in children_for(graph, organ, q, a, threshold=threshold, asked_set=asked_set):
            steps.append({"question": q, "answer": a, "next_question": nq})
            if nq and nq not in enqueued:
                enqueued.add(nq)
                queue.append(nq)
    return steps


def edge_set(steps):
    return {(s["question"], s["next_question"]) for s in steps}
