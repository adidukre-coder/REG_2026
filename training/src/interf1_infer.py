"""
Interface-1 inference glue: slide features -> chain-of-thought JSON.
  model heads (predict diagnostic answers)  ->  traversal_engine (build CoT structure)
  ->  compose_report (final report)  ->  list of {question, answer, next_question}.
This is the core of predict_chain_of_thought (used both for local scoring and, later,
inside the submission container).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cot_artifacts"))
from src.model import REGAttributeModel
from src.compose_report import compose_report
from traversal_engine import load_graph, assemble_cot

LAB = "/nfs-stor/adinath.dukre/adinath/reg2026_work/labels/train_labels.json"
REPORT_Q = "What is the final pathology report?"
ORGAN_Q = "What is the organ?"


def build_organ_answer_to_key(labels_json=LAB):
    """Map the predicted 'What is the organ?' answer (e.g. 'Breast') -> graph organ key ('breast').

    Prefers the precomputed cot_artifacts/organ_answer_map.json (bundled in the container, no
    train_labels.json needed); falls back to computing it from labels_json when available.
    """
    precomp = Path(__file__).resolve().parent.parent / "cot_artifacts" / "organ_answer_map.json"
    if precomp.exists():
        return json.load(open(precomp))
    from collections import Counter, defaultdict
    recs = json.load(open(labels_json))
    by = defaultdict(Counter)
    for r in recs:
        a = r["labels"].get(ORGAN_Q)
        if a:
            by[a][r["organ"]] += 1
    return {ans: c.most_common(1)[0][0] for ans, c in by.items()}


def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    lm = ck["label_maps"]
    model = REGAttributeModel(lm, in_dim=1536,
                              per_task_attention=ck.get("per_task_attention", False)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, lm


def predict_chain_of_thought_from_features(features, model, lm, graph, roots,
                                           organ_map, threshold=0.25):
    """features: [N,1536] tensor on the model's device -> list of CoT step dicts."""
    with torch.no_grad():
        logits = model(features)                      # all 93 heads, one forward
    pred = {q: lm[q][int(logits[q].argmax())] for q in lm}

    organ_ans = pred.get(ORGAN_Q, "")
    organ_key = organ_map.get(organ_ans, organ_ans.lower())  # 'Breast' -> 'breast'
    if organ_key not in roots:                        # safety fallback
        organ_key = min(roots, key=lambda k: 0)       # pick any valid key (rare)

    steps = assemble_cot(organ_key, lambda q: pred.get(q), graph, roots, threshold=threshold)

    # Compose the report ONLY from the attributes the traversal actually visited for this slide.
    # Feeding all 93 raw heads pollutes the report with spurious diagnoses/fields GT never asked
    # (the decision tree prunes them). This pruning lifts report token-F1 ~0.51 -> ~0.78 (free).
    visited = {s["question"] for s in steps} | {s["next_question"] for s in steps}
    visited.discard("")
    visited.discard(REPORT_Q)
    # "What is the procedure?" is a header field every report needs, but a low-confidence edge
    # occasionally lets the traversal skip straight past it (same short-circuit that can also skip
    # "#1 diagnosis" - handled separately in compose_report's fallback). Always include it if the
    # model predicted one, regardless of whether the traversal visited it as a step.
    visited.add("What is the procedure?")
    pred_active = {q: pred[q] for q in visited if q in pred}

    # GUARANTEE the CoT ends with the final-report step (challenge requires it; report=40% of Metric A).
    report = compose_report(pred_active)
    others = [s for s in steps if s["question"] != REPORT_Q]
    # ensure an incoming edge to the report exists (else link the last reached step to it)
    if others and not any(s["next_question"] == REPORT_Q for s in others):
        others[-1]["next_question"] = REPORT_Q
    steps = others + [{"question": REPORT_Q, "answer": report, "next_question": ""}]
    return steps
