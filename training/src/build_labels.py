"""
REG2026 — build supervised training labels for the Interface-1 vision heads.

CODE lives in repo (src/); OUTPUT (derived labels) goes to /nfs-stor work/labels.

Produces:
  labels/label_maps.json    {question: [answer0, answer1, ...]}  (index = class id; idx0 = most frequent)
  labels/train_labels.json  [{id, organ, labels: {question: answer}}]  one per slide
Each head = a classifier over its question's answer list. A slide only carries labels
for the questions it was asked (the rest are masked out at training time).
"""
import json
from collections import Counter
from pathlib import Path

DATA = Path("/nfs-stor/adinath.dukre/adinath/reg2026/train_CoT_v01.json")
ART = Path("/home/adinath.dukre/adinath/reg_2026/REG2026/cot_artifacts")
OUT = Path("/nfs-stor/adinath.dukre/adinath/reg2026_work/labels")
OUT.mkdir(parents=True, exist_ok=True)

cot = json.load(open(DATA))
vocab = json.load(open(ART / "answer_vocab_global.json"))  # question -> {answer: count}

# 1) label maps: per question, answers ordered by frequency desc (idx 0 = majority class)
label_maps = {
    q: [a for a, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
    for q, counts in vocab.items()
}
json.dump(label_maps, open(OUT / "label_maps.json", "w"), indent=1, ensure_ascii=False)

# 2) per-slide labels (most-common answer per question handles the ~4% multi-answer cases)
records = []
for c in cot:
    by_q = {}
    for s in c["chain-of-thought"]:
        if not s["question"]:
            continue
        by_q.setdefault(s["question"], Counter())[s["answer"]] += 1
    labels = {q: cc.most_common(1)[0][0] for q, cc in by_q.items()}
    records.append({"id": c["id"], "organ": c.get("organ", "?"), "labels": labels})
json.dump(records, open(OUT / "train_labels.json", "w"), indent=1, ensure_ascii=False)

# 3) summary
n_q = len(label_maps)
asked_counts = Counter(q for r in records for q in r["labels"])
print(f"slides labeled       : {len(records)}")
print(f"distinct attributes  : {n_q}")
print(f"label_maps.json      : {OUT/'label_maps.json'}")
print(f"train_labels.json    : {OUT/'train_labels.json'}")
print("\nexample slide record:")
print(json.dumps(records[0], indent=1, ensure_ascii=False)[:500])
print("\nclass counts for the 3 gate attributes (idx0 = majority):")
for q in ["Is there any abnormality present?", "Is there any neoplasm present?",
          "What is the histologic type of neoplasm?"]:
    if q in label_maps:
        print(f"  {q}  -> {len(label_maps[q])} classes; asked in {asked_counts[q]} slides")
