"""
Classify the local sample ROIs with the Interface-0 detector and write the participant
predictions file. Also reports accuracy vs the known labels in rois_mapping.txt.
Writes: submission_evaluation_code/data/predictions/interf0/predictions.json  [{id, answer}]
"""
import json, os, sys
sys.path.insert(0, "src")
from interf0_infer import load_rgb, roi_is_tissue, ANSWER_TISSUE, ANSWER_BACKGROUND

ROI = "submission_evaluation_code/data/cases/interf0"
OUT = "submission_evaluation_code/data/predictions/interf0/predictions.json"

rows = [l.rstrip("\n").split("\t") for l in open(f"{ROI}/rois_mapping.txt")]
hdr = rows[0]; idx = {c: i for i, c in enumerate(hdr)}

preds = []; correct = 0; n = 0
print(f"{'id':12s}{'gt_label':11s}{'pred':11s}{'ok':>4}")
for r in rows[1:]:
    aid = r[idx["anonymous_id"]]; img = r[idx["image"]]; gt = r[idx["label"]]
    rgb = load_rgb(f"{ROI}/{img}")
    is_t = roi_is_tissue(rgb)
    pred = "tissue" if is_t else "background"
    ans = ANSWER_TISSUE if is_t else ANSWER_BACKGROUND
    preds.append({"id": f"{aid}.jpg", "answer": ans})
    ok = (pred == gt); correct += ok; n += 1
    print(f"{aid+'.jpg':12s}{gt:11s}{pred:11s}{'OK' if ok else 'XX':>4}")

print(f"\nclassifier accuracy on sample: {correct}/{n} = {correct/n:.3f}")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(preds, open(OUT, "w"), indent=2)
print(f"wrote {len(preds)} interf0 predictions -> {OUT}")
