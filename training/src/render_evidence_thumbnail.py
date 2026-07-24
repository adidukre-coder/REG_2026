"""
Render a WSI thumbnail with per-question ABMIL attention evidence overlaid, for a handful
of illustrative questions on one slide -- the visual proof that per_task_attention=True
gives genuinely distinct, question-appropriate spatial grounding (not just a shared
attention map reused for every question).

Usage: python src/render_evidence_thumbnail.py <stem> <out_png>
"""
import json, sys
import openslide
from PIL import Image, ImageDraw, ImageFont

STEM = sys.argv[1]
OUT_PNG = sys.argv[2]
THUMB_MAX = 1000

WSI = f"/nfs-stor/adinath.dukre/adinath/reg2026_work/converted_slides/{STEM}.tiff"
PAYLOAD = f"/nfs-stor/adinath.dukre/adinath/reg2026_work/evidence_demo_pertask/{STEM}.json"
PATCH_SIZE_L0 = 256  # patch size in level-0 px (MAG=20, forced mpp=0.5 -> level0 == 20x)

QUESTIONS_TO_SHOW = [
    "What is the organ?",
    "What is the #1 diagnosis?",
    "Is there any invasion present?",
    "What is the overall score?",
]
COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]  # red, blue, green, orange

d = json.load(open(PAYLOAD))
qmap = {q["question"]: q for q in d["questions"]}

s = openslide.OpenSlide(WSI)
w0, h0 = s.level_dimensions[0]
thumb = s.get_thumbnail((THUMB_MAX, THUMB_MAX)).convert("RGB")
scale = thumb.width / w0
draw = ImageDraw.Draw(thumb, "RGBA")

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
except Exception:
    font = ImageFont.load_default()

legend = []
for qi, qname in enumerate(QUESTIONS_TO_SHOW):
    q = qmap.get(qname)
    if not q:
        continue
    color = COLORS[qi % len(COLORS)]
    for ev in q["evidence"][:3]:
        x0 = ev["x"] * scale; y0 = ev["y"] * scale
        x1 = x0 + PATCH_SIZE_L0 * scale; y1 = y0 + PATCH_SIZE_L0 * scale
        alpha = max(40, int(ev["weight"] * 255))
        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        draw.rectangle([x0, y0, x1, y1], outline=rgb + (255,), width=2)
    conf = "high" if q["attn_entropy_norm"] < 0.3 else ("med" if q["attn_entropy_norm"] < 0.55 else "low")
    correctness = {True: "correct", False: "WRONG", None: "n/a"}[q["correct"]]
    legend.append((color, qname, q["answer"], conf, q["attn_entropy_norm"], correctness))

# legend panel below the thumbnail
pad = 10
line_h = 20
panel_h = pad * 2 + line_h * len(legend)
canvas = Image.new("RGB", (thumb.width, thumb.height + panel_h), "white")
canvas.paste(thumb, (0, 0))
pd = ImageDraw.Draw(canvas)
y = thumb.height + pad
for color, qname, ans, conf, ent, correctness in legend:
    rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
    pd.rectangle([pad, y + 3, pad + 14, y + 17], fill=rgb)
    tag = f"[{correctness}, conf={conf} ent={ent:.2f}]"
    pd.text((pad + 20, y), f"{qname}  ->  {ans[:40]}   {tag}", fill="black", font=font)
    y += line_h

canvas.save(OUT_PNG)
print(f"saved {OUT_PNG}  ({canvas.width}x{canvas.height})")
