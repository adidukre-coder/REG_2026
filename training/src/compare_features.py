"""
Compare H-Optimus features from ORIGINAL (non-pyramidal) vs CONVERTED (pyramidal JPEG Q90)
versions of the same slides, at matching patch coordinates. High cosine similarity => the
convert-to-pyramidal step did NOT degrade features (validates the convert-all decision).
"""
import sys, h5py, numpy as np

ORIG = sys.argv[1]   # job_dir for originals
CONV = sys.argv[2]   # job_dir for converted
SLIDES = sys.argv[3:]  # slide stems (no extension)

def load(job_dir, stem):
    p = f"{job_dir}/20x_256px_0px_overlap/features_hoptimus0/{stem}.h5"
    with h5py.File(p, "r") as h:
        return np.asarray(h["coords"]), np.asarray(h["features"], dtype=np.float32)

def cos(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

print(f"{'slide':22s} {'common/orig/conv patches':28s} {'mean cos':>9} {'min cos':>9}")
all_means = []
for s in SLIDES:
    co, fo = load(ORIG, s)
    cc, fc = load(CONV, s)
    do = {tuple(int(x) for x in c): fo[i] for i, c in enumerate(co)}
    dc = {tuple(int(x) for x in c): fc[i] for i, c in enumerate(cc)}
    common = sorted(set(do) & set(dc))
    if not common:
        print(f"{s:22s} NO COMMON COORDS (seg masks differ) — comparing mean vectors instead")
        m = cos(fo.mean(0), fc.mean(0))
        print(f"   mean-vector cosine = {m:.4f}")
        all_means.append(m); continue
    sims = np.array([cos(do[c], dc[c]) for c in common])
    print(f"{s:22s} {len(common):>6}/{len(do):>5}/{len(dc):>5}{'':9} {sims.mean():>9.4f} {sims.min():>9.4f}")
    all_means.append(float(sims.mean()))

print(f"\nOVERALL mean cosine similarity: {np.mean(all_means):.4f}")
print(">= 0.99  -> conversion is harmless, convert-all VALIDATED")
print("0.95-0.99-> minor shift, acceptable")
print("< 0.95   -> meaningful degradation, reconsider (raise Q / keep originals)")
