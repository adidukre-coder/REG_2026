"""
PROOF STEP 1 — timing stress test: run the FULL per-case pipeline (raw slide -> pyramidal
convert -> hest seg -> coords -> H-Optimus features -> CoT) with the hest config on the
WORST-CASE slides, to prove hest fits the T4 5-min/case budget now that we always
pyramidal-convert. Reports total wall time + T4 estimate (A100 x2.7).

Run via sbatch with REG_SEGMENTER=hest REG_SEG_WORKERS=0 REG_MAX_PATCHES=2000 and the offline
caches set. Usage: python src/diag_hest_timing.py <slide1> <slide2> ...
"""
import os, sys, time, traceback
sys.path.insert(0, "cot_artifacts")
from src.interf1_pipeline import predict_cot, SEGMENTER, SEG_WORKERS, MAX_PATCHES

CK = "/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints/reg_attr_analysis.pt"
T4 = 2.7   # measured A100->T4 slowdown factor on this pipeline
LIMIT = 300

print(f"CONFIG: SEGMENTER={SEGMENTER} SEG_WORKERS={SEG_WORKERS} MAX_PATCHES={MAX_PATCHES}", flush=True)
worst = 0.0; nfail = 0
for sp in sys.argv[1:]:
    nm = os.path.basename(sp)
    sz = os.path.getsize(sp)/1e9 if os.path.exists(sp) else -1
    t0 = time.time()
    try:
        cot = predict_cot(sp, CK, gpu=0)
        dt = time.time()-t0; t4 = dt*T4; worst = max(worst, t4)
        flag = "OK    " if t4 < 280 else "SLOW!!"
        print(f"{flag} {nm} {sz:.1f}GB | {dt:.0f}s A100 -> T4~{t4:.0f}s | organ={cot[0]['answer']} steps={len(cot)}", flush=True)
    except Exception as e:
        nfail += 1
        print(f"FAIL  {nm} {sz:.1f}GB | {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
print(f"\n==== WORST T4-est = {worst:.0f}s (limit {LIMIT}s); failures={nfail} ====", flush=True)
print("VERDICT:", "PASS (fits with margin)" if (worst < 250 and nfail == 0) else
      ("TIGHT (fits but <50s margin)" if (worst < LIMIT and nfail == 0) else "FAIL (too slow or crashed)"), flush=True)
