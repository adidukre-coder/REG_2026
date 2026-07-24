"""Run the interf1 pipeline on a slice of the released Test-Phase-1 WSIs (test1/) to validate
stability + timing across ALL of them. Logs per-slide: time, organ, #steps, OK/FALLBACK/FAIL.
Usage: validate_all_wsis.py <chunk_idx> <n_chunks>"""
import sys, time, glob, traceback
sys.path.insert(0, "cot_artifacts")
from src.interf1_pipeline import predict_cot
CK="/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints/reg_attr_analysis.pt"
idx=int(sys.argv[1]); n=int(sys.argv[2])
slides=sorted(glob.glob("/nfs-stor/adinath.dukre/adinath/reg2026/test1/*.tiff"))
mine=slides[idx::n]    # strided split so each chunk has a mix of sizes
print(f"chunk {idx}/{n}: {len(mine)} slides", flush=True)
slow=[]; fail=[]
for s in mine:
    name=s.split("/")[-1]
    try:
        t0=time.time(); cot=predict_cot(s, CK, gpu=0); dt=time.time()-t0
        tag="OK"
        if dt*2.7 > 280: slow.append((name,dt)); tag="SLOW"
        print(f"  {tag} {name} {dt:.0f}s (T4~{dt*2.7:.0f}s) organ={cot[0]['answer']} steps={len(cot)}", flush=True)
    except Exception as e:
        fail.append((name,str(e))); print(f"  FAIL {name} {type(e).__name__}: {e}", flush=True)
print(f"\nchunk {idx} DONE: {len(mine)} slides, {len(slow)} slow(>280s T4 est), {len(fail)} failed", flush=True)
if slow: print("SLOW:", slow, flush=True)
if fail: print("FAILED:", fail, flush=True)
