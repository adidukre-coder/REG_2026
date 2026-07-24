import sys, time, traceback
sys.path.insert(0, "cot_artifacts")
from src.interf1_pipeline import predict_cot
CK="/nfs-stor/adinath.dukre/adinath/reg2026_work/checkpoints/reg_attr_analysis.pt"
idx=int(sys.argv[1]); n=int(sys.argv[2])
slides=[l.strip() for l in open("/nfs-stor/adinath.dukre/adinath/reg2026_work/remaining_slides.txt") if l.strip()]
mine=slides[idx::n]
print(f"chunk {idx}/{n}: {len(mine)} slides", flush=True)
for s in mine:
    name=s.split("/")[-1]
    try:
        t0=time.time(); cot=predict_cot(s, CK, gpu=0); dt=time.time()-t0
        tag="SLOW" if dt*2.7>280 else "OK"
        print(f"  {tag} {name} {dt:.0f}s (T4~{dt*2.7:.0f}s) organ={cot[0]['answer']} steps={len(cot)}", flush=True)
    except Exception as e:
        print(f"  FAIL {name} {type(e).__name__}: {e}", flush=True)
print(f"chunk {idx} DONE", flush=True)
