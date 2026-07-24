"""
Convert OpenSlide-UNREADABLE slides (AdobeDeflate-compressed, ~959 prostate + others)
into OpenSlide-readable, tiled, PYRAMIDAL, JPEG-compressed TIFFs via pyvips (streaming).
This both fixes readability AND makes them pyramidal (fast segmentation).

CPU-only (pyvips) -> runs on the login node. Output keeps the same filename so labels match.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyvips

SRC = "/nfs-stor/adinath.dukre/adinath/reg2026/train"
DST = "/nfs-stor/adinath.dukre/adinath/reg2026_work/converted_slides"
BAD = "/nfs-stor/adinath.dukre/adinath/reg2026_work/openslide_unreadable.json"
os.makedirs(DST, exist_ok=True)


def convert(name):
    out = os.path.join(DST, name)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return (name, "skip", 0)
    try:
        t = time.time()
        img = pyvips.Image.new_from_file(os.path.join(SRC, name), access="sequential")
        img.tiffsave(out, tile=True, tile_width=256, tile_height=256,
                     pyramid=True, compression="jpeg", Q=90, bigtiff=True)
        return (name, "ok", time.time() - t)
    except Exception as e:
        if os.path.exists(out):
            os.remove(out)  # don't leave partial files
        return (name, f"ERR:{type(e).__name__}:{str(e)[:60]}", 0)


def main():
    names = json.load(open(BAD))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(names)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    names = names[:limit]
    print(f"converting {len(names)} slides with {workers} workers -> {DST}")
    ok = skip = err = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(convert, n): n for n in names}
        for i, f in enumerate(as_completed(futs), 1):
            name, status, dt = f.result()
            if status == "ok": ok += 1
            elif status == "skip": skip += 1
            else: err += 1; print(f"  {name}: {status}")
            if i % 50 == 0 or i == len(names):
                print(f"  [{i}/{len(names)}] ok={ok} skip={skip} err={err} "
                      f"elapsed={time.time()-t0:.0f}s")
    print(f"DONE: ok={ok} skip={skip} err={err} in {time.time()-t0:.0f}s -> {DST}")


if __name__ == "__main__":
    main()
