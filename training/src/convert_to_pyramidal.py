"""
Convert the READABLE original slides (JPEG, non-pyramidal) into tiled PYRAMIDAL JPEG TIFFs,
so segmentation is fast (~5x) and the whole GPU extraction fits in short (3h) chunks.
CPU-only (pyvips) -> runs on the login node with no time limit. Writes to the SAME
converted_slides/ dir (same filenames) so ALL 11,209 slides end up there, all pyramidal.

Usage: python src/convert_to_pyramidal.py [limit] [workers]
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyvips

SRC = "/nfs-stor/adinath.dukre/adinath/reg2026/train"
DST = "/nfs-stor/adinath.dukre/adinath/reg2026_work/converted_slides"
CSV = "/nfs-stor/adinath.dukre/adinath/reg2026_work/train_readable.csv"
os.makedirs(DST, exist_ok=True)


def convert(name):
    out = os.path.join(DST, name)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return "skip"
    try:
        img = pyvips.Image.new_from_file(os.path.join(SRC, name), access="sequential")
        img.tiffsave(out, tile=True, tile_width=256, tile_height=256,
                     pyramid=True, compression="jpeg", Q=90, bigtiff=True)
        return "ok"
    except Exception as e:
        if os.path.exists(out):
            os.remove(out)
        return f"ERR:{type(e).__name__}:{str(e)[:50]}"


def main():
    names = [l.split(",")[0] for l in open(CSV).read().splitlines()[1:]]
    if len(sys.argv) > 1:
        names = names[:int(sys.argv[1])]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    print(f"converting {len(names)} readable originals -> pyramidal ({workers} workers)")
    c = {"ok": 0, "skip": 0, "err": 0}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(convert, n): n for n in names}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            c["ok" if r == "ok" else "skip" if r == "skip" else "err"] += 1
            if r.startswith("ERR"):
                print(f"  {futs[f]}: {r}")
            if i % 200 == 0 or i == len(names):
                print(f"  [{i}/{len(names)}] ok={c['ok']} skip={c['skip']} err={c['err']} "
                      f"{time.time()-t0:.0f}s")
    print(f"DONE ok={c['ok']} skip={c['skip']} err={c['err']} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
