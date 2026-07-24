"""
REG2026 feature-extraction driver — mini-shard work queue.

WHY: TRIDENT runs phase-by-phase per shard (all seg -> all coords -> all feat), so a big
shard produces NO features until its whole seg+coords finish (>3h) — useless under the 3h
GPU cap. This driver splits PENDING slides into small mini-shards (~30) and runs many in
parallel across the 4 GPUs, so each finishes fast and FEATURES STREAM OUT. Fully resumable:
it recomputes pending (skips slides that already have features) on every launch.

Run on a GPU node:  python src/extract_driver.py --shard_size 30 --parallel 12
"""
import os, sys, json, argparse, itertools, subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

WORK = "/nfs-stor/adinath.dukre/adinath/reg2026_work"
CONV = f"{WORK}/converted_slides"
FEAT_JOB = f"{WORK}/features"
FEATDIR = f"{FEAT_JOB}/20x_256px_0px_overlap/features_hoptimus0"
LAB = f"{WORK}/labels/train_labels.json"
TRIDENT = f"{WORK}/TRIDENT"
MINI = f"{WORK}/mini"; LOGS = f"{WORK}/logs"
# Always use the reg2026 env python for the TRIDENT subprocess (has openslide/timm/etc.),
# even if this driver is launched from (base).
PY = "/home/adinath.dukre/miniconda3/envs/reg2026/bin/python"
if not os.path.exists(PY):
    PY = sys.executable
os.makedirs(MINI, exist_ok=True); os.makedirs(LOGS, exist_ok=True)


def pending_slides():
    labels = {(r["id"] if str(r["id"]).endswith(".tiff") else r["id"] + ".tiff"): r["organ"]
              for r in json.load(open(LAB))}
    present = {f for f in os.listdir(CONV) if f.endswith(".tiff")}
    done = {f[:-3] for f in os.listdir(FEATDIR)} if os.path.isdir(FEATDIR) else set()  # stems (x.h5 -> x)
    byorg = defaultdict(list)
    for name, org in labels.items():
        if name in present and name[:-5] not in done:   # name x.tiff -> stem x
            byorg[org].append(name)
    for o in byorg:
        byorg[o].sort()
    return [x for tup in itertools.zip_longest(*byorg.values()) for x in tup if x]  # organ-interleaved


def run_shard(idx, names):
    # use whatever GPUs SLURM allocated (robust to gpu:1/2/4), round-robin across them
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    gpu = visible[idx % len(visible)]
    csv = f"{MINI}/s{idx:05d}.csv"
    with open(csv, "w") as f:
        f.write("wsi,mpp\n"); [f.write(f"{n},0.5\n") for n in names]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    with open(f"{LOGS}/mini_{idx:05d}.out", "w") as log:
        subprocess.run(
            [PY, "run_batch_of_slides.py", "--task", "all", "--wsi_dir", CONV,
             "--custom_list_of_wsis", csv, "--job_dir", FEAT_JOB,
             "--patch_encoder", "hoptimus0", "--mag", "20", "--patch_size", "256",
             "--gpus", "0", "--max_workers", "1", "--skip_errors"],  # 1 = valid + minimal forking
            env=env, cwd=TRIDENT, stdout=log, stderr=log)
    return idx, len(names)


def feat_count():
    return len(os.listdir(FEATDIR)) if os.path.isdir(FEATDIR) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_size", type=int, default=30)
    ap.add_argument("--parallel", type=int, default=12)
    a = ap.parse_args()
    pend = pending_slides()
    shards = [pend[i:i + a.shard_size] for i in range(0, len(pend), a.shard_size)]
    print(f"pending slides: {len(pend)} | mini-shards: {len(shards)} (size {a.shard_size}) "
          f"| parallel {a.parallel} | already have {feat_count()} features", flush=True)
    if not shards:
        print("nothing pending — all features extracted."); return
    done_shards = 0
    with ProcessPoolExecutor(max_workers=a.parallel) as ex:
        futs = {ex.submit(run_shard, i, s): i for i, s in enumerate(shards)}
        for f in as_completed(futs):
            i, n = f.result(); done_shards += 1
            print(f"[shard {done_shards}/{len(shards)}] done ({n} slides) | "
                  f"total features now: {feat_count()}", flush=True)
    print(f"FINISHED this run. Features: {feat_count()}")


if __name__ == "__main__":
    main()
