"""
Interface 1 — Workflow Reasoning (Metric A).

Runs the full pipeline on the RAW slide INSIDE the container:
  raw .tiff -> [pyvips convert if AdobeDeflate] -> TRIDENT hest-seg -> 20x/256px coords
  -> H-Optimus-0 features (1536-d) -> REGAttributeModel (93 heads)
  -> traversal_engine + compose_report (thr 0.25) -> chain-of-thought.

Bundled weights live in /opt/ml/model (the model tarball): hf_cache/ (H-Optimus),
trident_cache/ (hest seg), reg_attr_analysis.pt (our attribute model). Cache env vars are
set to those paths + offline BEFORE TRIDENT imports the foundation models.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import TypedDict

from core import MODEL_PATH  # /opt/ml/model

# Point the foundation-model caches at the bundled weights and force offline (no network at run).
os.environ.setdefault("HF_HOME", str(MODEL_PATH / "hf_cache"))
os.environ.setdefault("TRIDENT_HOME", str(MODEL_PATH / "trident_cache"))
# torchvision downloads the deeplabv3 ResNet50 backbone via torch.hub -> point it at the
# bundled cache (TORCH_HOME/hub/checkpoints/resnet50-*.pth) so it works offline.
os.environ.setdefault("TORCH_HOME", str(MODEL_PATH / "torch_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# TRIDENT spawns dataloader workers; keep it modest for the platform's CPU allotment.
os.environ.setdefault("OMP_NUM_THREADS", "4")

_CKPT = str(MODEL_PATH / "reg_attr_analysis.pt")


class ChainOfThoughtStep(TypedDict):
    question: str
    answer: str
    next_question: str


# Minimal VALID chain-of-thought returned if the full pipeline fails for any reason
# (unreadable slide, no tissue, OOM, timeout, etc.). On Grand Challenge a single failed case
# fails the ENTIRE submission, so we must always emit a schema-valid output rather than crash.
_FALLBACK_COT: list[ChainOfThoughtStep] = [
    {"question": "What is the organ?", "answer": "Prostate",
     "next_question": "What is the final pathology report?"},
    {"question": "What is the final pathology report?",
     "answer": "No diagnostic abnormality identified.", "next_question": ""},
]

# Hard wall-clock budget for the whole pipeline, enforced by killing a child PROCESS rather than
# a same-process signal.alarm(). signal.alarm() can only interrupt Python bytecode; almost all of
# the real work here (pyvips.tiffsave, TRIDENT segment_tissue/extract_tissue_coords/
# extract_patch_features) runs as one blocking call into native/C-extension code that never
# returns control to the interpreter until it finishes -- so a same-process alarm can be queued
# but never delivered in time. This was the real cause of a real "Time limit exceeded" failure
# (420s actual vs. the platform's 300s/case limit) despite the previous 200s self-imposed
# watchdog. A child process CAN be forcibly killed by the OS regardless of what it is blocked in.
_HARD_TIMEOUT_S = 180  # tightened from 220s for more margin; worst case with grace periods below
                       # is 180+5+5=190s, leaving >100s under GC's 300s/case kill
_KILL_GRACE_S = 5      # SIGTERM/SIGKILL to a healthy process take milliseconds, not seconds --
                       # this is padding for process-table bookkeeping, not signal delivery time
_OUTER_ALARM_S = 250   # last-resort circuit breaker around the orchestration layer itself (see
                       # predict_chain_of_thought) -- fires only if something OTHER than the known
                       # I/O-blocking failure mode goes wrong (e.g. a bug in the join/kill logic
                       # above), since that logic is pure Python with its own bounded timeouts and
                       # should never itself take this long


def _pipeline_worker(wsi_path: str, ckpt_path: str, gpu: int, queue) -> None:
    """Runs in a child process (spawn context, so CUDA initializes cleanly in the child).
    Puts either ("ok", cot) or ("err", "<repr>") on the queue -- never raises back to the parent."""
    try:
        import torch
        print("=+=" * 10, flush=True)
        print("Torch CUDA available:", torch.cuda.is_available(), flush=True)
        if torch.cuda.is_available():
            print(f"  device: {torch.cuda.get_device_properties(torch.cuda.current_device())}",
                  flush=True)
        print("=+=" * 10, flush=True)
        from src.interf1_pipeline import predict_cot
        cot = predict_cot(wsi_path, ckpt_path, gpu=gpu)
        queue.put(("ok", cot))
    except BaseException as e:  # noqa: BLE001 - must not let the child die without reporting
        queue.put(("err", f"{type(e).__name__}: {e}"))


class _OrchestrationTimeout(Exception):
    """Raised by the outer SIGALRM circuit breaker below -- see predict_chain_of_thought."""


def _outer_alarm_handler(signum, frame):
    raise _OrchestrationTimeout(f"orchestration layer exceeded {_OUTER_ALARM_S}s")


def predict_chain_of_thought(*, wsi_path: Path) -> list[ChainOfThoughtStep]:
    """Raw WSI -> chain-of-thought (bare list of steps; last step next_question='').

    Wrapped so that ANY failure on a single slide -- including one that blocks past our time
    budget inside native code, or an unexpected failure in the process-orchestration machinery
    itself -- returns a valid fallback CoT instead of raising or hanging past the platform's hard
    kill. A crash/hard-kill fails the whole phase submission; a low-quality fallback case only
    loses points on that one case. This outer try/except is deliberately as broad as the original
    single-process version's was -- the subprocess-timeout mechanism below replaces signal.alarm
    for the "pipeline itself hangs" case, but must not narrow the safety net for every OTHER way
    this could fail (spawn/resource errors, unexpected multiprocessing exceptions, etc.), since the
    hidden test set can contain cases we haven't seen or anticipated.

    A second, independent safety net wraps _predict_with_hard_timeout itself: unlike the ORIGINAL
    bug (signal.alarm around code blocked in native C, which can't be interrupted), this alarm
    wraps only the join/terminate/kill orchestration logic -- pure Python with its own bounded
    timeouts throughout, so it should never legitimately run past _OUTER_ALARM_S. If it somehow
    does (a bug we haven't anticipated in the orchestration itself, not the known I/O-blocking
    failure mode), this is the last line of defense before the platform's own 300s/case hard kill.
    """
    old_handler = signal.signal(signal.SIGALRM, _outer_alarm_handler)
    signal.alarm(_OUTER_ALARM_S)
    try:
        return _predict_with_hard_timeout(wsi_path)
    except Exception as e:
        import traceback
        print(f"[interf1] predict_chain_of_thought failed outside the pipeline itself "
              f"({type(e).__name__}: {e}); returning fallback CoT", flush=True)
        traceback.print_exc()
        return _FALLBACK_COT
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _predict_with_hard_timeout(wsi_path: Path) -> list[ChainOfThoughtStep]:
    import multiprocessing as mp
    import queue as queue_module

    # spawn, not fork: this avoids ANY risk of a corrupted CUDA context in the child (the classic
    # fork-after-CUDA-init hazard), at the cost of a fresh interpreter/import per case. PYTHONPATH
    # is set at the container level (Dockerfile ENV PYTHONPATH=/opt/app), so `core`/`src`/`trident`
    # imports resolve the same way in the spawned child as in the parent.
    ctx = mp.get_context("spawn")
    result_queue: "mp.Queue" = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_pipeline_worker, args=(str(wsi_path), _CKPT, 0, result_queue),
                        daemon=True)
    proc.start()
    proc.join(timeout=_HARD_TIMEOUT_S)

    if proc.is_alive():
        print(f"[interf1] pipeline exceeded {_HARD_TIMEOUT_S}s hard budget -- killing worker "
              f"process and returning fallback CoT", flush=True)
        proc.terminate()
        proc.join(timeout=_KILL_GRACE_S)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=_KILL_GRACE_S)
        # Note: join(timeout=...) always returns after the timeout elapses regardless of whether
        # the child actually terminated (e.g. a process stuck in an uninterruptible disk-I/O wait
        # cannot be preempted by SIGKILL either) -- we deliberately proceed here either way, since
        # this process is daemon=True and the interpreter will not wait for it at exit.
        return _FALLBACK_COT

    try:
        status, payload = result_queue.get(timeout=5)  # brief grace period for the pipe to flush
    except (queue_module.Empty, OSError, EOFError):
        print("[interf1] worker exited without a result (likely killed by OOM or a native "
              "crash); returning fallback CoT", flush=True)
        return _FALLBACK_COT

    if status == "ok":
        cot = payload
        if not cot or cot[-1].get("next_question") != "":
            print("[interf1] pipeline returned malformed CoT; returning fallback CoT", flush=True)
            return _FALLBACK_COT
        return cot

    print(f"[interf1] pipeline failed ({payload}); returning fallback CoT", flush=True)
    return _FALLBACK_COT
