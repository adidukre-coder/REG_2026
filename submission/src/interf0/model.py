"""
Interface 0 — Visual Grounding (Metric B): tissue-vs-background detection on a 256px ROI.

The question is always "Is tissue visible in this ROI?". Metric B rewards:
  B1 (0.3) background ROIs answered as no-tissue / not-assessable
  B2 (0.3) a tissue ROI and its perturbed copy get the SAME answer
  B3 (0.4) paired tissue regions get the SAME answer

Strategy: a robust classical tissue detector (H&E tissue is colored/saturated; background is
blank white) + a FIXED canonical answer per class. Identical tissue answers make B2/B3 = 1.0;
correct background rejection drives B1. Pure PIL+numpy (no cv2/torch), deterministic, no weights
-> maximally consistent, which is exactly what B2/B3 reward. Scored 1.0/1.0/1.0 on the local sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core import load_json_file, load_roi_image

ANSWER_TISSUE = "Yes, tissue is visible in this region."
ANSWER_BACKGROUND = (
    "No, there is no tissue visible in this region; it is background only "
    "and cannot be assessed."
)

# Tissue if enough colored (saturated) or non-white content. Background ROIs measure ~0 on
# both; tissue ~0.6-0.96 -> low thresholds separate with a huge margin and stay stable under
# the perturbed variants (so the original/perturbed answers stay identical).
_SAT_LEVEL = 0.10
_SAT_FRAC_MIN = 0.05
_NONWHITE_MIN = 0.08


def _roi_is_tissue(roi_image) -> bool:
    """roi_image: PIL RGB image -> True if tissue is present."""
    arr = np.asarray(roi_image.convert("RGB")).astype(np.float32)  # HxWx3, 0-255
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)  # HSV S channel, 0-1
    sat_frac = float((sat > _SAT_LEVEL).mean())
    nonwhite = 1.0 - float((arr > 200).all(axis=2).mean())
    return (sat_frac > _SAT_FRAC_MIN) or (nonwhite > _NONWHITE_MIN)


def predict_visual_context_response(
    *,
    question_path: Path,
    roi_image_path: Path,
) -> str:
    """ROI -> brief tissue/background answer (plain string).

    Wrapped so ANY failure (odd image format, corrupt file, etc.) returns a valid string
    instead of raising — on Grand Challenge a single failed case fails the whole submission.
    """
    try:
        roi_image = load_roi_image(location=roi_image_path)
        return ANSWER_TISSUE if _roi_is_tissue(roi_image) else ANSWER_BACKGROUND
    except Exception as e:
        import traceback
        print(f"[interf0] failed ({type(e).__name__}: {e}); returning safe fallback", flush=True)
        traceback.print_exc()
        # Safe, non-committal answer that won't be penalized as a wrong tissue claim.
        return "The region cannot be reliably assessed."
