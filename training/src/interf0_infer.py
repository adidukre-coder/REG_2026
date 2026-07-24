"""
Interface-0 (Metric B) inference: tissue-vs-background detection on a 256px ROI.

The challenge question is always "Is tissue visible in this ROI?". Metric B =
  0.3*B1 (background ROIs must be rejected: answer says no-tissue/not-assessable)
+ 0.3*B2 (a tissue ROI and its perturbed copy must get the SAME answer)
+ 0.4*B3 (paired tissue regions must get the SAME answer).

Strategy: a robust classical tissue detector (H&E tissue is colored/saturated; background
is blank white) + a FIXED canonical answer per class. Identical tissue answers make B2/B3
automatically 1.0; correct background rejection drives B1. No training, no GPU, fully
deterministic -> maximally consistent (which is exactly what B2/B3 reward).
"""
from __future__ import annotations
import numpy as np
import cv2

# Fixed canonical answers (identical string for every tissue ROI -> B2/B3 = SAME = 1.0).
ANSWER_TISSUE = "Yes, tissue is visible in this region."
ANSWER_BACKGROUND = ("No, there is no tissue visible in this region; it is background only "
                     "and cannot be assessed.")

# Tissue if enough colored (saturated, non-white) content. Background ROIs measure ~0 on
# both, tissue ~0.6-0.96 -> a low threshold separates with a huge margin and is robust to
# the perturbed variants.
SAT_FRAC_MIN = 0.05      # fraction of pixels with HSV saturation > SAT_LEVEL
SAT_LEVEL = 0.10
NONWHITE_MIN = 0.08      # fraction of non-near-white pixels


def roi_is_tissue(img_rgb: np.ndarray) -> bool:
    """img_rgb: HxWx3 uint8 RGB -> True if tissue is present."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1] / 255.0
    sat_frac = float((sat > SAT_LEVEL).mean())
    nonwhite = 1.0 - float((img_rgb > 200).all(axis=2).mean())
    return (sat_frac > SAT_FRAC_MIN) or (nonwhite > NONWHITE_MIN)


def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)                       # BGR
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def predict_visual_context_response(image_path_or_rgb, question: str | None = None) -> str:
    """The Interface-0 submission function: ROI -> brief tissue/background answer."""
    img = load_rgb(image_path_or_rgb) if isinstance(image_path_or_rgb, str) else image_path_or_rgb
    return ANSWER_TISSUE if roi_is_tissue(img) else ANSWER_BACKGROUND
