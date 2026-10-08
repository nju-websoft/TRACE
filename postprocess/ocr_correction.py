#!/usr/bin/env python3
"""
OCR-vocabulary edit-distance correction for flowchart triplets.

Core algorithm (no I/O orchestration; see correct_loo.py for the LOO driver):
  1. PaddleOCR the source image → set of recognized text tokens (vocab)
  2. For each predicted triplet's source/end/condition, find vocab entry within
     Levenshtein distance ≤ max_dist; if a unique closest match exists, replace.
  3. Reserved relation tokens (connectedTo / partOf / connected_with) are kept verbatim.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import Levenshtein


RESERVED_RELATIONS = {"connectedTo", "partOf", "connected_with"}
SUPPORTED_IMG_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp")


# ---------------------------------------------------------------------------
# Garbled-token detector (English word-frequency based)
# ---------------------------------------------------------------------------
import re as _re


def looks_garbled(value: str, min_token_len: int = 3, freq_thr: float = 5e-7) -> bool:
    """
    Decide whether a string contains an alphabetic span that "does not look like English" and should be replaced.
    True  -> at least one alphabetic token of length >= min_token_len with word frequency < freq_thr (likely an OCR typo)
    False -> all alphabetic tokens are real English words, or all symbols/single-letter vars (no evidence, keep original)

    Uses the wordfreq library; if unavailable, falls back to False (never blocks).

    Suitable for FCA/FCB: handwritten English + math + short tokens.
    Not suitable for FlowLearn, whose node names are rare coined words (rare words would be misjudged as garbled).
    """
    if not value:
        return False
    try:
        from wordfreq import word_frequency
    except ImportError:
        return False
    tokens = _re.findall(r"[A-Za-z]{" + str(min_token_len) + r",}", str(value))
    if not tokens:
        return False
    for tok in tokens:
        if word_frequency(tok.lower(), "en") < freq_thr:
            return True
    return False


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
PADDLE_PRESETS = {
    # english mobile: good on plain English text ("Input V", "True")
    "en_mobile": dict(lang="en", ocr_version="PP-OCRv5"),
    # ch server: good on handwritten math ("R=R*n", "n=1"), but occasionally drops chars in plain English
    "ch_server": dict(lang="ch", ocr_version="PP-OCRv5"),
}


def build_paddle_ocr(preset: str = "en_mobile", **override):
    """Lazy-import + construct PaddleOCR. preset in PADDLE_PRESETS keys, or custom kwargs."""
    from paddleocr import PaddleOCR
    cfg = dict(PADDLE_PRESETS.get(preset, PADDLE_PRESETS["en_mobile"]))
    cfg.update(override)
    return PaddleOCR(use_doc_unwarping=False, **cfg)


def build_paddle_ocrs(presets: Iterable[str] = ("en_mobile",)):
    """Construct several PaddleOCR instances at once. Returns list[(name, model)]."""
    return [(p, build_paddle_ocr(p)) for p in presets]


def _parse_ocr_result(result, min_score: float = 0.0) -> List[Tuple[str, float]]:
    """Normalize the PaddleOCR result -> list[(text, score)], filtered by score >= min_score."""
    out: List[Tuple[str, float]] = []
    for page in result or []:
        if hasattr(page, "get"):
            rec_texts = page.get("rec_texts") or []
            rec_scores = page.get("rec_scores") or [1.0] * len(rec_texts)
            for t, s in zip(rec_texts, rec_scores):
                t = (t or "").strip()
                if t and s >= min_score:
                    out.append((t, float(s)))
            continue
        for item in page or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            v = item[1]
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                txt, score = v[0], float(v[1])
            else:
                txt, score = v, 1.0
            txt = (txt or "").strip()
            if txt and score >= min_score:
                out.append((txt, score))
    return out


def _run_one_ocr(ocr_model, img_cv2, min_score: float):
    try:
        result = ocr_model.predict(img_cv2)
    except Exception as e:
        print(f"  [ocr] predict fail: {e}")
        return []
    return _parse_ocr_result(result, min_score=min_score)


def extract_ocr_vocab(image_path: str, ocr_model, min_score: float = 0.0) -> List[str]:
    """Single-model OCR vocab (kept for backward compatibility)."""
    import cv2
    import numpy as np
    from PIL import Image

    try:
        img = Image.open(image_path).convert("RGB")
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  [ocr] load fail {image_path}: {e}")
        return []

    pairs = _run_one_ocr(ocr_model, img_cv2, min_score)
    return sorted({t for t, _ in pairs})


def extract_ocr_vocab_union(image_path: str, ocr_models, min_score: float = 0.0) -> List[str]:
    """Union of multiple models' OCR vocab (e.g. en_mobile + ch_server complementing each other)."""
    import cv2
    import numpy as np
    from PIL import Image

    try:
        img = Image.open(image_path).convert("RGB")
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  [ocr] load fail {image_path}: {e}")
        return []

    vocab = set()
    for _name, m in ocr_models:
        for t, _s in _run_one_ocr(m, img_cv2, min_score):
            vocab.add(t)
    return sorted(vocab)


# ---------------------------------------------------------------------------
# Edit-distance matching
# ---------------------------------------------------------------------------
def closest_vocab_match(value: str, vocab: Iterable[str], max_dist: int = 2) -> Optional[str]:
    """
    Return the unique closest vocab entry within max_dist (case-insensitive Levenshtein).
    Tie ⇒ return None (don't pick arbitrarily).
    """
    if not value:
        return None
    target = value.strip().lower()
    if not target:
        return None

    best: List[Tuple[int, str]] = []
    best_d = max_dist + 1
    for w in vocab:
        d = Levenshtein.distance(target, w.lower())
        if d < best_d:
            best_d, best = d, [(d, w)]
        elif d == best_d:
            best.append((d, w))
    if best_d > max_dist or not best:
        return None
    if len(best) > 1:
        return None
    return best[0][1]


def correct_value(value, vocab, max_dist: int = 2, min_len: int = 0, garbled_only: bool = False):
    """Return corrected value if a unique close match exists, else original.

    Guards:
      - min_len: strings shorter than min_len are left unchanged (avoids mangling Y/N/i<m)
      - skip if equal ignoring case (avoids OCR case-noise like Begin->begin)
      - garbled_only: replace only when the value does not look like an English word (via wordfreq)
    """
    if value is None:
        return value
    s = str(value).strip()
    if not s or len(s) < min_len:
        return value
    if garbled_only and not looks_garbled(s):
        return value
    match = closest_vocab_match(s, vocab, max_dist)
    if match is None:
        return value
    if match.lower() == s.lower():
        return value
    return match


def correct_relation(value, vocab, max_dist: int = 2, min_len: int = 0, garbled_only: bool = False):
    """Same as correct_value but skip reserved relation tokens."""
    if value is None or str(value).strip() in RESERVED_RELATIONS:
        return value
    return correct_value(value, vocab, max_dist, min_len=min_len, garbled_only=garbled_only)


# ---------------------------------------------------------------------------
# Triplet-level
# ---------------------------------------------------------------------------
def correct_triplets(
    triplets: List[dict],
    vocab: List[str],
    max_dist: int = 2,
    skip_condition: bool = False,
    min_len: int = 0,
    garbled_only: bool = False,
) -> Tuple[List[dict], int]:
    """
    Return (corrected_triplets, num_fields_changed).
    Preserves all non-text fields (arrow_id, arrow_label, arrow_bbox, raw_vqa_output, ...).
    """
    out = []
    n_changed = 0
    for t in triplets:
        new_t = dict(t)
        for fname in ("source_node", "target_node", "source", "end"):
            if fname in new_t:
                fixed = correct_value(new_t[fname], vocab, max_dist,
                                      min_len=min_len, garbled_only=garbled_only)
                if fixed != new_t[fname]:
                    n_changed += 1
                new_t[fname] = fixed
        if "condition" in new_t and not skip_condition:
            fixed = correct_relation(new_t["condition"], vocab, max_dist,
                                     min_len=min_len, garbled_only=garbled_only)
            if fixed != new_t["condition"]:
                n_changed += 1
            new_t["condition"] = fixed
        out.append(new_t)
    return out, n_changed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_image(stem: str, image_dir: str) -> Optional[Path]:
    """Locate <image_dir>/<stem><ext> across common extensions."""
    image_dir = Path(image_dir)
    direct = image_dir / stem
    if direct.exists() and direct.is_file():
        return direct
    for ext in SUPPORTED_IMG_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
        p = image_dir / f"{Path(stem).stem}{ext}"
        if p.exists():
            return p
    return None
