"""
Model loading, canvas segmentation and preprocessing for the EMNIST letter demo.

Kept separate from the Gradio UI so it can be imported and tested on its own:

    from recognizer import Recognizer, MODELS
    rec = Recognizer()
    result = rec.recognize(canvas_rgba, model_key="densenet", mode="word")
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
REPO_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelInfo:
    key: str
    label: str
    framework: str          # "keras" or "torch"
    path: Path
    input_size: int         # side of the image we build before model-specific transforms
    test_accuracy: str      # as reported in the training notebook (855-image test split)
    notes: str = ""


MODELS: dict[str, ModelInfo] = {
    "densenet": ModelInfo(
        key="densenet",
        label="DenseNet (fine-tuned, Keras)",
        framework="keras",
        path=REPO_DIR / "models/DenseNet201/DenseNet201_Finetuned.keras",
        input_size=64,
        test_accuracy="88.9%",
        notes="File is named DenseNet201, but the saved network is DenseNet121 "
              "(ImageNet weights, last 25 layers fine-tuned). Input 64x64x3 in [0, 1].",
    ),
    "mobilenet_transfer": ModelInfo(
        key="mobilenet_transfer",
        label="MobileNetV2 – transfer learning (PyTorch)",
        framework="torch",
        path=REPO_DIR / "models/MobileNet/checkpoints/mobilenetv2_best.pth",
        input_size=28,
        test_accuracy="80.7%",
        notes="ImageNet features frozen, new 26-class head. 28x28 -> 224x224 with ImageNet normalization.",
    ),
    "mobilenet_scratch": ModelInfo(
        key="mobilenet_scratch",
        label="MobileNetV2 – trained from scratch (PyTorch)",
        framework="torch",
        path=REPO_DIR / "models/MobileNet/checkpoints/mobilenetv2_scratch_best.pth",
        input_size=28,
        test_accuracy="90.3%",
        notes="Same architecture, random init, all layers trained. Same input pipeline as the transfer model.",
    ),
}


# ---------------------------------------------------------------------------
# Preprocessing (ported from EMNIST_Capstone_DenseNet_v4.ipynb, preprocess_character)
# ---------------------------------------------------------------------------

# Measured on held-out EMNIST letters (28x28): the letter's longest side is ~23-24 px,
# it is centred by its bounding box, and strokes are ~3.9 px wide.
EMNIST_SIZE = 28
EMNIST_BOX = 24
EMNIST_STROKE = 3.6


def _stroke_width(binary: np.ndarray) -> float:
    """Approximate pen width in pixels: 2 x mean distance-to-edge along stroke centres."""
    # Pad with background first: a crop that is all ink (e.g. a straight "I") has no edge to measure to
    padded = cv2.copyMakeBorder(binary.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dt = cv2.distanceTransform(padded, cv2.DIST_L2, 3)
    ridge = dt[(dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt > 0)]
    return float(2 * ridge.mean()) if ridge.size else 1.0


def to_emnist(char_img: np.ndarray, ink_threshold: float = 0.25) -> np.ndarray:
    """
    White-ink-on-black crop (H, W) in [0, 1]  ->  28x28 EMNIST-style image in [0, 1].

    1. Re-draw the strokes at a width that will come out at ~3.6 px after shrinking,
       so a thin brush, a thick brush, a big letter and a small letter all look alike.
    2. Shrink so the longest side is 24 px (aspect ratio kept), centre in a 28x28 frame.
    3. Soften edges slightly and scale so the brightest ink is 1.0.
    """
    img = np.asarray(char_img, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    binary = (img > ink_threshold).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros((EMNIST_SIZE, EMNIST_SIZE), np.float32)

    ys, xs = np.nonzero(binary)
    binary = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = binary.shape
    scale = EMNIST_BOX / max(h, w)

    # Normalise stroke width in the original resolution
    target = EMNIST_STROKE / scale
    current = _stroke_width(binary)
    pad = int(np.ceil(target)) + 2
    binary = cv2.copyMakeBorder(binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    diff = target - current
    if diff >= 1:
        k = int(round(diff)) | 1
        binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    elif diff <= -2:
        k = int(round(min(-diff, current - 2))) | 1      # never erode a stroke away
        if k >= 3:
            eroded = cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            # keep the letter's shape: skip if erosion breaks it apart or removes too much
            n_before = cv2.connectedComponents(binary)[0]
            if eroded.sum() > 0.25 * binary.sum() and cv2.connectedComponents(eroded)[0] <= n_before:
                binary = eroded

    ys, xs = np.nonzero(binary)
    binary = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    h, w = binary.shape
    scale = EMNIST_BOX / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = cv2.resize(binary, (nw, nh), interpolation=cv2.INTER_AREA)

    out = np.zeros((EMNIST_SIZE, EMNIST_SIZE), np.float32)
    y0, x0 = (EMNIST_SIZE - nh) // 2, (EMNIST_SIZE - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    out = cv2.GaussianBlur(out, (3, 3), 0.6)
    if out.max() > 0:
        out /= out.max()
    return out


def canvas_to_ink(canvas: np.ndarray) -> np.ndarray:
    """
    Gradio sketchpad composite (H, W, 3|4) uint8  ->  ink map (H, W) float in [0, 1],
    1 = ink. Works whether the background is white or transparent.
    """
    arr = np.asarray(canvas)
    if arr.ndim == 2:
        gray = arr.astype(np.float32) / 255.0
        return 1.0 - gray
    rgb = arr[..., :3].astype(np.float32) / 255.0
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if arr.shape[-1] == 4:
        alpha = arr[..., 3].astype(np.float32) / 255.0
    else:
        alpha = np.ones_like(lum)
    return np.clip(alpha * (1.0 - lum), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

@dataclass
class Glyph:
    box: tuple[int, int, int, int]      # x, y, w, h on the canvas
    crop: np.ndarray                    # ink crop, other glyphs masked out
    row: int = 0
    space_before: bool = False


def _union(a: list[int], b: list[int]) -> list[int]:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return [x0, y0, x1 - x0, y1 - y0]


def _merge_in_row(items: list[tuple[list[int], list[int]]]) -> list[tuple[list[int], list[int]]]:
    """
    Within ONE text line, merge pieces that sit on top of each other: the dot of an i/j,
    a crossbar drawn as a separate stroke, a letter broken by a faint stroke.
    (Doing this per line is what keeps letters from different lines apart.)
    """
    items = sorted(items, key=lambda it: it[0][0])
    out: list[tuple[list[int], list[int]]] = []
    for box, labs in items:
        if out:
            pb, pl = out[-1]
            overlap = min(pb[0] + pb[2], box[0] + box[2]) - max(pb[0], box[0])
            if overlap > 0 and overlap >= 0.5 * min(pb[2], box[2]):
                out[-1] = (_union(pb, box), pl + labs)
                continue
        out.append((box, list(labs)))
    return out


def _split_wide(box, mask, ref_w, row_h, stroke):
    """
    Split a piece that is clearly several touching letters (e.g. a double T whose crossbars
    join) at its thinnest columns. Conservative on purpose so M and W stay whole.
    Returns a list of (box, sub-mask).
    """
    x, y, w, h = box
    if not (w > 1.8 * ref_w and w > 1.35 * row_h):
        return [(box, mask)]
    k = max(2, int(round(w / ref_w)))
    cols = mask.sum(axis=0).astype(float)
    cuts = []
    for i in range(1, k):
        ideal = int(i * w / k)
        lo, hi = max(1, ideal - int(0.3 * w / k)), min(w - 1, ideal + int(0.3 * w / k))
        if hi <= lo:
            continue
        c = lo + int(np.argmin(cols[lo:hi]))
        if cols[c] <= 1.6 * stroke:            # only cut through a thin join
            cuts.append(c)
    if not cuts:
        return [(box, mask)]
    pieces, start = [], 0
    for c in cuts + [w]:
        sub = np.zeros_like(mask)
        sub[:, start:c] = mask[:, start:c]
        ys, xs = np.nonzero(sub)
        if len(xs):
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            pieces.append(([x + x0, y + y0, x1 - x0, y1 - y0], sub[y0:y1, x0:x1]))
        start = c
    return pieces


def segment(ink: np.ndarray, mode: str = "word", ink_threshold: float = 0.25) -> list[Glyph]:
    """
    Split the drawing into letters.

    mode="letter": everything on the canvas is one letter.
    mode="word":   find the text lines first, then the letters in each line, read
                   left-to-right, top-to-bottom. Wide gaps become spaces, lines become new lines.
    """
    binary = (ink > ink_threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []

    if mode == "letter":
        ys, xs = np.nonzero(binary)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        return [Glyph(box=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)), crop=ink[y0:y1, x0:x1].copy())]

    # Close tiny gaps inside strokes (shaky mouse, faint anti-aliased pen); scale with pen width
    stroke = _stroke_width(binary)
    k = 3 if stroke < 4 else 5
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)

    min_area = max(8, 1.5 * stroke * stroke)
    comps = [([int(v) for v in stats[i][:4]], [i]) for i in range(1, n) if stats[i][4] >= min_area]
    if not comps:
        return []

    # 1. Text lines: cluster the main pieces by vertical centre
    sizes = np.array([max(b[2], b[3]) for b, _ in comps])
    big = sizes >= 0.4 * np.median(sizes)
    main = [c for c, m in zip(comps, big) if m]
    small = [c for c, m in zip(comps, big) if not m]
    median_h = float(np.median([b[3] for b, _ in main]))
    rows: list[list] = []
    centres: list[float] = []
    for c in sorted(main, key=lambda c: c[0][1] + c[0][3] / 2):
        cy = c[0][1] + c[0][3] / 2
        if not rows or abs(cy - centres[-1]) > 0.7 * median_h:
            rows.append([c]); centres.append(cy)
        else:
            rows[-1].append(c)
            centres[-1] = float(np.mean([b[1] + b[3] / 2 for b, _ in rows[-1]]))
    # Dots and small strokes join the nearest line
    for c in small:
        cy = c[0][1] + c[0][3] / 2
        r = int(np.argmin([abs(cy - ce) for ce in centres]))
        if abs(cy - centres[r]) < 1.2 * median_h:
            rows[r].append(c)

    # 2. Letters within each line
    glyphs: list[Glyph] = []
    for r, row in enumerate(rows):
        pieces = _merge_in_row(row)
        row_h = float(np.median([b[3] for b, _ in pieces]))
        ref_w = float(np.median([b[2] for b, _ in pieces]))
        if len(pieces) < 3:
            ref_w = min(ref_w, 0.8 * row_h)
        letters = []
        for box, labs in pieces:
            x, y, w, h = box
            mask = np.isin(labels[y:y + h, x:x + w], labs)
            letters += _split_wide(box, mask, ref_w, row_h, stroke)
        letters = [(b, m) for b, m in letters if max(b[2], b[3]) >= 0.3 * row_h]
        letters.sort(key=lambda bm: bm[0][0])

        gaps = [b2[0] - (b1[0] + b1[2]) for (b1, _), (b2, _) in zip(letters, letters[1:])]
        typical = max(float(np.median(gaps)), 0.0) if gaps else 0.0
        space_gap = max(0.75 * row_h, 2.5 * typical)
        prev_right = None
        for (x, y, w, h), m in letters:
            crop = ink[y:y + h, x:x + w] * m
            space = prev_right is not None and (x - prev_right) > space_gap
            glyphs.append(Glyph(box=(x, y, w, h), crop=crop, row=r, space_before=space))
            prev_right = x + w
    return glyphs


def crop_to_page(ink: np.ndarray, trim_pct: float = 0.0) -> tuple[np.ndarray, tuple[int, int]]:
    """
    For screen captures: if a page border / window frame surrounds the writing (a thin piece of
    ink spanning most of the image), keep only what's inside it. Then trim trim_pct from each side.
    Returns (cropped ink, (x offset, y offset)).
    """
    H, W = ink.shape
    x0, y0, x1, y1 = 0, 0, W, H
    binary = (ink > 0.25).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w > 0.5 * W and h > 0.5 * H and area < 0.1 * w * h:      # a frame, not writing
            if best is None or w * h < best[2] * best[3]:            # innermost frame
                best = (x, y, w, h)
    if best is not None:
        x, y, w, h = best
        m = max(4, int(0.01 * max(w, h)))
        x0, y0, x1, y1 = x + m, y + m, x + w - m, y + h - m
    ty, tx = int((y1 - y0) * trim_pct / 100), int((x1 - x0) * trim_pct / 100)
    x0, y0, x1, y1 = x0 + tx, y0 + ty, x1 - tx, y1 - ty
    return ink[y0:y1, x0:x1], (x0, y0)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    letter: str
    confidence: float
    probs: np.ndarray                    # (26,)
    model_input: np.ndarray              # (S, S) image the model saw, for display
    glyph: Glyph = field(repr=False, default=None)


def _no_foot(crop: np.ndarray) -> bool:
    """True if a glyph is a (possibly slanted) bar with no horizontal foot, i.e. an I rather than an L."""
    m = crop > 0.25
    h, w = m.shape
    if h == 0 or w >= 0.6 * h:
        return False
    cols = np.nonzero(m[int(0.8 * h):].any(axis=0))[0]
    return len(cols) == 0 or (cols.max() - cols.min() + 1) < max(0.5 * w, 0.18 * h)


class Recognizer:
    def __init__(self):
        self._models = {}
        self._lock = threading.Lock()
        self._torch_tf = None
        self._device = None

    # -- loading ------------------------------------------------------------
    def load(self, key: str):
        with self._lock:
            if key in self._models:
                return self._models[key]
            info = MODELS[key]
            if not info.path.is_file():
                raise FileNotFoundError(f"Model weights not found: {info.path}")
            if info.framework == "keras":
                import keras
                model = keras.models.load_model(info.path, compile=False)
            else:
                import torch
                from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
                if self._device is None:
                    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # Same transform the notebook used (resize 232 -> crop 224 -> ImageNet norm).
                    # Building it doesn't download anything.
                    self._torch_tf = MobileNet_V2_Weights.DEFAULT.transforms()
                model = mobilenet_v2(weights=None, num_classes=26)
                state = torch.load(info.path, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                model.to(self._device).eval()
            self._models[key] = model
            return model

    # -- prediction ---------------------------------------------------------
    def predict_crops(self, crops: list[np.ndarray], model_key: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """Returns [(probs(26,), model_input(S,S)), ...]."""
        if not crops:
            return []
        info = MODELS[model_key]
        model = self.load(model_key)
        imgs = np.stack([to_emnist(c) for c in crops])                               # (N, 28, 28)

        # From here on, each model gets exactly what it got in training (see notebooks).
        if info.framework == "keras":
            import tensorflow as tf
            x = tf.image.resize(imgs[..., None], (info.input_size, info.input_size))  # bilinear, like load_split
            x = tf.image.grayscale_to_rgb(x).numpy()                                 # (N, 64, 64, 3)
            probs = np.asarray(model.predict(x, verbose=0))
        else:
            import torch
            x = torch.from_numpy(imgs).unsqueeze(1).repeat(1, 3, 1, 1)               # (N, 3, 28, 28)
            x = self._torch_tf(x).to(self._device)
            with torch.inference_mode():
                probs = torch.softmax(model(x), dim=1).cpu().numpy()
        return list(zip(probs, imgs))

    def recognize(self, canvas: np.ndarray, model_key: str, mode: str = "word") -> tuple[str, list[Prediction]]:
        ink = canvas_to_ink(canvas)
        glyphs = segment(ink, mode=mode)
        outs = self.predict_crops([g.crop for g in glyphs], model_key)
        preds = []
        for g, (p, img) in zip(glyphs, outs):
            i = int(np.argmax(p))
            letter = CLASS_NAMES[i]
            # EMNIST's "L" class includes lowercase l, which is just a vertical bar. Output is in
            # capitals, so a bare vertical bar (no foot) is far more likely to be an I.
            if letter == "L" and _no_foot(g.crop):
                letter = "I"
            preds.append(Prediction(letter, float(p[i]), p, img, g))
        return compose_text(preds), preds


def compose_text(preds: list[Prediction]) -> str:
    text = ""
    last_row = None
    for p in preds:
        if last_row is not None and p.glyph.row != last_row:
            text += "\n"
        elif p.glyph.space_before:
            text += " "
        text += p.letter
        last_row = p.glyph.row
    return text


def annotate(canvas: np.ndarray, preds: list[Prediction]) -> np.ndarray:
    """Draw boxes and predicted letters over the drawing (RGB uint8)."""
    ink = canvas_to_ink(canvas)
    img = (255 - ink * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Room above the drawing so every label fits
    top = 44
    img = cv2.copyMakeBorder(img, top, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for p in preds:
        x, y, w, h = p.glyph.box
        y += top
        c = p.confidence
        color = (34, 160, 80) if c >= 0.8 else (230, 150, 20) if c >= 0.5 else (210, 50, 50)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        label = f"{p.letter} {c:.0%}"
        fs = max(0.5, min(1.0, w / 110))
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        ty = y - 8 if y - th - 12 > 0 else y + h + th + 8
        cv2.rectangle(img, (x, ty - th - 5), (x + tw + 8, ty + base), color, -1)
        cv2.putText(img, label, (x + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
    return img
