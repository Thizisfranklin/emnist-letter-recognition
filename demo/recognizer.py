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
    dt = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)
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


def _merge_overlapping(boxes: list[list[int]], labels_per_box: list[list[int]]):
    """Merge components whose horizontal extents overlap (i/j dots, a crossbar drawn separately...)."""
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ax, ay, aw, ah = boxes[i]
                bx, by, bw, bh = boxes[j]
                overlap = min(ax + aw, bx + bw) - max(ax, bx)
                if overlap <= 0:
                    continue
                narrow = min(aw, bw)
                # vertical distance between them (0 if they overlap vertically)
                v_gap = max(0, max(ay, by) - min(ay + ah, by + bh))
                tall = max(ah, bh)
                if overlap >= 0.5 * narrow and v_gap <= 0.8 * tall:
                    x0, y0 = min(ax, bx), min(ay, by)
                    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    labels_per_box[i] += labels_per_box[j]
                    del boxes[j], labels_per_box[j]
                    merged = True
                    break
            if merged:
                break
    return boxes, labels_per_box


def segment(ink: np.ndarray, mode: str = "word", ink_threshold: float = 0.25) -> list[Glyph]:
    """
    Split the drawing into letters.

    mode="letter": everything on the canvas is one letter.
    mode="word":   connected pieces of ink become letters, read left-to-right,
                   top-to-bottom. Big horizontal gaps become spaces, new rows become new lines.
    """
    binary = (ink > ink_threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []

    if mode == "letter":
        ys, xs = np.nonzero(binary)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        return [Glyph(box=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)), crop=ink[y0:y1, x0:x1].copy())]

    # Close tiny gaps inside strokes so a letter isn't split by a shaky mouse
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)

    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 15:                       # specks
            continue
        comps.append(([int(x), int(y), int(w), int(h)], [i]))
    if not comps:
        return []

    boxes = [c[0] for c in comps]
    lab_lists = [c[1] for c in comps]
    boxes, lab_lists = _merge_overlapping(boxes, lab_lists)

    heights = [b[3] for b in boxes]
    median_h = float(np.median(heights))

    # Drop leftover dots/specks that didn't merge into anything
    keep = [k for k, b in enumerate(boxes) if max(b[2], b[3]) >= 0.25 * median_h]
    boxes = [boxes[k] for k in keep]
    lab_lists = [lab_lists[k] for k in keep]

    # Group into rows by vertical centre
    order = sorted(range(len(boxes)), key=lambda k: boxes[k][1] + boxes[k][3] / 2)
    rows: list[list[int]] = []
    row_centre = None
    for k in order:
        cy = boxes[k][1] + boxes[k][3] / 2
        if row_centre is None or abs(cy - row_centre) > 0.7 * median_h:
            rows.append([k])
            row_centre = cy
        else:
            rows[-1].append(k)
            row_centre = np.mean([boxes[m][1] + boxes[m][3] / 2 for m in rows[-1]])

    glyphs: list[Glyph] = []
    for r, row in enumerate(rows):
        row.sort(key=lambda k: boxes[k][0])
        # A gap is a space if it's clearly wider than the usual gap between letters in this row
        gaps = [boxes[b][0] - (boxes[a][0] + boxes[a][2]) for a, b in zip(row, row[1:])]
        typical = float(np.median(gaps)) if gaps else 0.0
        space_gap = max(0.6 * median_h, 2.2 * max(typical, 0.0))
        prev_right = None
        for k in row:
            x, y, w, h = boxes[k]
            mask = np.isin(labels[y:y + h, x:x + w], lab_lists[k])
            crop = ink[y:y + h, x:x + w] * mask
            space = prev_right is not None and (x - prev_right) > space_gap
            glyphs.append(Glyph(box=(x, y, w, h), crop=crop, row=r, space_before=space))
            prev_right = x + w
    return glyphs


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
            preds.append(Prediction(CLASS_NAMES[i], float(p[i]), p, img, g))
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
