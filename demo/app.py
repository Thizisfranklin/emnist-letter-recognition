"""
EMNIST handwriting demo: pick a model, draw a letter or word, and the prediction is
added to a text document you can edit and download.

    pip install -r demo/requirements.txt
    python demo/app.py            # http://127.0.0.1:7860
    python demo/app.py --share    # also prints a temporary public link
"""

from __future__ import annotations

import argparse
import tempfile
import threading
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from recognizer import CLASS_NAMES, MODELS, Recognizer, annotate

rec = Recognizer()

MODEL_CHOICES = [(m.label, k) for k, m in MODELS.items()]
DEFAULT_MODEL = "mobilenet_scratch"
MODES = [("Single letter", "letter"), ("Word(s)", "word")]
TXT_PATH = Path(tempfile.gettempdir()) / "emnist_demo" / "my_text.txt"
TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
TXT_PATH.write_text("", encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _composite(sketch) -> np.ndarray | None:
    """Pull the flattened drawing out of the sketchpad value."""
    if sketch is None:
        return None
    img = sketch.get("composite") if isinstance(sketch, dict) else sketch
    if img is None:
        return None
    img = np.asarray(img)
    return img if img.size else None


def _strip(preds, tile=96, per_row=12) -> np.ndarray:
    """One image showing the 28x28 input each letter became, with its prediction underneath."""
    cells = []
    for p in preds:
        t = cv2.resize((p.model_input * 255).astype(np.uint8), (tile, tile), interpolation=cv2.INTER_NEAREST)
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2RGB)
        cap = np.full((26, tile, 3), 255, np.uint8)
        label = f"{p.letter} {p.confidence:.0%}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(cap, label, ((tile - tw) // 2, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        cell = np.vstack([t, cap])
        cells.append(cv2.copyMakeBorder(cell, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    while len(cells) % per_row and len(cells) > per_row:
        cells.append(np.full_like(cells[0], 255))
    rows = [np.hstack(cells[i:i + per_row]) for i in range(0, len(cells), per_row)]
    return np.vstack(rows)


def _append(doc: str, new: str, mode: str) -> str:
    if not new:
        return doc
    if mode == "word" and doc and not doc[-1].isspace():
        doc += " "
    return doc + new


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def recognize(sketch, model_key, mode, doc, auto_add, auto_clear):
    canvas = _composite(sketch)
    nothing = (None, None, {}, [], "", doc, "Draw something first.")
    if canvas is None:
        return nothing
    text, preds = rec.recognize(canvas, model_key, mode)
    if not preds:
        return nothing

    tiles = _strip(preds)
    # Top-5 for the first letter (single-letter mode) or the least confident one (word mode)
    focus = preds[0] if mode == "letter" else min(preds, key=lambda p: p.confidence)
    top5 = {CLASS_NAMES[i]: float(focus.probs[i]) for i in np.argsort(focus.probs)[::-1][:5]}
    table = []
    for n, p in enumerate(preds, 1):
        order = np.argsort(p.probs)[::-1]
        table.append([n, p.letter, f"{p.confidence:.0%}",
                      f"{CLASS_NAMES[order[1]]} {p.probs[order[1]]:.0%}",
                      f"{CLASS_NAMES[order[2]]} {p.probs[order[2]]:.0%}"])

    new_doc = _append(doc, text, mode) if auto_add else doc
    avg = np.mean([p.confidence for p in preds])
    status = (f"**{MODELS[model_key].label}** (test acc {MODELS[model_key].test_accuracy}) read {len(preds)} letter{'s' * (len(preds) != 1)} "
              f"· average confidence {avg:.0%}" + (" · added to text" if auto_add else ""))
    return annotate(canvas, preds), tiles, top5, table, text, new_doc, status


def compare_all(sketch, mode):
    canvas = _composite(sketch)
    if canvas is None:
        return []
    rows = []
    for key, info in MODELS.items():
        text, preds = rec.recognize(canvas, key, mode)
        conf = np.mean([p.confidence for p in preds]) if preds else 0.0
        rows.append([info.label, text.replace("\n", " ⏎ "), f"{conf:.0%}", info.test_accuracy])
    return rows


def save_txt(doc: str) -> str:
    TXT_PATH.write_text(doc, encoding="utf-8")
    return str(TXT_PATH)


def warm_up(model_key):
    rec.load(model_key)
    m = MODELS[model_key]
    return f"Loaded **{m.label}** · test accuracy {m.test_accuracy}. {m.notes}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
#doc textarea { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 1.15rem; }
#recognized input { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 1.3rem; letter-spacing: .08em; }
"""

# Gradio's sketchpad starts with no tool selected, and clearing it from Python doesn't
# reliably reset the canvas. These snippets press the sketchpad's own toolbar buttons.
_PICK_BRUSH = """
  const pick = (tries) => {
    const btn = document.querySelector('#sketch button[aria-label="Brush"]');
    if (btn) btn.click(); else if (tries > 0) setTimeout(() => pick(tries - 1), 250);
  };
  setTimeout(() => pick(40), 200);
"""
SELECT_BRUSH_JS = "() => {" + _PICK_BRUSH + "}"
CLEAR_CANVAS_JS = """
(...args) => {
  if (args.length && args[0] === false) return args;
  document.querySelector('#sketch button[aria-label="Clear canvas"]')?.click();
""" + _PICK_BRUSH + """
  return args;
}
"""

ABOUT = "\n".join(
    f"- **{m.label}** — test accuracy {m.test_accuracy} on the 855-image capstone test split. {m.notes}"
    for m in MODELS.values()
) + """

**How a drawing becomes model input.** The drawing is split into letters wherever there's a gap
(word mode), or treated as one letter (single-letter mode). Each letter is converted to EMNIST
format: strokes redrawn at a consistent width, the longest side scaled to 24 px and centred in a
28×28 frame. That 28×28 image then goes through the same resizing each model was trained with
(64×64 for DenseNet, 224×224 with ImageNet normalisation for MobileNetV2).

**Tips.** Print letters with a small gap between them, write words on one line, and leave a wider
gap between words. Upper- and lowercase share a class, so results are shown in capitals.
"""

with gr.Blocks(title="EMNIST Letter Recognition Demo") as demo:
    gr.Markdown("# ✍️ EMNIST Letter Recognition\nPick a model, draw a letter or a word, and watch it get typed into the text file.")

    with gr.Row():
        # ---------------- left: input -----------------
        with gr.Column(scale=3):
            with gr.Row():
                model_dd = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, label="Model", scale=3)
                mode_rb = gr.Radio(MODES, value="word", label="What are you drawing?", scale=2)
            sketch = gr.Sketchpad(
                label="Draw here",
                type="numpy",
                image_mode="RGBA",
                canvas_size=(700, 300),
                fixed_canvas=True,
                brush=gr.Brush(default_size=8, colors=["#000000"], color_mode="fixed"),
                layers=False,
                height=400,
                elem_id="sketch",
            )
            with gr.Row():
                go_btn = gr.Button("Recognize", variant="primary", scale=3)
                clear_btn = gr.Button("Clear drawing", scale=1)
            with gr.Row():
                auto_add = gr.Checkbox(value=True, label="Add result to text automatically")
                auto_clear = gr.Checkbox(value=True, label="Clear drawing after recognizing")
            status = gr.Markdown()

        # ---------------- right: output -----------------
        with gr.Column(scale=2):
            recognized = gr.Textbox(label="Recognized (edit if needed)", elem_id="recognized", interactive=True)
            add_btn = gr.Button("Add to text ↓")
            doc = gr.Textbox(label="my_text.txt", lines=9, max_lines=20, interactive=True, elem_id="doc",
                             placeholder="Recognized letters show up here…")
            with gr.Row():
                space_btn = gr.Button("Space", size="sm")
                nl_btn = gr.Button("New line", size="sm")
                bksp_btn = gr.Button("⌫ Backspace", size="sm")
                clr_doc_btn = gr.Button("Clear text", size="sm", variant="stop")
            dl_btn = gr.DownloadButton("Download my_text.txt", value=str(TXT_PATH))

    with gr.Row():
        annotated = gr.Image(label="What the model read", type="numpy", interactive=False, scale=3)
        with gr.Column(scale=2):
            top5 = gr.Label(num_top_classes=5, label="Top 5 guesses (single letter, or least confident letter)")
    tiles = gr.Image(label="Model input for each letter (28×28 EMNIST format)", type="numpy",
                     interactive=False, height=150)
    per_letter = gr.Dataframe(headers=["#", "Prediction", "Confidence", "2nd guess", "3rd guess"],
                              label="Per-letter detail", interactive=False, wrap=True)

    with gr.Accordion("Compare all three models on this drawing", open=False):
        gr.Markdown("Turn off *Clear drawing after recognizing* first, or press this before *Recognize*.")
        cmp_btn = gr.Button("Run all three models")
        cmp_tbl = gr.Dataframe(headers=["Model", "Reads", "Avg confidence", "Test accuracy"], interactive=False)

    with gr.Accordion("About the models", open=False):
        gr.Markdown(ABOUT)

    # ---------------- wiring -----------------
    go_btn.click(recognize, [sketch, model_dd, mode_rb, doc, auto_add, auto_clear],
                 [annotated, tiles, top5, per_letter, recognized, doc, status]
                 ).then(None, auto_clear, None, js=CLEAR_CANVAS_JS)
    clear_btn.click(None, js=CLEAR_CANVAS_JS)
    add_btn.click(_append, [doc, recognized, mode_rb], doc)
    space_btn.click(lambda d: d + " ", doc, doc)
    nl_btn.click(lambda d: d + "\n", doc, doc)
    bksp_btn.click(lambda d: d[:-1], doc, doc)
    clr_doc_btn.click(lambda: "", None, doc)
    doc.change(save_txt, doc, dl_btn)
    cmp_btn.click(compare_all, [sketch, mode_rb], cmp_tbl)
    model_dd.change(warm_up, model_dd, status)
    demo.load(warm_up, model_dd, status)
    demo.load(None, js=SELECT_BRUSH_JS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    # Load every model in the background so switching models is instant once the page is up
    threading.Thread(target=lambda: [rec.load(k) for k in MODELS], daemon=True).start()
    demo.launch(share=args.share, server_port=args.port, css=CSS,
                theme=gr.themes.Soft(primary_hue="indigo"))
