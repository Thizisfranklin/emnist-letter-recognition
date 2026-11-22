"""
Computer Vision Demo: pick a model, draw a letter or word, and the prediction is
typed into my_text.txt.

    pip install -r demo/requirements.txt
    python demo/app.py            # http://127.0.0.1:7860
    python demo/app.py --share    # also prints a temporary public link
"""

from __future__ import annotations

import argparse
import threading

import gradio as gr
import numpy as np

from recognizer import MODELS, Recognizer

rec = Recognizer()

MODEL_CHOICES = [(m.label, k) for k, m in MODELS.items()]
DEFAULT_MODEL = "mobilenet_scratch"
TEAM = "Michael Sheehey, Nathan Butler, Franklin Osualaaham, Sarthakkumar Patel, Joseph Uzoigwe"


def _composite(sketch) -> np.ndarray | None:
    """Pull the flattened drawing out of the sketchpad value."""
    if sketch is None:
        return None
    img = sketch.get("composite") if isinstance(sketch, dict) else sketch
    if img is None:
        return None
    img = np.asarray(img)
    return img if img.size else None


def recognize(sketch, model_key, doc):
    canvas = _composite(sketch)
    if canvas is None:
        return doc
    text, preds = rec.recognize(canvas, model_key, mode="word")
    if not text:
        return doc
    # A single letter is appended directly (so you can build a word letter by letter);
    # a whole word gets a space in front of it.
    if len(text) > 1 and doc and not doc[-1].isspace():
        doc += " "
    return doc + text


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
() => {
  document.querySelector('#sketch button[aria-label="Clear canvas"]')?.click();
""" + _PICK_BRUSH + """
}
"""

CSS = """
#title { text-align: center; }
#title h1 { margin-bottom: 0.2rem; }
#doc textarea { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 1.2rem; }
"""

with gr.Blocks(title="Computer Vision Demo") as demo:
    gr.Markdown(f"# Computer Vision Demo\n**Team:** {TEAM}", elem_id="title")

    with gr.Row():
        with gr.Column(scale=3):
            model_dd = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, label="Model")
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
                clear_btn = gr.Button("Clear", scale=1)

        with gr.Column(scale=2):
            doc = gr.Textbox(label="my_text.txt", lines=18, max_lines=30, interactive=True,
                             elem_id="doc", placeholder="Recognized letters show up here…")

    go_btn.click(recognize, [sketch, model_dd, doc], doc).then(None, js=CLEAR_CANVAS_JS)
    clear_btn.click(None, js=CLEAR_CANVAS_JS)
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
