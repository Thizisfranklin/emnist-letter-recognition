"""
Computer Vision Demo: pick a model, then either draw on screen or write on a reMarkable
tablet (via its screen share), and the prediction is typed into my_text.txt.

    pip install -r demo/requirements.txt
    python demo/app.py            # http://127.0.0.1:7860
    python demo/app.py --share    # also prints a temporary public link
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import tempfile
import threading
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from recognizer import MODELS, Recognizer, annotate, canvas_to_ink, crop_to_page

rec = Recognizer()

MODEL_CHOICES = [(m.label, k) for k, m in MODELS.items()]
DEFAULT_MODEL = "mobilenet_scratch"
TEAM = "Michael Sheehey, Nathan Butler, Franklin Osualaaham, Sarthakkumar Patel, Joseph Uzoigwe"
TXT_PATH = Path(tempfile.gettempdir()) / "cv_demo" / "my_text.txt"
TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
TXT_PATH.write_text("", encoding="utf-8")


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


# ---------------------------------------------------------------------------
# reMarkable tab: the browser captures the reMarkable screen-share window, grabs a frame
# when asked, and the whole page is read into my_text.txt (so re-reading never duplicates).
# ---------------------------------------------------------------------------

def _decode(data_url: str) -> np.ndarray | None:
    if not data_url or "," not in data_url:
        return None
    buf = np.frombuffer(base64.b64decode(data_url.split(",", 1)[1]), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)          # BGR, white page with dark ink


def read_page(frame, model_key, trim_pct, doc, last_key, last_img):
    img = _decode(frame)
    if img is None:
        return doc, last_key, last_img
    # Skip work if nothing changed since the last read (auto-read runs every few seconds)
    key = hashlib.md5(cv2.resize(img, (200, 200)).tobytes() + f"{model_key}{trim_pct}".encode()).hexdigest()
    if key == last_key:
        return doc, last_key, last_img
    # Keep only the page (drops the reMarkable window frame/toolbar), then the edge trim
    ink, _ = crop_to_page(canvas_to_ink(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), trim_pct)
    if ink.size == 0:
        return doc, key, last_img
    # Small screen-share windows give tiny letters; enlarge so faint strokes survive
    short = min(ink.shape)
    if short < 1000:
        f = 1000 / short
        ink = cv2.resize(ink, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC).clip(0, 1)
    page = cv2.cvtColor(((1 - ink) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    text, preds = rec.recognize(page, model_key, mode="word")
    view = annotate(page, preds)
    if preds:                                   # zoom the preview to the writing
        ys, xs = np.nonzero(view.min(axis=2) < 250)
        pad = 40
        view = view[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]
    return text, key, view


def save_txt(doc: str) -> str:
    TXT_PATH.write_text(doc, encoding="utf-8")
    return str(TXT_PATH)


SCREEN_JS = """
() => {
  window.rmStream = null;
  const show = (on) => {
    const ph = document.querySelector('#rm-placeholder');
    if (ph) ph.style.display = on ? 'none' : 'flex';
  };
  window.rmStart = async () => {
    try {
      const s = await navigator.mediaDevices.getDisplayMedia({ video: { cursor: 'never' }, audio: false });
      window.rmStream = s;
      const v = document.querySelector('#rm-video');
      v.srcObject = s; await v.play(); show(true);
      s.getVideoTracks()[0].addEventListener('ended', () => { window.rmStream = null; show(false); });
    } catch (e) { console.log('screen share cancelled', e); }
  };
  window.rmStop = () => {
    if (window.rmStream) window.rmStream.getTracks().forEach(t => t.stop());
    window.rmStream = null; show(false);
  };
  window.rmGrab = () => {
    const v = document.querySelector('#rm-video');
    if (!window.rmStream || !v || !v.videoWidth) return '';
    const c = document.createElement('canvas');
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    return c.toDataURL('image/png');
  };
}
"""
GRAB_JS = "(frame, ...rest) => [window.rmGrab ? window.rmGrab() : '', ...rest]"

SCREEN_HTML = """
<div id="rm-wrap">
  <div id="rm-frame"><video id="rm-video" autoplay muted playsinline></video></div>
  <div id="rm-placeholder">Click <b>&nbsp;Start screen share&nbsp;</b> and pick the reMarkable window</div>
</div>
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
() => {
  document.querySelector('#sketch button[aria-label="Clear canvas"]')?.click();
""" + _PICK_BRUSH + """
}
"""

CSS = """
#title { text-align: center; }
#title h1 { margin-bottom: 0.2rem; }
#rm-wrap { position: relative; background: #f3f4f6; border-radius: 10px; overflow: hidden;
           height: 560px; display: flex; align-items: center; justify-content: center; }
#rm-frame { position: relative; overflow: hidden; line-height: 0; max-width: 100%; max-height: 100%; }
#rm-video { max-width: 100%; max-height: 560px; display: block; }
#rm-placeholder { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
                  color: #6b7280; font-size: 1.05rem; }
#doc textarea, #rm-doc textarea { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 1.2rem; }
"""

with gr.Blocks(title="Computer Vision Demo") as demo:
    gr.Markdown(f"# Computer Vision Demo\n**Team:** {TEAM}", elem_id="title")
    model_dd = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, label="Model")

    with gr.Tabs():
        # ---------------- Draw on screen -----------------
        with gr.Tab("Draw"):
            with gr.Row():
                with gr.Column(scale=3):
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

        # ---------------- reMarkable screen share -----------------
        with gr.Tab("reMarkable tablet"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML(SCREEN_HTML)
                    with gr.Row():
                        share_btn = gr.Button("Start screen share", scale=2)
                        stop_btn = gr.Button("Stop", scale=1)
                        read_btn = gr.Button("Read page", variant="primary", scale=2)
                    with gr.Row():
                        auto_cb = gr.Checkbox(value=False, label="Auto-read every 2 seconds")
                        trim = gr.Slider(0, 15, value=3, step=1, label="Ignore page edges (%)",
                                         info="The page border is found automatically; raise this if icons near the edge get read")
                with gr.Column(scale=2):
                    rm_doc = gr.Textbox(label="my_text.txt", lines=18, max_lines=30, interactive=True,
                                        elem_id="rm-doc", placeholder="Your tablet page shows up here as text…")
                    dl_btn = gr.DownloadButton("Download my_text.txt", value=str(TXT_PATH))
            with gr.Accordion("What the model read", open=False):
                read_view = gr.Image(show_label=False, type="numpy", interactive=False, height=600)

    frame = gr.Textbox(visible=False)
    last_key = gr.State("")
    last_img = gr.State(None)
    timer = gr.Timer(2.0, active=False)

    # Draw tab
    go_btn.click(recognize, [sketch, model_dd, doc], doc).then(None, js=CLEAR_CANVAS_JS)
    clear_btn.click(None, js=CLEAR_CANVAS_JS)

    # reMarkable tab
    read_inputs = [frame, model_dd, trim, rm_doc, last_key, last_img]
    read_outputs = [rm_doc, last_key, read_view]
    share_btn.click(None, js="() => window.rmStart()")
    stop_btn.click(None, js="() => window.rmStop()").then(lambda: (gr.Timer(active=False), False),
                                                          None, [timer, auto_cb])
    read_btn.click(read_page, read_inputs, read_outputs, js=GRAB_JS).then(lambda v: v, read_view, last_img)
    timer.tick(read_page, read_inputs, read_outputs, js=GRAB_JS, show_progress="hidden"
               ).then(lambda v: v, read_view, last_img)
    auto_cb.change(lambda on: gr.Timer(active=on), auto_cb, timer)
    rm_doc.change(save_txt, rm_doc, dl_btn)

    demo.load(None, js=SELECT_BRUSH_JS)
    demo.load(None, js=SCREEN_JS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    # Load every model in the background so switching models is instant once the page is up
    threading.Thread(target=lambda: [rec.load(k) for k in MODELS], daemon=True).start()
    demo.launch(share=args.share, server_port=args.port, css=CSS,
                theme=gr.themes.Soft(primary_hue="indigo"))
