# Computer Vision Demo

A small Gradio web app built on our trained models. Pick a model, draw a letter or a word with
the mouse (or a finger or stylus), and the prediction is typed into `my_text.txt` on screen.

## Run it

From the repository root:

```bash
pip install -r demo/requirements.txt
python demo/app.py            # open http://127.0.0.1:7860
python demo/app.py --share    # also prints a temporary public link (handy for presenting)
```

The app loads the weights already in this repo, so no Google Drive data is needed:

| Model in the dropdown | Weights file | Framework | Test acc.* |
|---|---|---|---|
| DenseNet (fine-tuned) | `models/DenseNet201/DenseNet201_Finetuned.keras` | Keras | 88.9% |
| MobileNetV2, transfer learning | `models/MobileNet/checkpoints/mobilenetv2_best.pth` | PyTorch | 80.7% |
| MobileNetV2, trained from scratch | `models/MobileNet/checkpoints/mobilenetv2_scratch_best.pth` | PyTorch | 90.3% |

\*As reported in each notebook on the 855-image capstone test split.

> Note: `DenseNet201_Finetuned.keras` actually contains a **DenseNet121** network
> (the v2–v4 notebooks build `tf.keras.applications.DenseNet121`). The demo labels it "DenseNet".

## Using it

1. **Model**: choose one of the three models (all are loaded at startup, so switching is instant).
2. **Draw here**: draw a letter or print a word, leaving small gaps between letters.
3. Press **Recognize**. The prediction is added to **my_text.txt** and the canvas clears for the next one.
   A single letter is appended directly, so you can build a word one letter at a time; a whole
   word gets a space in front of it. **Clear** wipes the canvas without recognizing.
4. **my_text.txt** can be edited directly (fix a letter, add spaces or new lines).

Tips: print letters with small gaps between them (cursive and touching letters are read as one
letter). Upper- and lowercase share a class in EMNIST Letters, so output is always in capitals.

## How a drawing becomes model input

Our models were trained on 28×28 EMNIST images, so each drawn letter is converted to that format first:

1. **Segment** (Word mode): connected strokes become letter candidates. Pieces stacked above each
   other (the dot on an *i*, a separately drawn crossbar) are merged. Letters are read left to
   right and line by line.
2. **Normalize the stroke width**: the strokes are redrawn so they end up about 3.6 px wide at
   28×28, close to the ~3.9 px measured on real EMNIST letters. Big, small, thin-brush and
   thick-brush letters therefore all look alike to the model.
3. **Fit to EMNIST framing**: the longest side is scaled to 24 px (aspect ratio kept) and the
   letter is centred in a 28×28 frame, as EMNIST does. Edges are slightly softened and the image
   is scaled to [0, 1].
4. **Model-specific step, identical to training**:
   - DenseNet: bilinear resize to 64×64 and copy to 3 channels (`load_split` in the notebook).
   - MobileNetV2 (both): copy to 3 channels, then `MobileNet_V2_Weights.DEFAULT.transforms()`
     (resize to 232, centre-crop to 224, ImageNet normalization), the same as `prepare_batch`.

Sanity check: on 300 held-out EMNIST images (the original test split, which the team didn't train
on), raw input scored 85% / 81% / 87% (DenseNet / MobileNet-transfer / MobileNet-scratch). Sending
the same images through the full drawing pipeline above (binarize, redraw, re-frame) scored
79% / 66% / 85%. The DenseNet v4 notebook's `preprocess_character()` was tuned for photographed
paper. On canvas drawings it over-thickened and blurred the strokes, scoring 8–31% on the same
check, so the demo uses the pipeline above instead.

## Files

- `app.py`: Gradio interface
- `recognizer.py`: model loading, segmentation, preprocessing and annotation; it has no UI code and can be imported on its own:

```python
from recognizer import Recognizer
rec = Recognizer()
text, preds = rec.recognize(rgba_image, model_key="mobilenet_scratch", mode="word")
```
