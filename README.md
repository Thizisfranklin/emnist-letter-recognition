# Handwritten Letter Recognition with Deep Learning

**Computer Vision | EMNIST Letters Tiny | CNN Model Comparison | Interactive Demo**

For our capstone project, we built and compared deep learning models that recognize handwritten English letters (A–Z). We used a smaller EMNIST dataset, trained three CNN architectures, analyzed their classification errors, and built a Gradio demo to try the models on handwritten input.

**Project status: Completed — preprocessing, model training, evaluation, and capstone presentation.**

[**Cleaned dataset — Google Drive**](https://drive.google.com/drive/folders/1Lb3EGMDdHWG85wYMiCmRS2rJZSia8MKU?usp=sharing)

## Dataset and Preprocessing

Our dataset comes from [Voxel51's EMNIST Letters Tiny](https://huggingface.co/datasets/Voxel51/emnist-letters-tiny). It contains **10,000 grayscale images**, each **28 × 28 pixels**, across **26 letter classes**. Uppercase and lowercase writing styles share the same class label.

During preprocessing, we found that the source-tagged test subset contained only **19 classes (A–S)**, with T–Z missing. Instead of using it as our main evaluation set, we reserved those images and made reproducible, stratified splits from the **8,547 source-training images** (seed 42).

| Split | Images | Classes |
|---|---:|---:|
| Training | 6,837 | 26 |
| Validation | 855 | 26 |
| Main test | 855 | 26 |
| Reserved original source test | 1,453 | 19 (A–S) |

We also checked image orientation, exact duplicates, class balance, stroke intensity, and visually ambiguous letters. We kept unusual but potentially valid handwriting rather than automatically removing it. The shared `.npz` images were normalized to **[0, 1]**; individual PNGs were saved for viewing and optional image-folder loading.

See the [preprocessing notebook, documentation, and shared image loader](Preprocessing/).

## Models and Results

We trained and evaluated the following models using the **same 855-image, 26-class held-out test set**:

| Model | Approach | Test accuracy |
|---|---|---:|
| **Baseline CNN** | Two convolutional layers, pooling, dropout, and a dense classifier | **88.42%** |
| **DenseNet121** | Fine-tuned CNN using transfer learning | **~88.9%** |
| **MobileNetV2 — transfer learning** | ImageNet feature extractor frozen; trained classifier head | **80.70%** |
| **MobileNetV2 — from scratch** | Random initialization; trained all layers | **90.29%** |

In our experiments, **MobileNetV2 trained from scratch performed best**. The frozen pretrained version did not perform as well, showing that transfer learning is not automatically the strongest option for a small grayscale handwriting task.

We also examined training/validation curves, per-class performance, and confusion matrices. For the baseline CNN, some of the largest confusions were **Q → G** and **I → L**, matching the visually similar letters noticed during preprocessing.

Explore the code and model artifacts:
- [Baseline CNN notebook and results](Baseline_Model/)
- [DenseNet notebooks, plots, and saved model](models/DenseNet201/)
- [MobileNetV2 notebook and model checkpoints](models/MobileNet/)

**Naming note:** The DenseNet folder and saved `.keras` filename say “DenseNet201,” but the model used in the final experiments is **DenseNet121**, as documented in the demo code.

## Interactive Demo

We also built a [**Gradio handwritten-letter recognition demo**](demo/) for our presentation.

It lets users:
- **Draw** a letter or a word on the screen and get predictions.
- **Share a reMarkable tablet screen** and recognize handwriting from the captured page.
- **Switch between the saved DenseNet and MobileNetV2 models** and export recognized text.

The demo segments handwriting into character candidates, converts them into EMNIST-style inputs, and applies the transforms required by each model. Letter recognition can still struggle with connected or cursive writing.

### Run the demo locally

From the repository root, install the dependencies and launch the application:

```bash
pip install -r demo/requirements.txt
python demo/app.py
```

Open **http://127.0.0.1:7860**. For a temporary public share link, run:

```bash
python demo/app.py --share
```

Gradio share links expire, so we haven't included our presentation's temporary link as a permanent demo URL. See [demo/README.md](demo/README.md) for instructions and troubleshooting.

## Repository Structure

```text
Preprocessing/       Preprocessing notebook, README, and shared image loader
Baseline_Model/      Baseline CNN notebook and results documentation
models/DenseNet201/  DenseNet experiments, saved model, and evaluation plots
models/MobileNet/    MobileNetV2 experiments and saved checkpoints
demo/                Gradio application, recognition pipeline, and run instructions
README.md            Project overview
```

## Limitations and Lessons

Our main test set was created from the original source-training subset, **not** the official full EMNIST test partition. Writer-independent separation was not verified, and each test class has relatively few examples. We also observed differences between performance on the benchmark test set and informal handwritten demo inputs, so the reported accuracy figures should not be treated as real-world handwriting accuracy.

The project reinforced that consistent preprocessing, architecture choice, model tuning, and how we prepare user-drawn characters all affect the final recognition system.

## Contributors

- Nathan Butler
- Michael Sheehey
- Franklin Osualaaham
- Joseph Uzoigwe
- Sarthakkumar Patel
