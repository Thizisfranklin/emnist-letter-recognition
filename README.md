# Handwritten Letter Recognition with Deep Learning

**Computer Vision | EMNIST Letters Tiny | CNNs | Transfer Learning | Handwriting Demo**

For our project, we compared deep learning approaches for recognizing handwritten English letters (A–Z). We prepared a shared EMNIST dataset, trained and evaluated three CNN architectures, investigated difficult letter pairs, and took the models beyond the test set by experimenting with handwritten pages and building an interactive Gradio demo.

[**Project Presentation Slides — Google Slides**](https://docs.google.com/presentation/d/17Jc2SwhbPnJ2WA2-AIsS3xI-joKKjaE8t_J3gP1UDDA/edit?slide=id.h3f6c1ef72254beed_0_30#slide=id.h3f6c1ef72254beed_0_30)

## A little background: From MNIST to EMNIST

**MNIST** (*Modified NIST*) is one of the best-known datasets in machine learning. Introduced in **1998**, it contains images of handwritten digits (0–9). It is often called the **“hello world” of computer vision** because recognizing these digits is a common starting point for learning image classification.

In **2017**, researchers introduced **EMNIST** (*Extended MNIST*), building on MNIST’s familiar 28 × 28 image format to include more challenging handwritten-character tasks, especially **letters as well as digits**. Our project uses **EMNIST Letters Tiny**, a smaller 10,000-image subset focused on recognizing **A–Z** rather than numbers.

*Background: [MNIST dataset](https://yann.lecun.com/exdb/mnist/) · [EMNIST at NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) · [EMNIST paper (Cohen et al., 2017)](https://arxiv.org/abs/1702.05373)*

[**Cleaned and preprocessed dataset — Google Drive**](https://drive.google.com/drive/folders/1Lb3EGMDdHWG85wYMiCmRS2rJZSia8MKU?usp=sharing)

## Dataset and Preprocessing

We used [Voxel51's EMNIST Letters Tiny](https://huggingface.co/datasets/Voxel51/emnist-letters-tiny): **10,000 grayscale images** at **28 × 28 pixels**, covering **26 classes (A–Z)**. Uppercase and lowercase versions of each letter share one label.

During exploratory data analysis, we found that the supplied source-test subset had only **19 classes (A–S)**; T–Z were missing. We reserved that subset and made reproducible stratified splits from the **8,547 source-training images** using seed 42.

| Split | Images | Classes |
|---|---:|---:|
| Training | 6,837 | 26 |
| Validation | 855 | 26 |
| Main test | 855 | 26 |
| Reserved original source test | 1,453 | 19 (A–S) |

We inspected image orientation, duplicate hashes, label counts, faint strokes, handwriting variation, and stroke intensity. We retained ambiguous but potentially valid handwriting instead of automatically discarding it. Exported `.npz` arrays contain normalized pixel values in **[0, 1]**; PNG copies are available for inspection and image-based loading.

See the [preprocessing notebook, documentation, and shared image loader](Preprocessing/).

## Models and Results

The project compared **three CNN architectures**, with **two different training approaches for MobileNetV2**. The following figures come from the same **855-image, 26-class held-out capstone test set**, rather than the external handwritten pages.

| Model | Training approach | Test accuracy | Macro F1 |
|---|---|---:|---:|
| **Baseline CNN** | Two convolutional layers, max pooling, dropout, and a dense classifier | **88.42%** | **0.883** |
| **DenseNet121** | ImageNet-based transfer learning with fine-tuning | **~88.9%** | — |
| **MobileNetV2 — transfer** | Pretrained ImageNet features frozen; trained a new 26-class head | **80.70%** | **0.804** |
| **MobileNetV2 — scratch** | Same architecture, random initialization; trained all layers | **90.29%** | **0.902** |

### How we trained and tuned the models

- **Baseline CNN:** A simpler reference model trained with Adam and early stopping. The run restored weights from **epoch 11** after training for 16 epochs.
- **DenseNet121:** Adapted the input to **64 × 64 × 3** and fine-tuned the network. The DenseNet work explored hyperparameters using **Hyperband**, including learning rate, dropout, and how many layers to unfreeze; the presentation reported a best epoch of **25**.
- **MobileNetV2:** Repeated the grayscale input across three channels and used the pretrained model's image transforms to prepare **224 × 224** inputs. We compared **frozen pretrained features** against **training the same architecture from scratch**, using the fixed splits. The pretrained run's best validation-loss checkpoint was at **epoch 20**; the scratch run's best was **epoch 7**, with early stopping after epoch 12.

**Main result:** **MobileNetV2 trained from scratch performed best overall** on our capstone test set. The frozen pretrained version was considerably weaker. This does not mean transfer learning is always inferior; features learned from natural photographs do not necessarily transfer well to small grayscale handwritten letters, especially when most pretrained layers remain frozen.

**Model code and artifacts:**
- [Baseline CNN notebook and detailed results](Baseline_Model/)
- [DenseNet experiments, Hyperband trials, visualizations, and saved model](models/DenseNet201/)
- [MobileNetV2 training notebook and both checkpoints](models/MobileNet/)

**Naming note:** The DenseNet folder and saved `DenseNet201_Finetuned.keras` filename retain an older name, but the actual final network is **DenseNet121**.

## Beyond the Test Set: Real Handwritten Images

We also experimented with **handwritten words on separate photographed pages**, rather than only testing on images already provided by EMNIST. This was a useful reality check: correctly classifying isolated, centered benchmark letters is different from recognizing handwriting in a photograph.

For the **DenseNet raw-image experiments**, the team explored additional **OpenCV preprocessing**:

1. **Thresholding / black-and-white conversion:** Separates darker ink strokes from the paper background.
2. **Gaussian blur:** Smooths small variations and noise, though excessive blur can erase fine stroke details.
3. **Bilateral filtering:** Reduces noise while aiming to preserve important edges and character boundaries.

The presentation shows actual handwritten pages, processed character images, and annotated predictions. **Results were mixed**, particularly on less clearly separated or more cursive characters. These extra processing experiments were **not the same as the shared EMNIST preprocessing applied to every training example**, and the photographed pages were **not a standardized external test set**. We therefore do not report their illustrative prediction rates as directly comparable to the test accuracies above.

See the [DenseNet experiments](models/DenseNet201/) for the associated notebooks and raw-image work.

## Error Analysis: Why Similar Letters Still Matter

We compared confusion matrices and per-class scores instead of relying only on overall accuracy.

- The **baseline CNN** frequently confused **Q → G (10 cases)** and **I → L (8 cases)** on the 855-image test set.
- The **MobileNetV2 scratch model** improved overall accuracy, but some letters still remained challenging: for example, its reported test F1 scores were **G: 0.652**, **I: 0.719**, and **L: 0.740**.
- In raw-handwriting experiments and the live demo, visually similar strokes and joined handwriting could still lead to incorrect letters. Informal examples included lookalike interpretations involving **I/J/Y** and **U/V**; these are **observations, not measured confusion counts**.

A higher overall accuracy did not eliminate every class-specific error. Variations in handwriting shape, touching characters, and segmentation matter as well as the choice of neural network.

## Interactive Gradio Demo

To show the complete pipeline, we developed an [**interactive handwriting-recognition application**](demo/) using the saved trained models.

The interface allows users to:

- **Draw letters or words** in the browser and see the recognized text.
- **Capture handwriting from a reMarkable tablet screen share** and read an entire page.
- **Choose between the DenseNet and MobileNetV2 checkpoints**, inspect annotated predictions, and download the recognized text as `my_text.txt`.

The [recognition pipeline](demo/recognizer.py) identifies candidate characters, normalizes strokes into EMNIST-style **28 × 28** images, and applies the input transforms required by each model. The [Gradio interface](demo/app.py) provides the drawing canvas and tablet screen-share workflow. The baseline CNN was evaluated in the notebooks but is **not** one of the selectable models in the current demo.

### Run the demo locally

From the repository root:

```bash
pip install -r demo/requirements.txt
python demo/app.py
```

Open **http://127.0.0.1:7860**. To create a temporary public Gradio link:

```bash
python demo/app.py --share
```

The public `gradio.live` link used during the presentation was temporary and may expire. The application code and the saved model weights remain in this repository; see [demo/README.md](demo/README.md) for more instructions.

## Repository Structure

```text
Preprocessing/       Data inspection, EDA, reproducible splits, and shared loaders
Baseline_Model/      Baseline CNN notebook and detailed performance report
models/DenseNet201/  DenseNet121 training/tuning, raw handwriting work, model and plots
models/MobileNet/    MobileNetV2 pretrained vs scratch experiments and checkpoints
demo/                Gradio interface, OpenCV recognition pipeline, run instructions
README.md            Project overview
```

## What We Learned and Limitations

- **Hyperparameter tuning helped, but is not a universal requirement.** DenseNet's Hyperband search helped us compare settings systematically; it is different from exhaustive grid search.
- **Checkpoint selection matters.** We monitored validation loss and used early stopping rather than automatically assuming the last epoch was best.
- **Transfer learning depends on the task.** The frozen ImageNet-trained MobileNetV2 features did not outperform training from scratch in this experiment.
- **Real handwriting is harder than a clean benchmark.** Thresholding, smoothing, segmentation, and consistent input formatting matter, while cursive or touching letters can remain difficult.
- **Benchmark results have limits.** The main test set was created from the original source-training portion, **not** the official complete EMNIST test split. Writer-independent splitting was not verified, test classes are relatively small, and the separate handwritten-page experiments were not a systematic generalization benchmark.

Overall, this project connected **data preparation → model development → evaluation → image preprocessing → an interactive application**, and showed why inspecting failures is as important as reporting accuracy.

## Contributors

- Nathan Butler
- Michael Sheehey
- Franklin Osualaaham
- Sarthakkumar Patel
