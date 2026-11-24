# EMNIST Handwritten Letter Recognition — Preprocessing & EDA

**My contribution to our Computer Vision capstone | Data validation • Exploratory analysis • Reproducible model-ready splits**

This folder documents the **data preprocessing and exploratory data analysis I completed** for our team's [Handwritten Letter Recognition with Deep Learning](../README.md) project. Our goal was to compare deep learning models that recognize handwritten English letters and then demonstrate their predictions on handwritten input.

My main responsibility was to prepare **one reliable, consistent dataset** that the team could reuse for training and evaluation. This included identifying an issue with the dataset's original test partition, inspecting handwriting variation, creating stratified splits, normalizing image arrays, and organizing the files for the model-development work.

**Status: Completed.** The team subsequently trained a baseline CNN, DenseNet121, and MobileNetV2 (pretrained and from scratch), evaluated the models, experimented with additional photographed handwriting, and built a Gradio demo.

## The Dataset

**Source:** [Voxel51 / EMNIST Letters Tiny](https://huggingface.co/datasets/Voxel51/emnist-letters-tiny)

- **10,000** handwritten character images
- **28 × 28** grayscale pixels per image
- **26 classes**, labeled A–Z (integer labels 0–25)
- Uppercase and lowercase writing styles grouped under the same letter class

These are small, isolated handwritten characters—not full handwritten words or document pages. Preparing the data consistently was important because different models use different input representations.

## My Preprocessing and EDA Workflow

### 1. Load, inspect, and validate the images

Using **Python and FiftyOne**, I loaded the data and reviewed the images, labels, and supplied source tags.

I:
- Verified that all **10,000** images loaded.
- Standardized the class labels to **0–25** (`A=0`, `Z=25`).
- Inspected orientation and preserved the original image orientation.
- Checked for exact duplicate image content using **SHA-256** hashes; no exact duplicates were found.
- Investigated blank or faint images, foreground-pixel counts, and overlapping split indices.

These automated checks did not flag major problems, but they cannot guarantee that every image has a correct human-readable label.

### 2. Find and address the original test-set problem

One of the most important findings came from comparing the source-provided partitions. Although the dataset covers all **26 letters**, the supplied **1,453-image source-test partition contained only A–S**; it was missing **T–Z**.

That made it unsuitable as the **primary all-class test set** for our 26-class project.

Rather than mix the existing source-test data into training, I **kept it separate** and created three stratified modeling splits from the **8,547 source-training images** using `random_state=42`.

| Dataset | Images | Letter classes | Use |
|---|---:|---:|---|
| Training | 6,837 | 26 | Fit model parameters |
| Validation | 855 | 26 | Model selection and early stopping |
| Main test | 855 | 26 | Consistent final model evaluation |
| Reserved original source test | 1,453 | 19 (A–S) | Separate supplementary checks; not the main 26-class evaluation |

All three modeling splits share the same A–Z label mapping and similar class proportions. **Our main test set is a newly held-out portion of the source-training data, not the official full EMNIST benchmark test set.**

### 3. Explore class balance and handwriting quality

I examined class counts, per-class proportions, sample images, and grayscale foreground intensity distributions.

- The training split had **238 images of G** (smallest class) and **286 images of Q** (largest class), a largest-to-smallest ratio of approximately **1.20**.
- Class imbalance was **modest**, and stratification kept class proportions similar in the training, validation, and main test splits.
- Letter shapes varied in brightness, thickness, and writing style, even within the same class.
- Some samples looked ambiguous, especially **U/V/Y**, **R/V**, **G/Q**, **L/I**, **Q/P**, and **C/O**.

I **kept the ambiguous samples** rather than automatically treating unusual writing as incorrect labels. A difficult-to-read character can still be a valid example of handwriting; removing it without verification could make the modeling task artificially easier.

This analysis also helped set up an important evaluation question: **Would the models continue to confuse visually similar letters even when overall test accuracy was high?**

### 4. Normalize and export model-ready files

I converted the images to `float32` and scaled their grayscale pixel intensities from **[0, 255]** to **[0, 1]**:

```python
X = X.astype(np.float32) / 255.0
```

I exported:
- **`.npz` arrays:** normalized images with shape `(N, 28, 28, 1)`, integer labels, and sample identifiers for training and evaluation.
- **Grayscale `.png` images:** arranged by split and letter class to make the dataset easy to inspect and load as images.
- **Metadata and summary files:** class distributions, a manifest, preprocessing configuration, and quality findings.

I also created a [shared image-loader notebook](Shared_Image_Loader_.ipynb) to help teammates read the prepared images without repeating preprocessing.

## Shared Dataset and Code

**Dataset folder:** [CLEANED IMAGES AND REPRESENTATIONS — Google Drive](https://drive.google.com/drive/folders/1Lb3EGMDdHWG85wYMiCmRS2rJZSia8MKU?usp=sharing)

```text
CLEANED IMAGES AND REPRESENTATIONS/
├── train.npz
├── val.npz
├── test.npz
├── original_test_optional.npz
├── images_png/
│   ├── train/A-Z/
│   ├── val/A-Z/
│   └── test/A-Z/
├── manifest.csv
├── class_distribution.csv
├── preprocessing_config.json
└── quality_findings.txt
```

**Files in this GitHub folder:**
- [`EMNIST_Capstone_Preprocessing_Done.ipynb`](EMNIST_Capstone_Preprocessing_Done.ipynb) — data preparation, validation, visualizations, and exports.
- [`Shared_Image_Loader_.ipynb`](Shared_Image_Loader_.ipynb) — loading images for downstream modeling.
- [`README.md`](README.md) — this walkthrough.

### Notes for reusing the data

1. Use the **same** `train.npz`, `val.npz`, and `test.npz` partitions for fair model comparisons.
2. **Do not divide the NPZ pixels by 255 again**; they are already normalized.
3. Keep the test partition separate from model fitting and hyperparameter tuning.
4. Apply any model-specific resizing, channel conversion, or ImageNet normalization as needed **after loading the shared dataset**, without changing its split membership.
5. If applying data augmentation, keep augmented examples within the **training** partition only.

## How My Work Supported the Final Models

The shared dataset was the starting point for the team's subsequent experiments:

| Team experiment | Result on our 855-image held-out test set |
|---|---:|
| Baseline CNN | **88.42%** accuracy |
| Fine-tuned DenseNet121 | **~88.9%** accuracy |
| MobileNetV2 with frozen pretrained features | **80.70%** accuracy |
| MobileNetV2 trained from scratch | **90.29%** accuracy |

The evaluation also revisited ambiguities noticed during preprocessing. For example, the baseline CNN frequently predicted **Q as G** and **I as L**. Even the best overall model continued to have weaker performance on some lookalike letters.

The team went a step further by experimenting with **separate photographed handwritten pages**. That work included additional **OpenCV thresholding, Gaussian blur, and bilateral filtering** for raw-image processing. These steps were **part of later model-specific and demo experiments**, not part of the common EMNIST preprocessing pipeline described in this folder.

We also presented an [interactive Gradio demo](../demo/) that lets users draw handwriting or capture a reMarkable tablet screen share. The app uses further character segmentation and model-specific transformations to turn raw handwriting into predictions.

I include these final outcomes to show **how the prepared dataset was used**; they were the result of the **team's modeling and application work**, not models I claim to have developed in this preprocessing notebook.

## Limitations and Takeaways

- **Test-set coverage matters:** the original source-test partition did not contain all 26 letters, so I explicitly checked label coverage before model training.
- **Consistent data splits matter:** using the same fixed partitions made model results easier to compare.
- **Data quality involves judgment:** visually ambiguous letters were not automatically mislabeled; I retained them and documented the uncertainty.
- **Writer independence was not confirmed:** the stratified splits preserve class proportions but do not establish separation by handwriting author.
- **The 855-image test split is small:** per-class metrics can vary, and these results should not be presented as official EMNIST benchmark scores.
- **Raw handwriting introduces new challenges:** photographed pages, connected strokes, and cursive styles are different from isolated, normalized EMNIST letters. The team's external handwriting examples were exploratory, not a standardized quantitative benchmark.

**My biggest takeaway:** data preprocessing is not simply resizing or normalizing images. Investigating the dataset, finding the missing classes in the original test split, and building a reusable, reproducible pipeline directly shaped how the team could train, evaluate, and demonstrate its models.

For the full project, model notebooks, evaluation, and demo instructions, see the [main project README](../README.md).
