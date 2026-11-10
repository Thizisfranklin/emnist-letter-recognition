# EMNIST Handwritten Letter Recognition — Data Preprocessing

## Overview

This notebook covers my data preprocessing and exploratory data analysis for our handwritten letter recognition capstone. We're using the [EMNIST Letters Tiny dataset](https://huggingface.co/datasets/Voxel51/emnist-letters-tiny) to compare an ANN baseline against two CNN architectures.

My responsibility was to prepare a consistent dataset that everyone could use for training and evaluation without having to repeat preprocessing or create separate data splits.

**Dataset:** 10,000 grayscale images (28 × 28), representing 26 letter classes (A–Z). Uppercase and lowercase variations belong to the same class.

## Preprocessing Workflow

### 1. Data Loading and Validation

I loaded the dataset using FiftyOne and inspected the images, labels, and source tags.

- All 10,000 images loaded successfully, with no rejected records.
- Converted images to grayscale and standardized labels from 0–25 (`A=0`, `Z=25`).
- Checked image orientation and retained the original orientation.
- Checked for exact duplicates using SHA-256 hashes; none were found.

### 2. Train, Validation, and Test Splitting

During EDA, I noticed that the source-tagged test subset only contained 19 classes (A–S), while the source training subset contained all 26.

Since our models need to recognize A–Z, I created new stratified splits from the 8,547 source-training images using `random_state=42`.

| Split | Images | Classes |
|---|---:|---:|
| Train | 6,837 | 26 |
| Validation | 855 | 26 |
| Test | 855 | 26 |
| Reserved original test | 1,453 | 19 |

I kept the original test subset separate for possible additional evaluation. Our main test set is a newly held-out portion of the source-training data, not the official EMNIST test split.

### 3. Exploratory Data Analysis and Quality Checks

I explored the class distributions, visualized samples from each letter, and compared pixel intensities across the splits.

I also calculated per-class counts, minimum and maximum frequencies, range, standard deviation, and imbalance ratios to assess class balance.

Some characters were visually ambiguous, particularly:

- **U/V/Y** and **R/V**
- **G/Q/9** and **L/I/1**
- **Q/P** and **C/O**

There were also variations in stroke thickness and brightness.

I retained these samples because unusual handwriting doesn't necessarily indicate incorrect labeling. Automatically removing them could introduce bias or eliminate valid writing styles.

I also checked for blank images, unusually faint images, low foreground-pixel counts, and overlapping split indices. No issues were flagged by these checks.

These checks do not guarantee that every label is correct, since I did not manually verify all 10,000 images.

### 4. Normalization and Export

I normalized the grayscale pixel values from `[0, 255]` to `[0, 1]` using:

```python
X = X.astype(np.float32) / 255.0
```

The final dataset was exported in two formats:

| Format | Contents | Use |
|---|---|---|
| `.npz` | Normalized image arrays and labels | Model training and evaluation |
| `.png` | Individual grayscale images arranged by split and letter | Visualization and optional image-folder loading |

The NPZ images have shape `(N, 28, 28, 1)` and are already normalized. No additional normalization is required.

## Shared Dataset

The processed dataset is available in our Google Drive folder:

**`CLEANED IMAGES AND REPRESENTATIONS`**

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

I also created a shared image-loader notebook for loading the data into ANN and CNN models.

**For model development:**
- Use the provided `train.npz`, `val.npz`, and `test.npz` files.
- Keep the existing splits and label mapping.
- ANN models can flatten images into 784 features.
- CNN models can use the image dimensions directly, with channel ordering adjusted if necessary.
- Do not normalize the NPZ images again.
- Keep the test set separate from training and hyperparameter tuning.

## Limitations

- Some letters are visually ambiguous, and individual labels were not all manually verified.
- Writer-independent splitting could not be confirmed.
- The reserved source-test subset contains only 19 classes.
- No augmentation was performed during preprocessing. Any augmentation should be applied to training data only.

## Summary

The preprocessing stage is complete. I validated the dataset, investigated class distributions and handwriting variations, addressed the incomplete source-test class coverage, normalized the images, and exported fixed splits for the team.

The main goal was to ensure that all three architectures use the same dataset so their results can be compared fairly.

**Next:** Model training and evaluation, including per-class performance, confusion matrices, and potentially testing on independently collected handwriting.
