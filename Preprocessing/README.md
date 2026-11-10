# EMNIST Handwritten Letter Recognition

**Deep learning | Data preprocessing & model comparison**

> **Project status — Work in progress**  
> I'm still finishing the project write-up and organizing the code. The preprocessing notebook, Python files, and shared image loader will be added as I finalize them. I'm also checking that all exported image folders and files have transferred correctly to Google Drive.

## Project overview
 We're building models to recognize handwritten English letters from images. We plan to compare **three deep learning architectures**:

| Model | Role |
|---|---|
| Fully connected Artificial Neural Network (ANN) | Baseline |
| Convolutional Neural Network (CNN) — Architecture 1 | First image-based model |
| Convolutional Neural Network (CNN) — Architecture 2 | Second image-based model |

Since I am responsible for **data preprocessing**. My goal was to prepare one consistent dataset so everyone on the team could train and evaluate their models using the **same images, labels, and splits** rather than downloading and splitting the data independently.

### Dataset at a glance

| Item | Details |
|---|---|
| Source | [Voxel51 / EMNIST Letters Tiny](https://huggingface.co/datasets/Voxel51/emnist-letters-tiny) |
| Images | 10,000 |
| Image size | 28 × 28 pixels |
| Format | Grayscale |
| Classes | 26 letters (A–Z) |
| Labels | Uppercase and lowercase writing styles share the same letter class |

---

## My preprocessing workflow

### 1. Load and inspect the data

I loaded the dataset using **FiftyOne** and checked the image dimensions, labels, and source train/test tags. All **10,000 images loaded**, with **no rejected records**. I also converted the images to grayscale and standardized the labels to **0–25** (`A = 0`, `Z = 25`) so the models would use the same label mapping.

### 2. Check image orientation and duplicates

I displayed sample letters to check whether the images were correctly oriented. After inspecting them, I kept the existing orientation rather than applying a transpose.

I also used image hashes to check for exact duplicates. **No exact duplicates were found.** This matters because putting identical images into different splits could make model results misleading.

### 3. Investigate and fix the dataset split issue

One issue stood out during exploration: the dataset's source-tagged **train** subset contained all 26 letters, but its source-tagged **test** subset contained only **A–S**. The letters **T–Z were missing** from that subset.

Instead of using that incomplete source test subset as our main evaluation set, I created new **stratified train, validation, and test splits** using the 8,547 source-training images. I used `random_state = 42` to make the split reproducible.

| Split | Images | Letter classes | Purpose |
|---|---:|---:|---|
| **Training** | 6,837 | 26 | Learn the letter patterns |
| **Validation** | 855 | 26 | Tune and compare models during development |
| **Test** | 855 | 26 | Final held-out model evaluation |
| Reserved source test | 1,453 | 19 | Optional additional evaluation (A–S only) |

**Important:** Our main 855-image test split is held out from the source-training subset; it is **not** the official EMNIST test partition. I kept the 1,453 source-test images separate rather than mixing them into model training.

### 4. Explored data quality and handwriting variation

I visualized five examples of each letter across the main splits and created **class-distribution charts, heatmaps, and pixel-intensity visualizations**.

A major observation was how differently people write the same letter. Some examples were hard to distinguish even by eye:

- **U / V / Y:** Similar handwritten shapes
- **R / 8:** Some R examples resemble 8
- **G / Q / 9:** Some G examples resemble Q or the number 9
- **L / I / 1:** Some lowercase L examples resemble I or 1
- **Q / P** and **C / O:** Similar-looking forms in some samples
- **J and Z:** Noticeable differences in writing style, stroke thickness, and intensity

These are **potential ambiguities, not confirmed labeling mistakes**. I chose **not** to automatically delete or relabel these images because unusual handwriting can still be valid. I documented the observations so we can revisit them when evaluating model errors.

I also checked for blank images, unusually faint images, very low foreground-pixel counts, and overlapping split indices. **The checks did not flag any issues in this run.** That does **not** mean every image label has been manually verified.

### 5. Normalize the image data

Each grayscale pixel originally had a value from **0 to 255**. I divided the pixel values by `255.0` to bring them into the **0–1 range**, which gives the neural networks a consistent input scale.

I exported the normalized arrays as `.npz` files and also generated individual `.png` images organized by split and letter. The PNGs retain their 0–255 grayscale values; the NPZ arrays are **already normalized**.

---

## Shared dataset and file formats

**Google Drive folder:** `CLEANED IMAGES AND REPRESENTATIONS`  
*Team sharing link: to be added.*

The shared dataset is intended to include:

```text
CLEANED IMAGES AND REPRESENTATIONS/
├── train.npz
├── val.npz
├── test.npz
├── original_test_optional.npz
├── images_png/
│   ├── train/
│   │   ├── A/ ... Z/
│   ├── val/
│   │   ├── A/ ... Z/
│   └── test/
│       ├── A/ ... Z/
├── manifest.csv
├── class_distribution.csv
├── preprocessing_config.json
└── quality_findings.txt
```

The `.npz` files contain the **image arrays and labels**, so they do not need to be converted back into pictures for model training. The `images_png` folders provide individual image files for visual inspection or image-folder loaders. Both formats represent the **same fixed data splits**.

> **Upload check:** The PNG folders were being rechecked after an incomplete transfer to Drive. Please use the dataset only after all three splits and their A–Z folders have been verified.

## How teammates should use the data

All three models must use the provided **train / val / test** assignments. **Please do not create a new split.**

- **ANN baseline:** Flatten each 28 × 28 image into **784 features**.
- **CNN models:** Keep the spatial structure at **28 × 28 × 1** (or use channel-first shape for PyTorch).
- **Using NPZ:** Images are already normalized to **0–1**; **do not divide by 255 again**.
- **Using PNG:** Normalize the 0–255 pixel values **once** in the image loader.
- **Model evaluation:** Keep the test set out of training and hyperparameter tuning.

A separate **shared image-loader notebook** will be uploaded to make it easier for everyone to load the same files.

## Limitations and next steps

The visual review showed some ambiguous handwriting, but I did not manually verify every single label. I also did not establish whether writers are fully separated across the splits. These limitations should be considered when interpreting our model results.

No data augmentation was included in the shared preprocessing. If the team experiments with augmentation, it should be applied **only to training data**.

Once the models are trained, we plan to compare **accuracy, precision, recall, F1 score, and confusion matrices**, and test how well they recognize **new handwriting collected by our team**.

---

## Final note

My main goal was to give the team a **consistent, documented, reusable dataset**. Having one shared preprocessing pipeline and fixed data splits should make our ANN-versus-CNN comparison easier to reproduce and more meaningful.

*README will be updated as the model notebooks, loader, evaluation results, and project report are completed.*
