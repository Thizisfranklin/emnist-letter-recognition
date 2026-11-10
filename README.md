# EMNIST Handwritten Character Recognition

**Computer Vision | Deep Learning | Handwritten Character Classification**

>This project focuses on recognizing handwritten English letters using deep learning. We are using a 10,000-image subset of the EMNIST dataset to compare different neural network architectures and evaluate their performance.

**Project Status:** Data Preprocessing Completed | Model Development in Progress

[Cleaned and Preprocessed Dataset — Google Drive](https://drive.google.com/drive/folders/1Lb3EGMDdHWG85wYMiCmRS2rJZSia8MKU?usp=sharing)

## Overview

MNIST is a widely used benchmark for handwritten-digit recognition. EMNIST extends this benchmark to handwritten letters and digits using data from NIST Special Database 19.

Our project uses the **EMNIST Letters Tiny dataset**, containing 10,000 grayscale images (28 × 28 pixels) across 26 letter classes (A–Z). Uppercase and lowercase handwriting styles are grouped under the same class labels.

During preprocessing, we discovered that the supplied test subset contained only 19 of the 26 classes. To address this, we reserved the original test subset and created new stratified training, validation, and testing splits from the source-training data.

We will compare three deep learning architectures:

- **Artificial Neural Network (ANN/MLP):** Our baseline model.
- **Convolutional Neural Network (CNN 1):** Our first CNN architecture.
- **Convolutional Neural Network (CNN 2):** A second CNN architecture for comparison.

Our evaluation will focus on classification performance, particularly accuracy, precision, recall, F1 score, and confusion matrices. We also plan to test the models on independently collected handwriting to investigate how well they generalize beyond the EMNIST subset.

## Research Focus

Beyond comparing the ANN and CNN architectures, we also want to investigate how well the models generalize to handwriting outside the EMNIST dataset.

We plan to collect handwritten letters from our team members and additional volunteers with different handwriting styles. Each participant will write uppercase and lowercase letters (A–Z), with multiple samples per letter. Both uppercase and lowercase versions will belong to the same class, giving us 26 classes in total.

This independently collected dataset will serve as an additional test set, allowing us to compare model performance on EMNIST against handwriting from new writers and examine which characters are most frequently misclassified.

## Project Progress

| Stage | Status |
|---|---|
| Data preprocessing and EDA | Completed |
| ANN and CNN development | In progress |
| Model evaluation and comparison | Upcoming |
| External handwriting testing | Planned |

## Preprocessing and Shared Dataset

The preprocessing folder contains the preprocessing notebook/code, documentation, and shared image loader.

The cleaned dataset is available through the Google Drive link above. All three models will use the same training, validation, and testing splits to ensure a consistent comparison.

## Contributors

- Nathan Butler
- Michael Sheehey
- Franklin Osualaaham
- Joseph Uzoigwe
- Sarthakkumar Patel
