# Satellite Change Detection using U-Net

## Project Overview

This project was developed as part of the technical assessment for the **Satellite AI Research Intern** role at GalaxEye Space.

The objective of this assignment is to perform **binary change detection** on satellite imagery using pre-event and post-event image pairs. The model predicts regions where structural changes or building damage have occurred.

A lightweight U-Net-based semantic segmentation pipeline was implemented using TensorFlow and OpenCV.

---

# Dataset Description

The dataset contains:

- Pre-event satellite images
- Post-event satellite images
- Binary target masks

Directory structure:

```bash
dataset/
│
├── train/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
│
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
│
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```

Each target image is a binary segmentation mask:

- `0` → background
- `1` → changed region / building damage

---

# Methodology

## Model Architecture

A lightweight U-Net architecture was implemented for semantic segmentation.

### Input

- Concatenated pre-event and post-event images
- Total input channels: 6

### Output

- Binary segmentation mask

---

# Preprocessing Steps

The following preprocessing steps were applied:

- Images resized to `256 × 256`
- Pixel normalization (`0-1`)
- Concatenation of pre-event and post-event images
- Binary target mask processing

---

# Loss Functions

Multiple loss functions were experimented with:

## 1. Binary Cross Entropy (BCE)

Initial baseline approach.

## 2. Dice Loss

Used to address severe class imbalance.

## 3. Weighted BCE + Dice Loss

Implemented to penalize false negatives more strongly and mitigate foreground collapse.

---

# Challenges Encountered

The dataset presented several challenges:

- Severe class imbalance
- Sparse positive masks
- Foreground collapse during training
- Highly noisy SAR imagery
- CPU-only training constraints

The model frequently converged toward predicting background-only masks due to the dominance of negative pixels.

---

# Results

## Validation Metrics

| Metric | Score |
|---|---|
| Precision | 0.0 |
| Recall | 0.0 |
| F1 Score | 0.0 |
| IoU | 0.0 |

---

# Observations

Although the final model struggled to detect foreground regions effectively, the project demonstrates:

- Complete semantic segmentation pipeline
- Data preprocessing workflow
- U-Net implementation
- Experimentation with multiple loss functions
- Model evaluation pipeline
- Error analysis and debugging workflow

The assignment was approached as a research-oriented experimentation task rather than purely an optimization benchmark.

---

# Project Structure

```bash
AI_Assignment/
│
├── dataset/
│
├── notebooks/
│   └── segmentation.ipynb
│
├── models/
│   ├── change_detection_model.h5
│   └── change_detection_model.keras
│
├── outputs/
│
├── train.py
├── eval.py
├── config.yaml
├── requirements.txt
└── README.md
```

---

# Requirements

- Python 3.10+
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

# Installation

## Step 1 — Create Conda Environment

```bash
conda create -n galaxeye python=3.10 -y
```

## Step 2 — Activate Environment

```bash
conda activate galaxeye
```

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Run training using:

```bash
python train.py
```

The trained model will automatically be saved inside:

```bash
models/
```

---

# Evaluation

Run evaluation using:

```bash
python eval.py
```

This script computes:

- Precision
- Recall
- F1 Score
- IoU
- Confusion Matrix

---

# Model Weights

Model weights can be accessed here:

(Add your Google Drive or Hugging Face model link here)

---

# Future Improvements

Potential future improvements include:

- Attention U-Net
- Deep supervision
- Transformer-based architectures
- Advanced augmentation
- Focal Loss
- Better handling of SAR noise
- Training on GPU infrastructure
- Multi-scale feature extraction

---

# References

1. U-Net: Convolutional Networks for Biomedical Image Segmentation  
   https://arxiv.org/abs/1505.04597

2. TensorFlow Documentation  
   https://www.tensorflow.org/

3. OpenCV Documentation  
   https://opencv.org/

4. Scikit-learn Documentation  
   https://scikit-learn.org/

---

# Author

Aysha Afrah

Technical Assignment Submission for GalaxEye Space