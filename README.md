# Industrial Bearing Defect Inspection System (Ring Classification)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

> An end-to-end Computer Vision and Deep Learning system powered by a **Pure Convolutional Neural Network (CNN)**, integrated with an **Explainable AI (Grad-CAM)** and **Metrology Filtering Suite** for real-time industrial bearing ring defect classification and quality control inspection.

---

## Table of Contents
1. [Overview and Key Features](#overview-and-key-features)
2. [Project Pipeline and Directory Structure](#project-pipeline-and-directory-structure)
3. [Model Architecture (7-Stage Pure CNN)](#model-architecture-7-stage-pure-cnn)
4. [Explainable AI (XAI) and Defect Localization](#explainable-ai-xai-and-defect-localization)
5. [Installation and Prerequisites](#installation-and-prerequisites)
6. [Execution Guide](#execution-guide)
7. [Evaluation and Experimental Results](#evaluation-and-experimental-results)
8. [Push to GitHub Guide (HTTPS)](#push-to-github-guide-https)

---

## Overview and Key Features

In precision manufacturing, micro-defects on bearing surfaces (such as scratches, cracks, impact indentations, and surface deformities) can cause catastrophic mechanical failure in high-speed rotary equipment.

This project delivers an automated visual inspection solution designed for industrial quality control:
- **Zero Data Leakage Pipeline**: Partitioning strategy (Train 71%, Val 12.5%, Test 16.5%) strictly grouped by unique physical sample IDs (Sample 01-24) to guarantee rigorous and uncompromised evaluation.
- **Offline Data Augmentation**: Systematic geometry and illumination variations (90 deg, 180 deg, 270 deg rotations, vertical/horizontal reflections, brightness scaling) increasing the base dataset by 9x per class.
- **Custom Pure CNN Architecture**: A 7-stage custom convolutional neural network engineered from scratch without pre-trained transfer learning weights, featuring Batch Normalization and Dropout (0.4) for strong generalization.
- **Dynamic Optimization**: Training loop utilizing the Adam Optimizer, L2 Regularization (Weight Decay), and an adaptive Learning Rate Scheduler (ReduceLROnPlateau).
- **Explainable AI (XAI) and Metrology**: Fusion of Gradient-weighted Class Activation Mapping (Grad-CAM), Morphological Black-Hat transform, and Sobel Gradient filtering for objective defect localization and coordinate mapping.
- **Interactive Web Inspection Dashboard**: Modern Flask web application supporting live webcam capture, drag-and-drop file upload, test set presets, side-by-side QC red-dot defect comparison, and real-time inference latency tracking.

---

## Project Pipeline and Directory Structure

```text
RingClassification/
├── .vscode/                     # IDE workspace and Python analysis settings
├── dataset_raw/                 # Raw image captures and offline augmentations
│   ├── defect_ring/             # Defective bearing ring images (216 images)
│   └── normal_ring/             # Normal bearing ring images (216 images)
├── dataset_split/               # Structured subsets isolated by sample ID
│   ├── train/                   # 153 defect + 153 normal (IDs 01 - 17)
│   ├── val/                     # 27 defect + 27 normal   (IDs 18 - 20)
│   └── test/                    # 36 defect + 36 normal   (IDs 21 - 24)
├── onceCoding/                  # Initial dataset preparation scripts
│   ├── augment_and_rename.py    # Standardized renaming and geometric augmentation
│   └── split_dataset.py         # Leakage-free subset distribution
├── results/                     # Visual evaluation plots, curves, and matrices
│   ├── 01_data_loader_preview.png
│   ├── 03_training_history.png
│   ├── 04_confusion_matrix.png
│   └── 05_inference_result.png
├── saved_models/                # Optimal model checkpoint storage
│   └── pure_cnn_best.pth        # Saved Pure CNN checkpoint (~24 MB)
├── step01_data_preparation/     # [STAGE 1] Data transformation and PyTorch loaders
│   ├── __init__.py
│   └── dataset_loader.py
├── step02_model_architecture/   # [STAGE 2] Pure CNN architecture implementation
│   ├── __init__.py
│   └── pure_cnn.py
├── step03_training/             # [STAGE 3] Training loop and adaptive optimization
│   └── train.py
├── step04_evaluation/           # [STAGE 4] Quantitative evaluation on unseen test set
│   └── evaluate.py
├── step05_inference/            # [STAGE 5] Inference engine and XAI localizer
│   ├── __init__.py
│   ├── inference.py
│   └── defect_localizer.py
├── step06_web_app/              # [STAGE 6] Industrial web dashboard (Flask)
│   ├── __init__.py
│   ├── app.py
│   └── templates/
│       └── index.html
├── run_web_app.py               # Application launcher
├── requirements.txt             # Python package dependencies
├── .gitignore                   # Git exclusion rules
└── README.md                    # Project documentation
```

---

## Model Architecture (7-Stage Pure CNN)

The model employs a sequential convolutional structure designed to capture structural edge contours, surface texture irregularities, and localized defect patterns:

```text
[Input Grayscale Tensor: 1 x 128 x 128]
       │
       ▼
[BLOCK 1] Conv2D (1 -> 16, k=3, p=1) + BatchNorm2D + ReLU + MaxPool2D(2x2)
       │  Output: (16 x 64 x 64) -> Extracts low-level edges and ring contours
       ▼
[BLOCK 2] Conv2D (16 -> 32, k=3, p=1) + BatchNorm2D + ReLU + MaxPool2D(2x2)
       │  Output: (32 x 32 x 32) -> Extracts mid-level surface curvatures and textures
       ▼
[BLOCK 3] Conv2D (32 -> 64, k=3, p=1) + BatchNorm2D + ReLU + MaxPool2D(2x2)
       │  Output: (64 x 16 x 16) -> Encodes high-level anomaly and defect patterns
       ▼
[Stage 5: Flatten] 64 x 16 x 16 = 16,384 1D feature representations
       │
       ▼
[Stage 6: Dense / Fully Connected] Linear(16384 -> 128) + ReLU + Dropout(p=0.40)
       │
       ▼
[Stage 7: Output Layer] Linear(128 -> 2) -> Unnormalized Logits [Defect, Normal]
```

---

## Explainable AI (XAI) and Defect Localization

To ensure transparency and decision auditability, Stage 5 integrates a Unified Objective Industrial Metrology Suite:
1. **Mathematical Morphological Black-Hat**: `Closing(I) - I` isolates dark structural micro-scratches and pits from brighter metal surfaces.
2. **Sobel Gradient Magnitude**: Computes spatial intensity derivatives to highlight sharp localized edge discontinuities.
3. **Gradient-weighted Class Activation Mapping (Grad-CAM)**: Backpropagates defect-class gradients into the final convolutional feature maps (`conv3`) to generate coarse heatmaps of influential regions.
4. **Automated QC Annotation**: Fuses multi-layer activation peaks and annotates coordinate clusters on the original image with red indicator dots for operator inspection.

---

## Installation and Prerequisites

### System Requirements
- **Python**: Version 3.10 or higher
- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS
- **Hardware**: CPU (Intel/AMD) or NVIDIA GPU with CUDA support (optional)

### Setup Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd RingClassification
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Windows (Command Prompt / PowerShell)
   python -m venv venv
   venv\Scripts\activate

   # Linux / macOS / Git Bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Execution Guide

### 1. Launch the Web Inspection Dashboard
Run the primary web application launcher:
```bash
python run_web_app.py
```
Open a browser and navigate to: `http://127.0.0.1:5000`

Dashboard capabilities:
- File Upload: Drag-and-drop or select any bearing ring image.
- Live Webcam Inspection: Take instant high-resolution snapshots directly via web camera.
- One-Click Presets: Test directly against unseen test set samples.
- Visual Comparison: Side-by-side inspection between raw capture and red-dot QC annotated image with real-time confidence scores and latency metrics.

---

### 2. Standalone Pipeline Execution (CLI)

- **Stage 1: DataLoader Verification and Batch Preview**
  ```bash
  python step01_data_preparation/dataset_loader.py
  ```

- **Stage 2: Model Architecture Validation**
  ```bash
  python step02_model_architecture/pure_cnn.py
  ```

- **Stage 3: Model Training Loop**
  ```bash
  python step03_training/train.py
  ```
  The optimal model weights are automatically saved to `saved_models/pure_cnn_best.pth`, and the training convergence plot is saved to `results/03_training_history.png`.

- **Stage 4: Quantitative Test Evaluation**
  ```bash
  python step04_evaluation/evaluate.py
  ```

- **Stage 5: Standalone Inference and XAI Metrology**
  ```bash
  python step05_inference/inference.py
  ```

---

## Evaluation and Experimental Results

All generated performance visual artifacts are stored in the `results/` directory:
- `results/01_data_loader_preview.png`: Batch tensor verification grid (128x128 grayscale).
- `results/03_training_history.png`: Multi-epoch training vs validation loss and accuracy curves.
- `results/04_confusion_matrix.png`: Heatmap confusion matrix computed on 72 independent test set samples.
- `results/05_inference_result.png`: Multi-layer metrology breakdown (Black-Hat, Sobel, Grad-CAM, Red Dot Annotation).

---

## Push to GitHub Guide (HTTPS)

Follow these terminal commands to initialize and push this project to a remote GitHub repository over HTTPS:

### Step 1: Create a GitHub Repository
1. Navigate to GitHub and create a new repository (e.g., `RingClassification`).
2. Leave "Add a README file", "Add .gitignore", and "Choose a license" unchecked.
3. Copy the repository HTTPS URL (e.g., `https://github.com/<username>/RingClassification.git`).

---

### Step 2: Run Git Commands in Terminal / Git Bash

Open your terminal in the `RingClassification` root directory and execute:

```bash
# 1. Initialize git repository
git init

# 2. Configure your Git user identity
git config user.name "Your Name"
git config user.email "your_email@example.com"

# 3. Increase HTTP post buffer to ensure smooth upload of model weights (24 MB)
git config http.postBuffer 524288000

# 4. Stage all project files
git add .

# 5. Commit staged files
git commit -m "feat: complete bearing defect inspection system with Pure CNN, XAI, and Web Dashboard"

# 6. Ensure default branch is set to main
git branch -M main

# 7. Add remote GitHub repository
git remote add origin https://github.com/<YOUR_USERNAME>/RingClassification.git

# 8. Push to GitHub
git push -u origin main
```

---

### Authentication Note (GitHub Personal Access Token)
GitHub requires a Personal Access Token (PAT) when pushing over HTTPS:
1. Generate a token on GitHub:
   - Go to: **Settings** -> **Developer settings** -> **Personal access tokens** -> **Tokens (classic)**
   - Click **Generate new token (classic)**
   - Enter a note and check the **`repo`** scope
   - Click **Generate token** and copy the generated string (`ghp_xxxxxxxxxxxxxxxxxxxx`).
2. When prompted by git:
   - **Username**: Enter your GitHub username.
   - **Password**: Paste the Personal Access Token (PAT).
