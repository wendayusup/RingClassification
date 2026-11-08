"""
STAGE 4: SCIENTIFIC MODEL EVALUATION & TESTING
-----------------------------------------------
Evaluates the best-trained Pure CNN model on the unseen Test Dataset (72 images):
1. Loads optimal model checkpoint from saved_models/pure_cnn_best.pth.
2. Performs deterministic inference on the Test set (36 defect vs 36 normal).
3. Computes industrial-standard metrics:
   - Accuracy (%)
   - Precision, Recall, and F1-Score per class
   - Confusion Matrix (TP, FP, TN, FN)
4. Generates and saves a Confusion Matrix heatmap to results/04_confusion_matrix.png.
5. Prints detailed breakdown of any misclassified samples for inspection.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Ensure project root is in sys.path for robust package imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step01_data_preparation.dataset_loader import get_data_loaders
from step02_model_architecture.pure_cnn import PureCNN

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = PROJECT_ROOT / "saved_models" / "pure_cnn_best.pth"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model():
    """
    Main evaluation pipeline: runs test data through the saved model
    and exports quantitative performance metrics.
    """
    print("=== [STAGE 4] MODEL EVALUATION ON UNSEEN TEST DATASET ===")
    
    if not CHECKPOINT_PATH.exists():
        print(f"[ERROR] Checkpoint not found at: {CHECKPOINT_PATH}")
        print("Please train the model first by running Stage 3: train.py")
        return

    # 1. Load Test Dataset
    _, _, test_loader, classes = get_data_loaders(batch_size=16, image_size=(128, 128))
    total_test_images = len(test_loader.dataset)
    print(f"Loaded Test Samples: {total_test_images} images across {len(classes)} classes: {classes}")

    # 2. Reconstruct Model and Load Best Weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = PureCNN(in_channels=1, num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded weights saved from Epoch: {checkpoint['epoch']} (Best Val Acc: {checkpoint['val_accuracy']:.2f}%)\n")

    # 3. Perform Test Inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 4. Compute Performance Metrics
    total_correct = (all_preds == all_labels).sum()
    test_accuracy = (total_correct / total_test_images) * 100.0

    print("----------------------------------------------------------------------")
    print(f"Overall Test Accuracy: {total_correct}/{total_test_images} ({test_accuracy:.2f}%)")
    print("----------------------------------------------------------------------\n")

    print("Detailed Classification Report (Precision, Recall, F1-Score):")
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print(report)

    # 5. Confusion Matrix Calculation
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(f"                Predicted: {classes[0]} | Predicted: {classes[1]}")
    print(f"Actual: {classes[0]:<12} [ {cm[0][0]:^16} | {cm[0][1]:^16} ]")
    print(f"Actual: {classes[1]:<12} [ {cm[1][0]:^16} | {cm[1][1]:^16} ]\n")

    # 6. Plot and Save Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Test Dataset", fontsize=12, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=15)
    plt.yticks(tick_marks, classes)

    # Annotate cell values inside heatmap
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold"
            )

    plt.ylabel("Ground Truth (Actual)")
    plt.xlabel("Model Prediction")
    plt.tight_layout()

    cm_plot_path = RESULTS_DIR / "04_confusion_matrix.png"
    try:
        plt.savefig(cm_plot_path, dpi=150)
        print(f"[INFO] Confusion Matrix heatmap saved to: {cm_plot_path}")
    except PermissionError:
        alt_path = RESULTS_DIR / "04_confusion_matrix_latest.png"
        plt.savefig(alt_path, dpi=150)
        print(f"[WARNING] 04_confusion_matrix.png is open in an image viewer. Saved to: {alt_path}")
    plt.close()

    print("=== [STAGE 4] SUCCESS: Model evaluation complete ===")


if __name__ == "__main__":
    evaluate_model()
