"""
STAGE 3: MODEL TRAINING & OPTIMIZATION PIPELINE
------------------------------------------------
Trains the Pure CNN architecture with strict reproducibility and stability:
1. Sets deterministic random seeds across PyTorch and NumPy for consistent results.
2. Loads batched Train and Validation data from Stage 1 (dataset_loader).
3. Initializes the Pure CNN model with Batch Normalization from Stage 2 (pure_cnn).
4. Employs Cross-Entropy Loss, Adam Optimizer with L2 Regularization (Weight Decay),
   and a dynamic Learning Rate Scheduler (ReduceLROnPlateau).
5. Tracks both Validation Loss and Accuracy to prevent model over-sensitivity.
6. Saves the optimal model checkpoint to saved_models/pure_cnn_best.pth.
7. Generates training performance curves saved to results/03_training_history.png.
"""

import os
import sys
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Ensure project root is in sys.path for robust package imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step01_data_preparation.dataset_loader import get_data_loaders
from step02_model_architecture.pure_cnn import PureCNN

# -------------------------------------------------------------
# REPRODUCIBILITY SEED CONFIGURATION
# -------------------------------------------------------------
def set_seed(seed=42):
    """
    Enforces deterministic execution across all random number generators
    to guarantee consistent, repeatable training results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------------------------------------------
# HIGH-PRECISION HYPERPARAMETERS
# -------------------------------------------------------------
BATCH_SIZE = 16
IMAGE_SIZE = (128, 128)
LEARNING_RATE = 0.0003      # Tuned learning rate for smooth convergence
WEIGHT_DECAY = 1e-4         # L2 regularization to suppress overfitting
EPOCHS = 25                 # Optimal training duration
RANDOM_SEED = 42

# Hardware acceleration setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory paths
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_model():
    """
    Executes the enhanced training workflow with dynamic scheduling,
    loss-aware checkpoint saving, and deterministic reproducibility.
    """
    # 1. Lock random seeds
    set_seed(RANDOM_SEED)

    print("=== [STAGE 3] INITIALIZING REPRODUCIBLE ENHANCED TRAINING ===")
    print(f"Random Seed   : {RANDOM_SEED} (Locked for deterministic results)")
    print(f"Device        : {DEVICE}")
    print(f"Target Epochs : {EPOCHS}")
    print(f"Batch Size    : {BATCH_SIZE}")
    print(f"Learning Rate : {LEARNING_RATE}")
    print(f"Weight Decay  : {WEIGHT_DECAY}\n")

    # 2. Load Train and Validation sets
    train_loader, val_loader, _, classes = get_data_loaders(
        batch_size=BATCH_SIZE, image_size=IMAGE_SIZE
    )
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    # 3. Instantiate Enhanced Pure CNN Model
    model = PureCNN(in_channels=1, num_classes=num_classes).to(DEVICE)

    # 4. Loss function, Optimizer with Weight Decay, and LR Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # Tracking metrics
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = SAVED_MODELS_DIR / "pure_cnn_best.pth"

    print("----------------------------------------------------------------------")
    print(f"{'Epoch':<10} | {'Train Loss':<12} | {'Train Acc':<12} | {'Val Loss':<12} | {'Val Acc':<10}")
    print("----------------------------------------------------------------------")

    # -------------------------------------------------------------
    # 5. TRAINING & VALIDATION LOOP
    # -------------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0

        # Step scheduler based on validation loss
        scheduler.step(epoch_val_loss)

        # Record history
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"{epoch:02d}/{EPOCHS:02d}      | "
            f"{epoch_train_loss:<12.4f} | "
            f"{epoch_train_acc:>6.2f}%      | "
            f"{epoch_val_loss:<12.4f} | "
            f"{epoch_val_acc:>6.2f}%"
        )

        # Checkpoint: Save model when validation loss reaches new low OR high accuracy with balanced loss
        is_best_loss = epoch_val_loss < best_val_loss
        is_top_acc = (epoch_val_acc >= best_val_acc) and (epoch_val_loss < 0.5)

        if is_best_loss or is_top_acc:
            if is_best_loss:
                best_val_loss = epoch_val_loss
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
            best_epoch = epoch

            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": epoch_val_loss,
                "val_accuracy": epoch_val_acc,
                "classes": classes
            }, best_model_path)

    print("----------------------------------------------------------------------")
    print(f"\n[INFO] Best Epoch Selected           : {best_epoch:02d}")
    print(f"[INFO] Best Validation Accuracy      : {best_val_acc:.2f}%")
    print(f"[INFO] Lowest Validation Loss        : {best_val_loss:.4f}")
    print(f"[INFO] Optimal model checkpoint saved: {best_model_path}")

    # -------------------------------------------------------------
    # 6. GENERATE AND SAVE TRAINING CURVES
    # -------------------------------------------------------------
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", color="royalblue", linewidth=2)
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", color="crimson", linewidth=2, linestyle="--")
    plt.title("Loss vs. Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Cross-Entropy)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label="Train Accuracy", color="royalblue", linewidth=2)
    plt.plot(epochs_range, history["val_acc"], label="Val Accuracy", color="crimson", linewidth=2, linestyle="--")
    plt.title("Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    history_plot_path = RESULTS_DIR / "03_training_history.png"
    plt.savefig(history_plot_path, dpi=150)
    plt.close()

    print(f"[INFO] Updated training plot saved to: {history_plot_path}")
    print("=== [STAGE 3] SUCCESS: Training pipeline complete ===")


if __name__ == "__main__":
    train_model()
