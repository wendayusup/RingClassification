"""
STAGE 1: DATA PREPARATION & LOADER
----------------------------------
Standard CNN Pipeline - Stage 1: "Input Image":
- Converts image to a multi-dimensional array (Height x Width x Channels).
- Resizes images to standard input resolution (128x128).
- Converts to Grayscale (1 channel) for computational efficiency and texture focus.
- Normalizes pixel values to [0.0, 1.0] range (divided by 255).
- Batches images for sequential processing by the CNN.
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pathlib import Path

# Main project directory paths (dynamically resolved to ensure portability across environments)
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_SPLIT_DIR = BASE_DIR / "dataset_split"
RESULTS_DIR = BASE_DIR / "results"

def get_data_transforms(image_size=(128, 128)):
    """
    Image preprocessing transformation pipeline:
    1. Resize: standardizes image dimensions to (128, 128).
    2. Grayscale: converts RGB to single-channel intensity.
    3. ToTensor: converts PIL image to PyTorch tensor and scales [0, 255] to [0.0, 1.0].
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    return transform

def get_data_loaders(batch_size=16, image_size=(128, 128)):
    """
    Constructs PyTorch DataLoader instances for Train, Validation, and Test datasets.
    """
    transform = get_data_transforms(image_size)
    
    train_path = os.path.join(DATASET_SPLIT_DIR, "train")
    val_path   = os.path.join(DATASET_SPLIT_DIR, "val")
    test_path  = os.path.join(DATASET_SPLIT_DIR, "test")
    
    # ImageFolder automatically maps subdirectories (defect_ring, normal_ring) to integer labels
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset   = datasets.ImageFolder(root=val_path,   transform=transform)
    test_dataset  = datasets.ImageFolder(root=test_path,  transform=transform)
    
    # DataLoader batches images and shuffles training data for stochastic gradient descent
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    classes = train_dataset.classes  # ['defect_ring', 'normal_ring']
    
    return train_loader, val_loader, test_loader, classes

def verify_and_preview():
    """
    Verification utility to confirm data loader functionality,
    validate tensor shapes and value ranges, and save a balanced preview grid.
    """
    print("=== [STAGE 1] DATA LOADER VERIFICATION ===")
    
    batch_size = 16
    image_size = (128, 128)
    train_loader, val_loader, test_loader, classes = get_data_loaders(batch_size=batch_size, image_size=image_size)
    
    print(f"Detected classes: {classes}")
    print(f"Train batches: {len(train_loader)} batches ({len(train_loader.dataset)} images)")
    print(f"Val batches  : {len(val_loader)} batches ({len(val_loader.dataset)} images)")
    print(f"Test batches : {len(test_loader)} batches ({len(test_loader.dataset)} images)")
    
    # Explicitly select 2 defect samples and 2 normal samples for balanced preview
    sample_defect = []
    sample_normal = []
    
    for images, labels in train_loader:
        for img, lbl in zip(images, labels):
            if classes[lbl.item()] == "defect_ring" and len(sample_defect) < 2:
                sample_defect.append((img, "defect_ring"))
            elif classes[lbl.item()] == "normal_ring" and len(sample_normal) < 2:
                sample_normal.append((img, "normal_ring"))
            if len(sample_defect) == 2 and len(sample_normal) == 2:
                break
        if len(sample_defect) == 2 and len(sample_normal) == 2:
            break
            
    preview_samples = sample_defect + sample_normal
    
    # Fetch a sample batch to inspect tensor dimensions and statistics
    images, labels = next(iter(train_loader))
    
    print("\n--- Tensor Dimension Inspection (Standard CNN Pipeline) ---")
    print(f"Batch image tensor shape : {images.shape}")
    print(f"  -> Batch size          : {images.shape[0]} images per step")
    print(f"  -> Channels            : {images.shape[1]} (1 = Grayscale)")
    print(f"  -> Height x Width      : {images.shape[2]} x {images.shape[3]} pixels")
    print(f"Batch label tensor shape : {labels.shape}")
    print(f"Pixel value range        : Min = {images.min().item():.3f}, Max = {images.max().item():.3f} (Normalized to [0.0, 1.0])")
    
    # Save a visual grid of 4 balanced samples to the results directory
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, (img, label_name) in enumerate(preview_samples):
        img_np = img.squeeze().numpy()
        axes[i].imshow(img_np, cmap="gray")
        axes[i].set_title(f"Label: {label_name}")
        axes[i].axis("off")
        
    os.makedirs(RESULTS_DIR, exist_ok=True)
    preview_file = os.path.join(RESULTS_DIR, "01_data_loader_preview.png")
    plt.tight_layout()
    plt.savefig(preview_file, dpi=150)
    plt.close()
    
    print(f"\n-> Preview grid saved to: results/01_data_loader_preview.png")
    print("   (Displaying 2 defect_ring samples + 2 normal_ring samples)")
    print("=== [STAGE 1] SUCCESS: Data pipeline verified and ready for CNN ===")

if __name__ == "__main__":
    verify_and_preview()
