"""
DATASET SPLITTING SCRIPT (TRAIN / VALIDATION / TEST)
----------------------------------------------------
Splits the augmented dataset based on unique sample IDs (01 to 24)
to strictly prevent Data Leakage between subsets:
- Sample IDs 01 - 17: Train Set (71%)      -> 17 samples x 9 = 153 images per class
- Sample IDs 18 - 20: Validation Set (12.5%) -> 3 samples x 9 = 27 images per class
- Sample IDs 21 - 24: Test Set (16.5%)       -> 4 samples x 9 = 36 images per class
"""

import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = os.path.join(str(BASE_DIR), "dataset_split")

# Define target split IDs per subset
train_ids = [f"{i:02d}" for i in range(1, 18)]
val_ids   = [f"{i:02d}" for i in range(18, 21)]
test_ids  = [f"{i:02d}" for i in range(21, 25)]

categories = [
    {"folder": "defect_ring", "prefix": "defect"},
    {"folder": "normal_ring", "prefix": "normal"}
]

print("=== STARTING DATASET DISTRIBUTION (TRAIN / VAL / TEST) ===")

for cat in categories:
    src_folder = os.path.join(BASE_DIR, cat["folder"])
    prefix = cat["prefix"]
    
    files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\nProcessing {cat['folder']} (Total: {len(files)} files)...")
    
    counts = {"train": 0, "val": 0, "test": 0}
    
    for f in files:
        # File naming pattern: defect_01.jpg or defect_01_rot90.jpg
        # Extract the 2-digit sample ID (e.g., '01')
        parts = f.split("_")
        if len(parts) >= 2:
            sample_id = parts[1][:2]
            src_file = os.path.join(src_folder, f)
            
            if sample_id in train_ids:
                dst_file = os.path.join(SPLIT_DIR, "train", cat["folder"], f)
                shutil.copy2(src_file, dst_file)
                counts["train"] += 1
            elif sample_id in val_ids:
                dst_file = os.path.join(SPLIT_DIR, "val", cat["folder"], f)
                shutil.copy2(src_file, dst_file)
                counts["val"] += 1
            elif sample_id in test_ids:
                dst_file = os.path.join(SPLIT_DIR, "test", cat["folder"], f)
                shutil.copy2(src_file, dst_file)
                counts["test"] += 1

    print(f"-> Successfully copied to dataset_split:")
    print(f"   Train : {counts['train']} images")
    print(f"   Val   : {counts['val']} images")
    print(f"   Test  : {counts['test']} images")
    print(f"   Total : {sum(counts.values())} images")

print("\n=== DATASET DISTRIBUTION COMPLETED SUCCESSFULLY ===")
