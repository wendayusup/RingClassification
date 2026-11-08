"""
DATASET RENAMING & OFFLINE AUGMENTATION SCRIPT
----------------------------------------------
Automates the renaming of raw smartphone capture files into a standardized format
and applies systematic data augmentation transformations:
- Rotations: 90, 180, 270 degrees
- Flips: Horizontal and Vertical reflections
- Combined: Rotation 90 + Horizontal Flip
- Lighting variations: Brightness +/- 15%
Multiplies the dataset from 24 raw images per class to 216 images per class.
"""

import os
from pathlib import Path
from PIL import Image, ImageEnhance

BASE_DIR = Path(__file__).resolve().parent.parent

categories = [
    {"folder": "defect_ring", "prefix": "defect"},
    {"folder": "normal_ring", "prefix": "normal"}
]

print("=== STARTING DATASET RENAMING & AUGMENTATION ===")

for cat in categories:
    folder_path = os.path.join(BASE_DIR, cat["folder"])
    prefix = cat["prefix"]
    
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        continue
    
    # Retrieve all raw image files (prior to augmentation)
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith("temp_")])
    
    print(f"\nProcessing category: {cat['folder']} ({len(files)} raw files)...")
    
    # Step 1: Temporarily rename files to avoid name collisions during reordering
    temp_files = []
    for idx, filename in enumerate(files, start=1):
        old_path = os.path.join(folder_path, filename)
        temp_name = f"temp_{prefix}_{idx:02d}.jpg"
        temp_path = os.path.join(folder_path, temp_name)
        os.rename(old_path, temp_path)
        temp_files.append(temp_name)
        
    # Step 2: Rename to finalized standardized format (e.g., defect_01.jpg)
    orig_renamed_files = []
    for idx, temp_name in enumerate(temp_files, start=1):
        temp_path = os.path.join(folder_path, temp_name)
        final_orig_name = f"{prefix}_{idx:02d}.jpg"
        final_orig_path = os.path.join(folder_path, final_orig_name)
        os.rename(temp_path, final_orig_path)
        orig_renamed_files.append((final_orig_name, final_orig_path))
    
    print(f"-> Successfully renamed 24 raw files to {prefix}_01.jpg through {prefix}_24.jpg")
    
    # Step 3: Apply systematic data augmentation
    created_count = 0
    for name, path in orig_renamed_files:
        base_id = name.split(".")[0]  # e.g., 'defect_01'
        with Image.open(path) as img:
            img = img.convert("RGB")
            
            # 1. 90-degree rotation
            rot90 = img.rotate(90)
            rot90.save(os.path.join(folder_path, f"{base_id}_rot90.jpg"), quality=95)
            
            # 2. 180-degree rotation
            rot180 = img.rotate(180)
            rot180.save(os.path.join(folder_path, f"{base_id}_rot180.jpg"), quality=95)
            
            # 3. 270-degree rotation
            rot270 = img.rotate(270)
            rot270.save(os.path.join(folder_path, f"{base_id}_rot270.jpg"), quality=95)
            
            # 4. Horizontal Flip (Left-Right reflection)
            flip_h = img.transpose(Image.FLIP_LEFT_RIGHT)
            flip_h.save(os.path.join(folder_path, f"{base_id}_flipH.jpg"), quality=95)
            
            # 5. Vertical Flip (Top-Bottom reflection)
            flip_v = img.transpose(Image.FLIP_TOP_BOTTOM)
            flip_v.save(os.path.join(folder_path, f"{base_id}_flipV.jpg"), quality=95)
            
            # 6. Combined 90-degree rotation and Horizontal Flip
            rot90_flip = rot90.transpose(Image.FLIP_LEFT_RIGHT)
            rot90_flip.save(os.path.join(folder_path, f"{base_id}_rot90_flipH.jpg"), quality=95)
            
            # 7. Brightness enhancement (+15%)
            enhancer = ImageEnhance.Brightness(img)
            bright_up = enhancer.enhance(1.15)
            bright_up.save(os.path.join(folder_path, f"{base_id}_bright_up.jpg"), quality=95)
            
            # 8. Brightness reduction (-15%)
            bright_down = enhancer.enhance(0.85)
            bright_down.save(os.path.join(folder_path, f"{base_id}_bright_down.jpg"), quality=95)
            
            created_count += 8

    total_files = len(os.listdir(folder_path))
    print(f"-> Augmentation complete: +{created_count} transformed images generated.")
    print(f"-> Current file count in {cat['folder']}: {total_files} files.")

print("\n=== RENAMING & AUGMENTATION COMPLETED SUCCESSFULLY ===")
