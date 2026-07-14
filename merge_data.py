import os
import shutil
import random
import hashlib
from collections import defaultdict

TARGET_CLASSES = [
    "Bacterial Pneumonia",
    "Corona Virus Disease",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia"
]

SPLITS = ["train", "val", "test"]

# images_dict[split][target_class] = [list_of_absolute_paths]
images_dict = {
    split: {cls: [] for cls in TARGET_CLASSES}
    for split in SPLITS
}

seen_hashes = set()

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def add_images(source_dir, split, target_class):
    if not os.path.exists(source_dir):
        return
    for f in os.listdir(source_dir):
        if not f.startswith('.'):
            file_path = os.path.join(source_dir, f)
            file_hash = get_file_hash(file_path)
            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                images_dict[split][target_class].append(file_path)

# 1. Main dataset
ds1 = "data/Lung Disease Dataset"
for split in SPLITS:
    add_images(os.path.join(ds1, split, "Bacterial Pneumonia"), split, "Bacterial Pneumonia")
    add_images(os.path.join(ds1, split, "Corona Virus Disease"), split, "Corona Virus Disease")
    add_images(os.path.join(ds1, split, "Normal"), split, "Normal")
    add_images(os.path.join(ds1, split, "Tuberculosis"), split, "Tuberculosis")
    add_images(os.path.join(ds1, split, "Viral Pneumonia"), split, "Viral Pneumonia")

# 2. Dataset 2
ds2 = "data/dataset2"
for split in SPLITS:
    add_images(os.path.join(ds2, split, "COVID19"), split, "Corona Virus Disease")
    add_images(os.path.join(ds2, split, "NORMAL"), split, "Normal")
    add_images(os.path.join(ds2, split, "TURBERCULOSIS"), split, "Tuberculosis")
    
    # Custom handling for dataset2 pneumonia
    pneu_dir = os.path.join(ds2, split, "PNEUMONIA")
    if os.path.exists(pneu_dir):
        for f in os.listdir(pneu_dir):
            if not f.startswith('.'):
                file_path = os.path.join(pneu_dir, f)
                file_hash = get_file_hash(file_path)
                if file_hash not in seen_hashes:
                    seen_hashes.add(file_hash)
                    if "bacteria" in f.lower():
                        images_dict[split]["Bacterial Pneumonia"].append(file_path)
                    elif "virus" in f.lower():
                        images_dict[split]["Viral Pneumonia"].append(file_path)

# 3. Dataset 3
# Removed to reduce source bias (does not contain Pneumonia)

# 4. Dataset 4
# Removed to reduce source bias (does not contain Pneumonia)

# 6. Dataset 6
ds6 = "data/dataset6"
for split in SPLITS:
    add_images(os.path.join(ds6, split, "Covid-19"), split, "Corona Virus Disease")
    add_images(os.path.join(ds6, split, "Normal"), split, "Normal")
    add_images(os.path.join(ds6, split, "Pneumonia-Bacterial"), split, "Bacterial Pneumonia")
    add_images(os.path.join(ds6, split, "Tuberculosis"), split, "Tuberculosis")

# Now we sample and copy to create merged_dataset
out_dir = "data/merged_dataset"
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

os.makedirs(out_dir)

for split in SPLITS:
    print(f"--- Processing Split: {split} ---")
    split_dir = os.path.join(out_dir, split)
    
    # Calculate min class size to balance
    counts = {cls: len(images_dict[split][cls]) for cls in TARGET_CLASSES}
    print("Initial counts:")
    for cls, cnt in counts.items():
        print(f"  {cls}: {cnt}")
        
    min_count = min(counts.values())
    if min_count == 0:
        print(f"Warning: Minimum class count is 0 in split {split}. Adjusting to not crash.")
        # If val split has 0 for some reason, maybe we skip balancing or use max available
        pass
        
    print(f"Target balanced count: {min_count}")
    
    if min_count > 0:
        for cls in TARGET_CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            os.makedirs(cls_dir)
            
            # Use all images instead of downsampling to min_count. 
            # We handle imbalance via WeightedRandomSampler during training.
            all_imgs = images_dict[split][cls]
            
            for i, img_path in enumerate(all_imgs):
                file_ext = os.path.splitext(img_path)[1]
                new_name = f"{cls}_{i:05d}{file_ext}"
                dest_path = os.path.join(cls_dir, new_name)
                shutil.copy2(img_path, dest_path)
                
print("\nDone creating data/merged_dataset!")
