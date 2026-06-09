import os
import collections
from PIL import Image
import numpy as np

DATA_DIR = "data/Lung Disease Dataset"
SPLITS = ["train", "test", "val"]
CLASSES = ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]

def analyze_dataset():
    stats = {
        "total_images": 0,
        "split_counts": collections.defaultdict(dict),
        "class_counts": collections.defaultdict(int),
        "shapes": [],
        "corrupt_files": [],
        "null_values": 0 # Interpreting as unreadable files
    }

    print("Analyzing dataset...")
    
    for split in SPLITS:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir):
            print(f"Warning: Split {split} not found!")
            continue
            
        for class_name in CLASSES:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class {class_name} not found in {split}!")
                stats["split_counts"][split][class_name] = 0
                continue
                
            files = [f for f in os.listdir(class_dir) if not f.startswith('.')]
            count = len(files)
            stats["split_counts"][split][class_name] = count
            stats["class_counts"][class_name] += count
            stats["total_images"] += count
            
            # Sample a few images for shape stats (checking all might be slow if huge, 
            # but let's try checking all for robustness unless it's massive)
            # Actually, let's check basic properties for ALL to detect corrupt ones.
            for file_name in files:
                file_path = os.path.join(class_dir, file_name)
                try:
                    with Image.open(file_path) as img:
                        # Verify image content
                        img.verify() 
                        # Re-open to get info after verify
                        with Image.open(file_path) as img2:
                             stats["shapes"].append(img2.size)
                except Exception as e:
                    stats["corrupt_files"].append(file_path)
                    stats["null_values"] += 1

    print("\n--- Analysis Complete ---")
    
    # Calculate Shape Stats
    if stats["shapes"]:
        widths = [s[0] for s in stats["shapes"]]
        heights = [s[1] for s in stats["shapes"]]
        avg_w = np.mean(widths)
        avg_h = np.mean(heights)
        min_w, max_w = np.min(widths), np.max(widths)
        min_h, max_h = np.min(heights), np.max(heights)
    else:
        avg_w = avg_h = min_w = max_w = min_h = max_h = 0

    # Output Markdown Report
    print("\n# Dataset Specification & Analysis Report")
    print("\n## 1. Dataset Structure")
    print(f"- **Root Directory**: `{DATA_DIR}`")
    print(f"- **Splits**: {', '.join(SPLITS)}")
    print(f"- **Classes**: {', '.join(CLASSES)}")
    
    print("\n## 2. Dataset Statistics")
    print(f"- **Total Images**: {stats['total_images']}")
    print(f"- **Corrupt/Null Files**: {stats['null_values']}")
    
    print("\n### Class Distribution (Total)")
    print("| Class | Count | Percentage |")
    print("|---|---|---|")
    total = stats["total_images"]
    if total > 0:
        for cls, count in stats["class_counts"].items():
            pct = (count / total) * 100
            print(f"| {cls} | {count} | {pct:.2f}% |")
    
    print("\n### Distribution per Split")
    print("| Split | " + " | ".join(CLASSES) + " | Total |")
    print("|---|---|---|---|---|---|---|")
    for split in SPLITS:
        row = f"| {split} |"
        split_total = 0
        for cls in CLASSES:
            c = stats["split_counts"][split].get(cls, 0)
            row += f" {c} |"
            split_total += c
        row += f" {split_total} |"
        print(row)

    print("\n## 3. Data Quality & Requirements")
    print(f"- **Image Dimensions (WxH)**:")
    print(f"  - Min: {min_w}x{min_h}")
    print(f"  - Max: {max_w}x{max_h}")
    print(f"  - Average: {avg_w:.1f}x{avg_h:.1f}")
    
    if stats["corrupt_files"]:
        print("\n### Issues Found")
        print("The following files could not be opened:")
        for f in stats["corrupt_files"][:10]:
            print(f"- {f}")
        if len(stats["corrupt_files"]) > 10:
            print(f"... and {len(stats['corrupt_files']) - 10} more.")
    else:
        print("\n- **Integrity Check**: Pass (No corrupt files found)")

    # Imbalance check
    if total > 0:
        counts = list(stats["class_counts"].values())
        imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        print(f"\n- **Class Imbalance Ratio**: {imbalance_ratio:.2f}")
        if imbalance_ratio > 1.5:
             print("  - **Status**: Imbalanced. Consider resampling or weighting.")
        else:
             print("  - **Status**: Balanced.")

if __name__ == "__main__":
    analyze_dataset()
