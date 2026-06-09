# Dataset Specification: Lung Disease Dataset

## 1. Executive Summary
This report outlines the specifications for the "Lung Disease Dataset" used in the DeepLung project.
- **Total Samples**: 10,095 X-ray images.
- **Class Balance**: Perfectly balanced across 5 classes (~2,000 images per class).
- **Data Quality**: High. No corrupt files or null metadata found.
- **Resolution**: Highly variable (Min: 132x91, Max: 5623x4757). Resizing/pre-processing is required.

## 2. Dataset Structure
- **Root Location**: `data/Lung Disease Dataset`
- **Format**: Hierarchical directory structure (Split -> Class -> Images)
- **Splits**:
    - `train`: Training set (60% approx)
    - `test`: Testing set (20% approx)
    - `val`: Validation set (20% approx)

## 3. Class Distribution
The dataset contains 5 distinct classes representing different lung conditions.

| Class | Total Count | Percentage |
| :--- | :--- | :--- |
| Bacterial Pneumonia | 2,009 | 19.90% |
| Corona Virus Disease | 2,031 | 20.12% |
| Normal | 2,013 | 19.94% |
| Tuberculosis | 2,034 | 20.15% |
| Viral Pneumonia | 2,008 | 19.89% |

**Imbalance Check**:
- **Imbalance Ratio**: 1.01 (Max count / Min count)
- **Status**: **Balanced**. No class weighting or resampling required.

## 4. Split Distribution
Detailed breakdown of samples across the three partitions:

| Class | Train | Test | Val | Total |
| :--- | :--- | :--- | :--- | :--- |
| Bacterial Pneumonia | 1205 | 403 | 401 | 2009 |
| Corona Virus Disease | 1218 | 407 | 406 | 2031 |
| Normal | 1207 | 404 | 402 | 2013 |
| Tuberculosis | 1220 | 408 | 406 | 2034 |
| Viral Pneumonia | 1204 | 403 | 401 | 2008 |
| **Sum** | **6054** | **2025** | **2016** | **10095** |

## 5. Data Requirements & Constraints
### Image Dimensions
- **Minimum**: 132 x 91 pixels
- **Maximum**: 5623 x 4757 pixels
- **Average**: ~1030 x 815 pixels
- **Recommendation**: Due to high variance, all input images must be resized to a fixed input size (e.g., 224x224 or 256x256) during preprocessing.

### Null Values & Corruption
- **Corrupt Files**: 0 loaded failures.
- **Missing Data**: None. All class folders are populated.

## 6. Preprocessing Recommendations
1. **Resizing**: Resize all images to `(256, 256)` to handle resolution variance.
2. **Normalization**: Normalize pixel values to `[0, 1]` or standardize using ImageNet means if using transfer learning.
3. **Augmentation**: Since the dataset is balanced, heavy class-specific augmentation is not strictly necessary for balance, but general augmentation (rotation, flip) is recommended to improve generalization.
