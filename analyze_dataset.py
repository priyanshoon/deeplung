import os
from PIL import Image
import numpy as np

classes = ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]
for c in classes:
    path = os.path.join("data/merged_dataset/train", c)
    means = []
    stds = []
    for i, f in enumerate(os.listdir(path)):
        if i > 100: break # sample 100 images
        if not f.startswith('.'):
            img = Image.open(os.path.join(path, f)).convert('L')
            arr = np.array(img)
            means.append(arr.mean())
            stds.append(arr.std())
    print(f"{c}: mean={np.mean(means):.2f}, std={np.mean(stds):.2f}")
