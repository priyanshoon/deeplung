import os
from PIL import Image
import numpy as np

path = "randomtest"
means = []
stds = []
for f in os.listdir(path):
    if not f.startswith('.'):
        img = Image.open(os.path.join(path, f)).convert('L')
        arr = np.array(img)
        print(f"{f}: mean={arr.mean():.2f}, std={arr.std():.2f}")
        means.append(arr.mean())
        stds.append(arr.std())
print(f"Overall: mean={np.mean(means):.2f}, std={np.mean(stds):.2f}")
