import os
import numpy as np
from PIL import Image

def get_border_mean(path):
    img = Image.open(path).convert('L')
    arr = np.array(img)
    h, w = arr.shape
    border_pixels = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return border_pixels.mean()

for c in ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]:
    path = os.path.join("data/merged_dataset/train", c)
    means = []
    for i, f in enumerate(os.listdir(path)):
        if i > 500: break
        if not f.startswith('.'):
            means.append(get_border_mean(os.path.join(path, f)))
    print(f"{c}: Border Mean = {np.mean(means):.2f}")

path = "randomtest"
means = []
print("Randomtest:")
for f in os.listdir(path):
    if not f.startswith('.'):
        mean = get_border_mean(os.path.join(path, f))
        print(f"  {f}: Border Mean = {mean:.2f}")
