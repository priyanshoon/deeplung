import os
import numpy as np
from PIL import Image

def get_center_mean(path):
    img = Image.open(path).convert('L')
    arr = np.array(img)
    h, w = arr.shape
    h_s, h_e = int(h*0.25), int(h*0.75)
    w_s, w_e = int(w*0.25), int(w*0.75)
    center_pixels = arr[h_s:h_e, w_s:w_e]
    return center_pixels.mean()

for c in ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]:
    path = os.path.join("data/merged_dataset/train", c)
    means = []
    for i, f in enumerate(os.listdir(path)):
        if i > 500: break
        if not f.startswith('.'):
            means.append(get_center_mean(os.path.join(path, f)))
    print(f"{c}: Center Mean = {np.mean(means):.2f}")

path = "randomtest"
means = []
print("Randomtest:")
for f in os.listdir(path):
    if not f.startswith('.'):
        mean = get_center_mean(os.path.join(path, f))
        print(f"  {f}: Center Mean = {mean:.2f}")
