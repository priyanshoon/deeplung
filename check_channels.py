import os
from PIL import Image

classes = ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]
for c in classes:
    path = os.path.join("data/Lung Disease Dataset/train", c)
    modes = {}
    for i, f in enumerate(os.listdir(path)):
        if i > 500: break
        if not f.startswith('.'):
            img = Image.open(os.path.join(path, f))
            m = img.mode
            modes[m] = modes.get(m, 0) + 1
    print(f"{c}: {modes}")
