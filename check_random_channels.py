import os
from PIL import Image

path = "randomtest"
for f in os.listdir(path):
    if not f.startswith('.'):
        img = Image.open(os.path.join(path, f))
        print(f"{f}: {img.mode}")
