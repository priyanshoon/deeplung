import os
import torch
from torchvision import transforms
from PIL import Image

def process(path):
    img = Image.open(path).convert('L')
    eq = transforms.functional.equalize(img)
    auto = transforms.functional.autocontrast(img)
    print(f"{path} -> Orig: {img.getextrema()}, Eq: {eq.getextrema()}, Auto: {auto.getextrema()}")

process('randomtest/pneumonia-7.png')
process('data/merged_dataset/train/Bacterial Pneumonia/Bacterial Pneumonia_00000.jpeg')
process('data/merged_dataset/train/Corona Virus Disease/Corona Virus Disease_00000.jpeg')
