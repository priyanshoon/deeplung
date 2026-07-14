import torch
from torchvision import transforms
from PIL import Image
import os
import sys

sys.path.append('src')
from model import get_model

CLASS_NAMES = ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]

def predict(img_path):
    device = torch.device('cpu')
    model = get_model(num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    top_prob, top_idx = torch.max(probs, 0)
    return CLASS_NAMES[top_idx.item()], top_prob.item()

for f in os.listdir('randomtest'):
    if not f.startswith('.'):
        path = os.path.join('randomtest', f)
        cls, prob = predict(path)
        print(f"{f}: {cls} ({prob:.4f})")
