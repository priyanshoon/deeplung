import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# Add src to path to import model
sys.path.append(os.path.join(os.getcwd(), 'src'))
from model import get_model

CLASS_NAMES = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

def check_confidence(image_path, model_path):
    device = torch.device("cpu") # Use CPU for simplicity
    
    # Load model
    try:
        model = get_model(num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_class = torch.max(probabilities, 1)
            
        predicted_class = CLASS_NAMES[top_class.item()]
        confidence = top_prob.item()
        
        print(f"Results for {image_path}:")
        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print("All probabilities:")
        for i, prob in enumerate(probabilities[0]):
             print(f"{CLASS_NAMES[i]}: {prob.item():.4f}")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        check_confidence(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python check_model_confidence.py <image_path> <model_path>")
