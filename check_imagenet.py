import torch
from torchvision import models, transforms
from PIL import Image
import sys

def check_image(image_path):
    # Load pre-trained ResNet18
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()

    # Preprocess image
    preprocess = weights.transforms()
    
    try:
        img = Image.open(image_path).convert('RGB')
        batch = preprocess(img).unsqueeze(0)

        # Predict
        with torch.no_grad():
            prediction = model(batch).squeeze(0).softmax(0)
        
        # Get top 5 classes
        top5_prob, top5_catid = torch.topk(prediction, 5)
        
        print(f"Predictions for {image_path}:")
        for i in range(top5_prob.size(0)):
            category_name = weights.meta["categories"][top5_catid[i].item()]
            score = top5_prob[i].item()
            print(f"{category_name}: {score:.4f}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_image(sys.argv[1])
    else:
        print("Please provide an image path.")
