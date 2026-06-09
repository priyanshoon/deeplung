import torch
from torchvision import models, transforms
from PIL import Image
import os
import glob
from collections import Counter

def collect_stats(image_dir, num_images=20):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    preprocess = weights.transforms()
    
    image_paths = glob.glob(os.path.join(image_dir, "*.jpeg"))[:num_images]
    
    all_classes = []
    
    print(f"Processing {len(image_paths)} images from {image_dir}...")
    
    for image_path in image_paths:
        try:
            img = Image.open(image_path).convert('RGB')
            batch = preprocess(img).unsqueeze(0)
            
            with torch.no_grad():
                prediction = model(batch).squeeze(0).softmax(0)
            
            top5_prob, top5_catid = torch.topk(prediction, 5)
            
            classes = [weights.meta["categories"][id.item()] for id in top5_catid]
            all_classes.extend(classes)
            # print(f"{os.path.basename(image_path)}: {classes}")
            
        except Exception as e:
            print(f"Error {image_path}: {e}")

    counts = Counter(all_classes)
    print("\nMost common ImageNet classes for X-rays:")
    for cls, count in counts.most_common(20):
        print(f"{cls}: {count}")

if __name__ == "__main__":
    collect_stats("data/Lung Disease Dataset/train/Normal")
