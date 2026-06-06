import torch
import os
from src.model import get_model

def save_dummy_model():
    os.makedirs('checkpoints_dryrun', exist_ok=True)
    # Use non-pretrained weights to avoid downloading in dummy mode
    model = get_model(num_classes=5, pretrained=False)
    torch.save(model.state_dict(), 'checkpoints_dryrun/best_model.pth')
    print("Saved dummy model.")

if __name__ == "__main__":
    save_dummy_model()
