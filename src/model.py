import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes, pretrained=True):
    """
    Returns a ResNet18 model adjusted for the number of classes.
    """
    # Use weights parameter instead of pretrained=True (deprecated)
    # Fix for SSL certificate error on Mac
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
