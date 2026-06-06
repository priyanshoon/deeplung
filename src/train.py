import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
from data_loader import get_data_loaders
from model import get_model

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # For Mac M1/M2 chips, use mps if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get data loaders
    train_loader, val_loader, _, class_names = get_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    if train_loader is None:
        return
        
    print(f"Classes: {class_names}")
    
    # Initialize model
    model = get_model(num_classes=len(class_names))
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Deep copy the model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
            
    print(f"Best val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Pneumonia Detection Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset (containing train/val/test)')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')
    
    args = parser.parse_args()
    train(args)
