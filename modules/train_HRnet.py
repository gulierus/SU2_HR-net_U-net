"""
train_HRnet.py
Training pipeline for HRNet models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
import os

# Import HRNet models
from .HRnet_model import HRNet
from .dataset import CCPDatasetWrapper
from .config import *
from .config import MIN_CELLS, MAX_CELLS



def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for binary segmentation."""
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice


def combined_loss(pred, target, bce_weight=0.5):
    """Combined BCE + Dice loss."""
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + (1 - bce_weight) * dice


def train_epoch(model, dataloader, optimizer, device, loss_fn=combined_loss):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, device, loss_fn=combined_loss):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_hrnet_pipeline(
    train_samples=500,
    val_samples=100,
    epochs=200,
    batch_size=8,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=5,
    device=None,
    base_channels=32,
    dropout_rate=0.1,
    save_dir='/content/drive/MyDrive/SU2_Project_HRNet',
    save_best=True
):
    """
    Complete training pipeline for HRNet.
    
    Args:
        train_samples: Number of training samples
        val_samples: Number of validation samples
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        patience: Early stopping patience
        device: Device to train on (cuda/cpu)
        base_channels: Base number of channels (32 or 48)
        dropout_rate: Dropout rate
        save_best: Whether to save best model
    
    Returns:
        model: Trained model
        history: Training history dict
    """
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = HRNet(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        dropout_rate=dropout_rate
    )
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create datasets
    print("\nCreating datasets...")
    # train_dataset = CCPDatasetWrapper(length=train_samples)
    # val_dataset = CCPDatasetWrapper(length=val_samples)
    
    train_dataset = CCPDatasetWrapper(
                length=train_samples,
                min_n=MIN_CELLS,
                max_n=MAX_CELLS,
                use_clusters=True,
                cluster_sample_prob=0.5  # ← NOVÝ: 50% s clustery, 50% bez
    )
    
    val_dataset = CCPDatasetWrapper(
                length=val_samples,
                min_n=MIN_CELLS,
                max_n=MAX_CELLS,
                use_clusters=True,
                cluster_sample_prob=0.5  # ← NOVÝ
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=7, 
        min_lr=1e-6
    )
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_best:
                best_path = os.path.join(save_dir, 'hrnet_best.pth')
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"New best model saved! (Val Loss: {val_loss:.4f})")
                best_path = os.path.join(save_dir, 'hrnet_best.pth')
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f" Checkpoint saved: epoch {epoch+1}")
        
        # Early stopping
        if patience_counter >= patience * 2:  # 2x patience for early stopping
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model
    best_path = os.path.join(save_dir, 'hrnet_best.pth')
    if save_best and os.path.exists('best_model.pth'):
        print("\nLoading best model...")
        best_path = os.path.join(save_dir, 'hrnet_best.pth')
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    final_path = os.path.join(save_dir, 'hrnet_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path}")
    
    return model, history


if __name__ == "__main__":
    # Test the training pipeline
    print("Testing HRNet training pipeline...")
    
    model, history = train_hrnet_pipeline(
        train_samples=10,
        val_samples=5,
        epochs=2,
        batch_size=2,
    )
    
    print("\nTest completed successfully!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")