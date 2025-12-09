import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .loss import combined_loss
from ..data.dataset import CCPDatasetWrapper
from ..models.UnetPlusPlus_model import UNetPlusPlus
from ..config import (
    DEVICE, BATCH_SIZE, TRAIN_SAMPLES, VAL_SAMPLES, 
    MIN_CELLS, MAX_CELLS, PATCH_SIZE, SIM_CONFIG
)
from ..utils import patch_dataloader


def train_epoch(model, dataloader, optimizer, device, bce_weight=0.5):
    """
    Train model for one epoch.
    
    Args:
        model: U-Net++ model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        bce_weight: Weight for BCE component in combined loss
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks, bce_weight=bce_weight)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, device, bce_weight=0.5):
    """
    Validate model for one epoch.
    
    Args:
        model: U-Net++ model
        dataloader: Validation data loader
        device: Device to validate on
        bce_weight: Weight for BCE component in combined loss
    
    Returns:
        tuple: (average validation loss, average dice score)
    """
    model.eval()
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = combined_loss(outputs, masks, bce_weight=bce_weight)
            
            total_loss += loss.item()
            
            # Calculate Dice score
            pred = torch.sigmoid(outputs) > 0.5
            gt = masks > 0.5
            intersection = (pred * gt).sum()
            union = pred.sum() + gt.sum()
            dice = (2. * intersection / (union + 1e-6)).item()
            total_dice += dice
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    
    return total_loss / len(dataloader), total_dice / len(dataloader)


def train_unetplusplus_pipeline(
    train_samples=TRAIN_SAMPLES,
    val_samples=VAL_SAMPLES,
    epochs=20,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=5,
    device=DEVICE,
    bce_weight=0.5,
    # Clustering parameters
    use_clusters=True,
    cluster_sample_prob=0.5,
    cluster_prob=0.6,
    cluster_size_range=(2, 6),
    cluster_spread=8.0,
    # Model parameters
    features=[32, 64, 128, 256, 512],
    use_attention=True,
    dropout_rate=0.1,
    # Saving
    save_dir='.',
    save_best=True
):
    """
    Complete training pipeline for U-Net++ model.
    
    Args:
        train_samples: Number of training samples
        val_samples: Number of validation samples
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        patience: Early stopping patience (epochs)
        device: Device to train on ('cuda' or 'cpu')
        bce_weight: Weight for BCE component in combined loss (0.0-1.0)
        
        # Clustering parameters
        use_clusters: Whether to use clustered CCPs
        cluster_sample_prob: Probability of generating clustered sample (0.0-1.0)
        cluster_prob: Probability of forming a cluster within sample
        cluster_size_range: Range of CCPs per cluster
        cluster_spread: Spatial spread of cluster in pixels
        
        # Model parameters
        features: List of feature channels for encoder/decoder
        use_attention: Whether to use attention gates
        dropout_rate: Dropout rate for regularization
        
        # Saving
        save_dir: Directory to save models
        save_best: Whether to save best model
    
    Returns:
        tuple: (trained model, training history dict)
    """
    # Patch DataLoader to fix RecursionError
    patch_dataloader()
    
    # Create datasets
    print("Creating train dataset...")
    train_dataset = CCPDatasetWrapper(
        length=train_samples,
        min_n=MIN_CELLS,
        max_n=MAX_CELLS,
        use_clusters=use_clusters,
        cluster_sample_prob=cluster_sample_prob,
        cluster_prob=cluster_prob,
        cluster_size_range=cluster_size_range,
        cluster_spread=cluster_spread
    )
    
    print("Creating validation dataset...")
    val_dataset = CCPDatasetWrapper(
        length=val_samples,
        min_n=MIN_CELLS,
        max_n=MAX_CELLS,
        use_clusters=use_clusters,
        cluster_sample_prob=cluster_sample_prob,
        cluster_prob=cluster_prob,
        cluster_size_range=cluster_size_range,
        cluster_spread=cluster_spread
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Create model
    print("\nCreating U-Net++ model...")
    model = UNetPlusPlus(
        in_channels=1,
        out_channels=1,
        features=features,
        use_attention=use_attention,
        dropout_rate=dropout_rate
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: ~{num_params * 4 / 1024 / 1024:.2f} MB\n")
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    print("="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, bce_weight)
        
        # Validate
        val_loss, val_dice = validate_epoch(model, val_loader, device, bce_weight)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | LR: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint_path = f"{save_dir}/unetpp_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        
        # Check for best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            
            if save_best:
                best_path = f"{save_dir}/unetpp_best.pth"
                torch.save(model.state_dict(), best_path)
                print(f"New best model saved! (Val Loss: {val_loss:.4f})")
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{patience}")
            
            if early_stopping_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print()
    
    # Save final model
    final_path = f"{save_dir}/unetpp_final.pth"
    torch.save(model.state_dict(), final_path)
    
    print("="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Final val dice: {history['val_dice'][-1]:.4f}")
    print(f"\nModel saved to: {save_dir}/")
    print(f"  - unetpp_best.pth (best model)")
    print(f"  - unetpp_final.pth (final model)")
    print("="*70 + "\n")
    
    return model, history