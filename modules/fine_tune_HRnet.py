"""
Fine_tune_HRnet.py
Fine-tuning pipeline for HRNet models on clustered synthetic data.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Import HRNet model and training utilities
from .HRnet_model import HRNet
from .dataset import CCPDatasetWrapper
from .config import MIN_CELLS, MAX_CELLS, DEVICE
from .train_HRnet import train_epoch, validate_epoch


def finetune_hrnet_pipeline(
    pretrained_path,  # Path to pretrained model
    train_samples=1500,
    val_samples=300,
    epochs=20,
    batch_size=8,
    learning_rate=2e-4,  # Lower LR for fine-tuning
    weight_decay=1e-4,
    patience=5,
    device=None,
    base_channels=48,
    dropout_rate=0.1,
    save_dir='/content/drive/MyDrive/SU2_Project_HRNet',
    save_best=True,
    # Cluster parameters
    use_clusters=True,
    cluster_prob=0.6,
    cluster_size_range=(1, 8),
    cluster_spread=7.0
):
    """
    Fine-tuning pipeline for HRNet with clustered data.
    
    This function loads a pretrained HRNet model and fine-tunes it on synthetic
    data with clustered CCPs, which better represents real biological data.
    
    Args:
        pretrained_path (str): Path to pretrained model weights (.pth file)
        train_samples (int): Number of training samples to generate
        val_samples (int): Number of validation samples to generate
        epochs (int): Number of fine-tuning epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate (typically lower than initial training)
        weight_decay (float): Weight decay for optimizer
        patience (int): Patience for early stopping
        device: Device to train on (cuda/cpu), auto-detected if None
        base_channels (int): Base number of channels in HRNet (32 or 48)
        dropout_rate (float): Dropout rate
        save_dir (str): Directory to save models
        save_best (bool): Whether to save best model
        use_clusters (bool): Whether to use clustered synthetic data
        cluster_prob (float): Probability of creating a cluster vs isolated CCP
        cluster_size_range (tuple): Min and max number of CCPs per cluster
        cluster_spread (float): Spatial spread of CCPs within a cluster
    
    Returns:
        model: Fine-tuned model
        history: Training history dict with 'train_loss', 'val_loss', 'lr'
    """
    
    # =========================================================================
    # 1. Setup device
    # =========================================================================
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # =========================================================================
    # 2. Load pretrained model
    # =========================================================================
    print("\n" + "="*70)
    print("LOADING PRETRAINED MODEL")
    print("="*70)
    
    model = HRNet(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        dropout_rate=dropout_rate
    )
    
    # Load weights
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"✓ Model loaded from: {pretrained_path}")
    else:
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: ~{num_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # =========================================================================
    # 3. Create datasets with clusters
    # =========================================================================
    print("\n" + "="*70)
    print("CREATING CLUSTERED DATASETS")
    print("="*70)
    print(f"Cluster settings:")
    print(f"  - use_clusters: {use_clusters}")
    print(f"  - cluster_prob: {cluster_prob}")
    print(f"  - cluster_size_range: {cluster_size_range}")
    print(f"  - cluster_spread: {cluster_spread}")
    
    train_dataset = CCPDatasetWrapper(
        length=train_samples,
        min_n=MIN_CELLS,
        max_n=MAX_CELLS,
        use_clusters=use_clusters,
        cluster_prob=cluster_prob,
        cluster_size_range=cluster_size_range,
        cluster_spread=cluster_spread
    )
    
    val_dataset = CCPDatasetWrapper(
        length=val_samples,
        min_n=MIN_CELLS,
        max_n=MAX_CELLS,
        use_clusters=use_clusters,
        cluster_prob=cluster_prob,
        cluster_size_range=cluster_size_range,
        cluster_spread=cluster_spread
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
    
    print(f"✓ Train samples: {train_samples}")
    print(f"✓ Val samples: {val_samples}")
    
    # =========================================================================
    # 4. Setup optimizer with lower LR for fine-tuning
    # =========================================================================
    print("\n" + "="*70)
    print("OPTIMIZER SETUP")
    print("="*70)
    
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
    
    print(f"Initial learning rate: {learning_rate:.6f}")
    print(f"Weight decay: {weight_decay:.6f}")
    
    # =========================================================================
    # 5. Training loop
    # =========================================================================
    print("\n" + "="*70)
    print("STARTING FINE-TUNING")
    print("="*70)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device)
        
        # Update scheduler
        scheduler.step(val_loss)
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
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"  ✓ New best model! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'hrnet_finetuned_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Checkpoint saved: epoch {epoch+1}")
        
        # Early stopping
        if patience_counter >= patience * 2:  # 2x patience for early stopping
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # =========================================================================
    # 6. Load best model and save final
    # =========================================================================
    print("\n" + "="*70)
    print("SAVING FINAL MODEL")
    print("="*70)
    
    if save_best and os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
        print("✓ Best model loaded")
    
    print(f"\nFine-tuning completed! Best val loss: {best_val_loss:.4f}")
    
    # Save final model
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, 'hrnet_finetuned_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"✓ Final model saved to: {final_path}")
    
    # Also save best model to save_dir
    if save_best and os.path.exists('best_model.pth'):
        best_path = os.path.join(save_dir, 'hrnet_finetuned_best.pth')
        torch.save(model.state_dict(), best_path)
        print(f"✓ Best model saved to: {best_path}")
    
    return model, history


if __name__ == "__main__":
    """
    Example usage for testing the fine-tuning pipeline.
    """
    print("Testing HRNet fine-tuning pipeline...")
    
    # Example: Fine-tune a pretrained model
    pretrained_model_path = "/path/to/hrnet_best.pth"
    
    model, history = finetune_hrnet_pipeline(
        pretrained_path=pretrained_model_path,
        train_samples=100,  # Small for testing
        val_samples=20,
        epochs=2,
        batch_size=4,
        learning_rate=2e-4,
        cluster_prob=0.6,
        cluster_size_range=(2, 6),
        cluster_spread=7.0
    )
    
    print("\nTest completed successfully!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")