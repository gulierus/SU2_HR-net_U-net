import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import zipfile
import os
import sys

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_training_history(history):
    """Plots training and validation loss and dice score."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Dice
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_dice'], 'g-', label='Val Dice')
    plt.title('Validation Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def download_and_unzip(url, extract_to, chain_path=None):
    """Downloads a zip file and extracts it."""
    import requests
    
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        
    zip_path = os.path.join(extract_to, "temp.zip")
    
    print(f"Downloading {url}...")
    # Use the certificate chain if provided
    verify = chain_path if chain_path else True
    
    try:
        r = requests.get(url, stream=True, verify=verify)
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            
        os.remove(zip_path)
        print("Done!")
    except requests.exceptions.SSLError as e:
        print(f"SSL Error: {e}")
        print("Try providing the correct certificate chain path.")
    except Exception as e:
        print(f"Error: {e}")

def patch_dataloader():
    """
    Patches torch.utils.data.DataLoader to fix RecursionError by forcing persistent_workers=True.
    This is a workaround for a specific issue in some environments.
    """
    print("Attempting to fix RecursionError by retrieving clean DataLoader...")

    # 1. Identify the DataLoader class currently in memory
    TargetDataLoader = torch.utils.data.DataLoader

    # 2. Obtain a CLEAN, unpatched __init__ method by forcing a fresh import
    # We temporarily remove the submodule from sys.modules to bypass the cache
    if 'torch.utils.data.dataloader' in sys.modules:
        del sys.modules['torch.utils.data.dataloader']

    try:
        # Import a fresh copy of the module
        from torch.utils.data.dataloader import DataLoader as CleanDataLoader
        clean_init = CleanDataLoader.__init__
        print("Success: Retrieved a clean, unpatched __init__ method.")
    except ImportError:
        # Fallback if the internal structure is different
        print("Warning: Could not force reload. Trying to unwrap if possible.")
        clean_init = TargetDataLoader.__init__
        while hasattr(clean_init, '__wrapped__'):
            clean_init = clean_init.__wrapped__

    # 3. Define the patch using the CLEAN init
    def patched_dataloader_init(self, *args, **kwargs):
        # Apply the fix: force persistent_workers=True if num_workers > 0
        if kwargs.get('num_workers', 0) > 0 and 'persistent_workers' not in kwargs:
            kwargs['persistent_workers'] = True
            print(f"[INFO] Patched DataLoader for {self}: persistent_workers=True")
        
        # Call the clean original init
        clean_init(self, *args, **kwargs)

    # 4. Apply the patch to the existing DataLoader class object
    TargetDataLoader.__init__ = patched_dataloader_init

    print("Fixed: torch.utils.data.DataLoader has been successfully patched.")


def open_tiff_file(filepath):
    """
    Load a TIFF file (single or multi-frame) as numpy array.
    
    Args:
        filepath: Path to TIFF file
        
    Returns:
        numpy array with shape (frames, height, width) for multi-frame
        or (height, width) for single frame
    """
    try:
        # Try tifffile first (best for scientific TIFF)
        import tifffile
        return tifffile.imread(filepath)
    except ImportError:
        # Fallback to PIL
        from PIL import Image
        img = Image.open(filepath)
        
        # Handle multi-frame TIFF
        frames = []
        try:
            i = 0
            while True:
                img.seek(i)
                frames.append(np.array(img))
                i += 1
        except EOFError:
            pass
        
        if len(frames) == 1:
            return frames[0]
        else:
            return np.array(frames)

# ============================================================================
# MODEL SAVING UTILITIES
# ============================================================================

def save_model_to_drive(model, save_dir, model_name="hrnet", save_best=True):
    """
    Save trained model to Google Drive or local directory.
    
    Args:
        model: PyTorch model to save
        save_dir: Directory to save model
        model_name: Base name for saved files (default: "hrnet")
        save_best: Whether to also copy best_model.pth if it exists
    
    Returns:
        final_path: Path where final model was saved
    """
    import torch
    import shutil
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save final model
    final_path = os.path.join(save_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"✓ Final model saved to: {final_path}")
    
    # Save best model if exists
    if save_best and os.path.exists('best_model.pth'):
        best_path = os.path.join(save_dir, f"{model_name}_best.pth")
        shutil.copy('best_model.pth', best_path)
        print(f"✓ Best model saved to: {best_path}")
    
    return final_path