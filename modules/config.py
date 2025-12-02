import torch
import os
import yaml

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random Seed
SEED = 42

# Synthetic Data Config
PATCH_SIZE = 128
SIM_CONFIG = {
    "na": 1.49,
    "wavelength": 512,
    "px_size": 0.07,
    "wiener_parameter": 0.1,
    "apo_cutoff": 2.0,
    "apo_bend": 0.9
}
MIN_CELLS = 5
MAX_CELLS = 49

# Training Config
TRAIN_SAMPLES = 500
VAL_SAMPLES = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.1
EPOCHS = 200
PATIENCE = 5

# Paths (Colab specific defaults)
VAL_DATA_PATH = "/content/val_data"
VAL_TIF_PATH = f"{VAL_DATA_PATH}/val.tif"
VAL_CSV_PATH = f"{VAL_DATA_PATH}/val.csv"

# Load config from YAML if exists
CONFIG_PATH = "config.yaml"
if os.path.exists(CONFIG_PATH):
    print(f"Loading configuration from {CONFIG_PATH}...")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    # Update globals with values from YAML
    if "TRAIN_SAMPLES" in config: TRAIN_SAMPLES = config["TRAIN_SAMPLES"]
    if "VAL_SAMPLES" in config: VAL_SAMPLES = config["VAL_SAMPLES"]
    if "BATCH_SIZE" in config: BATCH_SIZE = config["BATCH_SIZE"]
    if "LEARNING_RATE" in config: LEARNING_RATE = float(config["LEARNING_RATE"])
    if "WEIGHT_DECAY" in config: WEIGHT_DECAY = float(config["WEIGHT_DECAY"])
    if "DROPOUT_RATE" in config: DROPOUT_RATE = float(config["DROPOUT_RATE"])
    if "EPOCHS" in config: EPOCHS = config["EPOCHS"]
    if "PATIENCE" in config: PATIENCE = config["PATIENCE"]
    if "SEED" in config: SEED = config["SEED"]
    
    # Data Generator Config
    if "MIN_CELLS" in config: MIN_CELLS = config["MIN_CELLS"]
    if "MAX_CELLS" in config: MAX_CELLS = config["MAX_CELLS"]
    if "PATCH_SIZE" in config: PATCH_SIZE = config["PATCH_SIZE"]
    if "SIM_CONFIG" in config: 
        # Update SIM_CONFIG keys individually to preserve defaults for missing keys
        for k, v in config["SIM_CONFIG"].items():
            SIM_CONFIG[k] = v

    print("Configuration updated.")
