# CCP Detection & Tracking Pipeline

**Authors:** Matyáš Veselý, Ruslan Guliev

Pipeline for detection and tracking of Clathrin-Coated Pits (CCPs) in TIRF-SIM microscopy data. The project combines deep learning models for detection with advanced tracking algorithms.

## Overview

This repository provides a complete solution for:
- **Synthetic training data generation** using SIM reconstruction
- **CCP detection** using U-Net++ and HRNet architectures
- **Particle tracking** using btrack (Bayesian) and LapTrack (LAP) algorithms
- **Evaluation** using HOTA metrics

### Models

| Model | Description |
|-------|-------------|
| **U-Net++** | Encoder-decoder architecture with nested skip connections and attention gates |
| **HRNet** | High-Resolution Network with multi-scale fusion for preserving spatial resolution |

### Tracking Methods

| Method | Description |
|--------|-------------|
| **btrack** | Bayesian tracking with hypothesis model, suitable for biological data [1] |
| **LapTrack** | Linear Assignment Problem, efficient for dense scenes with gap closing [2] |

---

## Repository Structure

```
ccp_pipeline/
├── config.py                 # Global configuration (device, hyperparameters)
├── utils.py                  # Utility functions (I/O, TIFF loading)
│
├── data/
│   ├── dataset.py            # SyntheticCCPDataset, CCPDatasetWrapper
│   └── simulation.py         # SIM reconstruction, OTF, Perlin noise
│
├── models/
│   ├── HRnet_model.py        # HRNet architecture
│   └── UnetPlusPlus_model.py # U-Net++ architecture
│
├── training/
│   ├── loss.py               # Combined BCE + Dice loss
│   ├── train_HRnet.py        # Training pipeline for HRNet
│   ├── train_UnetPlusPlus.py # Training pipeline for U-Net++
│   └── fine_tune_HRnet.py    # Fine-tuning on clustered data
│
└── evaluation/
    ├── tracking.py           # CCPDetector, btrack, LapTrack, HOTA metrics
    └── sweep.py              # Parameter sweep for tracking optimization
```

---

## Improved Synthetic Data Generator

**Problem:** the original generator can generate only solitude cells and can't simulate clusters which are in real biological data. 
**Solution:** New improvement of Synthetic Data Generator with additional cluster simulations.
Key component in `data/dataset.py`. The generator creates realistic microscopy data with accurate ground truth masks.

### Main Improvements

#### 1. Probabilistic Mixing of Clusters and Grid

The `cluster_sample_prob` parameter controls the sample ratio:
- `0.0` = grid-based only (uniformly distributed CCPs)
- `1.0` = clusters only
- `0.5` = 50% mix (recommended for training)

```python
dataset = CCPDatasetWrapper(
    use_clusters=True,
    cluster_sample_prob=0.5,  # 50% clustered samples
    cluster_prob=0.6,         # Probability of creating a cluster
    cluster_size_range=(2, 6),# Number of CCPs per cluster
    cluster_spread=8.0        # Spatial spread (px)
)
```

#### 2. Fixed Mask Generation

**Problem:** The original method used `np.sum` for overlapping Gaussians, creating bright "blobs" at overlap regions. The model then learned to detect blob centers instead of individual CCPs.

**Solution:** Using `np.maximum` preserves individual peaks even when overlapping:

```python
# OLD (incorrect):
y = np.minimum(np.sum(distances, -1), 1)  # Sums overlaps → blobs

# NEW (correct):
for i in range(n):
    point_mask = np.exp(-dist_i**2 / (2 * sigma**2))
    y = np.maximum(y, point_mask)  # Preserves individual peaks
```

#### 3. Clustering Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_clusters` | `False` | Enable cluster mode |
| `cluster_sample_prob` | `0.5` | Probability of clustered sample |
| `cluster_prob` | `0.6` | Probability of forming cluster vs. isolated CCP |
| `cluster_size_range` | `(2, 6)` | Range of CCPs per cluster |
| `cluster_spread` | `8.0` | σ of Gaussian position distribution in cluster (px) |

### Recommended Configuration

**For training from scratch:**
```python
cluster_sample_prob=0.5  # Balanced mix
```

**For fine-tuning on real data:**
```python
cluster_sample_prob=0.7  # More clusters (matches real data)
cluster_spread=7.0       # Tighter clusters
```

---

## Tracking Configuration

### btrack Parameters

```python
BTrackParams(
    max_search_radius=20.0,    # Max distance for linking
    dist_thresh=10.0,          # Distance threshold for hypothesis model
    time_thresh=3,             # Max frames for gap closing
    min_track_len=5,           # Min trajectory length
    segmentation_miss_rate=0.1 # Expected detection miss rate
)
```

### LapTrack Parameters

```python
LapTrackParams(
    track_cost_cutoff=15.0,       # Max cost for linking
    gap_closing_cost_cutoff=15.0, # Max cost for gap closing
    gap_closing_max_frame_count=3 # Max gap length
)
```

---

## References

[1] Ulicna, K., Vallardi, G., Charras, G. & Sheridan, A. R. (2021). **Automated deep lineage tree analysis using a Bayesian single cell tracking approach.** *Frontiers in Computer Science.* https://doi.org/10.3389/fcomp.2021.734559

[2] Fukai, Y. T., Kawaguchi, K. (2022). **LapTrack: Linear assignment particle tracking with tunable metrics.** *Bioinformatics*, 38(11), 3018–3020. https://doi.org/10.1093/bioinformatics/btac299