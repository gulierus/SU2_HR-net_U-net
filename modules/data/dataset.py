import numpy as np
import torch
import torch.utils.data as torch_data

from .simulation import SyntheticDataset
from ..config import MIN_CELLS, MAX_CELLS


class SyntheticCCPDataset(SyntheticDataset):
    """
    Synthetic CCP dataset with ground truth masks and optional clustering.
    
    This dataset generates synthetic microscopy images of Clathrin-Coated Pits (CCPs)
    with realistic biological distributions. It supports two generation modes:
    
    1. Grid-based: CCPs positioned on a regular grid with random offsets
    2. Cluster-based: CCPs arranged in realistic biological clusters
    
    Key Improvements:
    - Probabilistic mixing of clustered and isolated CCPs via cluster_sample_prob
    - Improved mask generation using np.maximum to prevent blob formation
    - Configurable cluster parameters (size, spread, probability)
    
    Args:
        min_n (int): Minimum number of CCPs per image
        max_n (int): Maximum number of CCPs per image
        radius_choices (list): Possible radii for CCPs
        contrast_fg_range (tuple): Foreground contrast range (0.0-1.0)
        contrast_bg_range (tuple): Background contrast range (0.0-1.0)
        use_clusters (bool): Enable cluster generation mode
        cluster_sample_prob (float): Probability of generating a clustered sample (0.0-1.0)
                                     0.0 = always grid, 1.0 = always clusters
        cluster_prob (float): Probability of forming a cluster within a sample
        cluster_size_range (tuple): Min and max CCPs per cluster
        cluster_spread (float): Spatial spread of CCPs within clusters (pixels)
    """
    
    def __init__(self, min_n=MIN_CELLS, max_n=MAX_CELLS, 
                 radius_choices=[1.0, 1.5, 2.0, 2.5],
                 contrast_fg_range=(0.0, 1.0), 
                 contrast_bg_range=(0.0, 1.0),
                 use_clusters=False, 
                 cluster_sample_prob=0.5,
                 cluster_prob=0.6, 
                 cluster_size_range=(2, 6), 
                 cluster_spread=8.0):
        super().__init__(contrast_fg_range, contrast_bg_range)
        
        self.min_n, self.max_n = min_n, max_n
        self.radius_choices = radius_choices
        self.thickness = 1.0
        self.max_offset = 8
        self.beta_a, self.beta_b = 2, 1
        
        # Grid for isolated CCPs (used when use_clusters=False or probabilistically)
        # Creates a 16-pixel spaced grid across the patch
        yy, xx = np.mgrid[15:self.patch_size - 1:16, 15:self.patch_size - 1:16]
        self.yy = yy.flatten()
        self.xx = xx.flatten()
        self.yyy, self.xxx = np.mgrid[:self.patch_size, :self.patch_size]
        
        # Clustering parameters
        self.use_clusters = use_clusters
        self.cluster_sample_prob = cluster_sample_prob  # NEW: probabilistic mixing
        self.cluster_prob = cluster_prob
        self.cluster_size_range = cluster_size_range
        self.cluster_spread = cluster_spread
    
    def __iter__(self):
        """Infinite iterator for streaming data generation."""
        while True:
            yield self.data_sample()
    
    def _generate_positions_grid(self, n):
        """
        Generate CCP positions on a regular grid with random offsets.
        
        This method creates a spatially uniform distribution of CCPs, suitable for
        training the model to detect isolated particles.
        
        Args:
            n (int): Number of CCPs to generate
            
        Returns:
            np.ndarray: Array of shape (n, 2) with [y, x] coordinates
        """
        indices = np.random.choice(len(self.yy), size=n, replace=False)
        offsets = np.random.uniform(-self.max_offset, self.max_offset, (n, 2))
        positions = np.column_stack([self.yy[indices], self.xx[indices]]) + offsets
        return positions
    
    def _generate_positions_clusters(self, n):
        """
        Generate CCP positions with realistic biological clustering.
        
        This method creates a mixture of clustered and isolated CCPs, mimicking
        the spatial distribution observed in real TIRF microscopy data. CCPs often
        appear in clusters due to biological processes like clathrin-mediated
        endocytosis hotspots.
        
        Algorithm:
        1. For each CCP to place, decide: cluster or isolated?
        2. If cluster: place multiple CCPs around a random center with Gaussian spread
        3. If isolated: place single CCP at random position
        
        Args:
            n (int): Number of CCPs to generate
            
        Returns:
            np.ndarray: Array of shape (n, 2) with [y, x] coordinates
        """
        positions = []
        remaining = n
        
        while remaining > 0:
            if np.random.random() < self.cluster_prob and remaining >= 2:
                # Create a cluster
                cluster_size = min(np.random.randint(*self.cluster_size_range), remaining)
                cx = np.random.uniform(15, self.patch_size - 15)
                cy = np.random.uniform(15, self.patch_size - 15)
                
                # Place CCPs around cluster center with Gaussian spread
                for _ in range(cluster_size):
                    dx = np.random.normal(0, self.cluster_spread / 2)
                    dy = np.random.normal(0, self.cluster_spread / 2)
                    px = np.clip(cx + dx, 5, self.patch_size - 5)
                    py = np.clip(cy + dy, 5, self.patch_size - 5)
                    positions.append([py, px])
                remaining -= cluster_size
            else:
                # Create isolated CCP
                px = np.random.uniform(10, self.patch_size - 10)
                py = np.random.uniform(10, self.patch_size - 10)
                positions.append([py, px])
                remaining -= 1
        
        return np.array(positions)
    
    def data_sample(self):
        """
        Generate one training sample (image, mask pair).
        
        This method orchestrates the complete sample generation pipeline:
        1. Decide on generation mode (grid vs cluster) based on cluster_sample_prob
        2. Generate CCP positions
        3. Create high-resolution rings for SIM simulation
        4. Generate ground truth mask using improved method
        5. Simulate realistic SIM microscopy acquisition
        
        Returns:
            tuple: (x, y) where
                x (np.ndarray): Simulated SIM image (128, 128)
                y (np.ndarray): Ground truth mask (128, 128) with values in [0, 1]
        """
        n = np.random.randint(self.min_n, self.max_n)
        
        # Choose position generation method
        # NEW: Probabilistic mixing allows training on both distributions
        if self.use_clusters and np.random.random() < self.cluster_sample_prob:
            positions = self._generate_positions_clusters(n)
            n = len(positions)  # Actual count may differ due to clustering
        else:
            positions = self._generate_positions_grid(n)
        
        # Generate CCP properties (size/brightness variation)
        classes = np.random.beta(self.beta_a, self.beta_b, n) * 0.9 + 0.1
        radii = np.random.choice(self.radius_choices, size=n)
        
        # Create high-resolution ring structures for SIM input
        target_distance = classes * radii
        distance = np.hypot(self.yyy[..., None] - positions[:, 0], 
                           self.xxx[..., None] - positions[:, 1])
        abs_distance = np.abs(distance - target_distance)
        parts = np.where(abs_distance > self.thickness, 0, 
                        np.log(np.interp(abs_distance / self.thickness, [0, 1], [np.e, 1])))
        full_image = np.sum(parts, -1)
        
        # ====================================================================
        # IMPROVED MASK GENERATION
        # ====================================================================
        # OLD METHOD (commented out):
        # Used np.sum which caused overlapping Gaussians to create bright blobs
        # This confused the model into detecting blob centers instead of individual CCPs
        #
        # distances = np.maximum(classes - distance / ((1 - classes) * 2 + radii + self.thickness * 2), 0)
        # y = np.minimum(np.sum(distances, -1), 1)
        
        # NEW METHOD:
        # Use np.maximum to keep only the brightest point at each pixel
        # This preserves individual CCP peaks even when they overlap
        # Result: Model learns to detect individual CCPs in dense clusters
        y = np.zeros((self.patch_size, self.patch_size))
        for i in range(n):
            dist_i = np.hypot(self.yyy - positions[i, 0], self.xxx - positions[i, 1])
            sigma = 1.5  # Smaller sigma = sharper peaks, easier to separate overlapping CCPs
            point_mask = np.exp(-dist_i**2 / (2 * sigma**2))
            y = np.maximum(y, point_mask)  # KEY: maximum instead of sum
        
        # Simulate realistic SIM microscopy acquisition
        x = super()._simulate_sim(full_image)
        
        return x, y


class CCPDatasetWrapper(torch_data.Dataset):
    """
    PyTorch Dataset wrapper for on-the-fly synthetic CCP generation.
    
    This wrapper provides a standard PyTorch Dataset interface for the synthetic
    data generator. It includes data augmentation and proper tensor conversion.
    
    Key Features:
    - On-the-fly generation (no pre-computed dataset required)
    - Geometric augmentation (flips, rotations)
    - Intensity augmentation (noise, contrast)
    - Automatic normalization
    
    Args:
        length (int): Number of samples per epoch
        min_n (int): Minimum CCPs per image
        max_n (int): Maximum CCPs per image
        use_clusters (bool): Enable clustering
        cluster_sample_prob (float): Probability of clustered sample (0.0-1.0)
        cluster_prob (float): Probability of forming cluster within sample
        cluster_size_range (tuple): Range of CCPs per cluster
        cluster_spread (float): Spatial spread of clusters (pixels)
    
    Example:
        >>> # 50% grid samples, 50% clustered samples
        >>> dataset = CCPDatasetWrapper(
        ...     length=1000,
        ...     use_clusters=True,
        ...     cluster_sample_prob=0.5
        ... )
        >>> img, mask = dataset[0]
        >>> print(img.shape, mask.shape)  # torch.Size([1, 128, 128]) for both
    """
    
    def __init__(self, length=500, 
                 min_n=MIN_CELLS, 
                 max_n=MAX_CELLS,
                 use_clusters=False, 
                 cluster_sample_prob=0.5,
                 cluster_prob=0.6, 
                 cluster_size_range=(2, 6), 
                 cluster_spread=8.0):
        super().__init__()
        self.length = length
        self._synthetic = SyntheticCCPDataset(
            min_n=min_n, 
            max_n=max_n,
            use_clusters=use_clusters,
            cluster_sample_prob=cluster_sample_prob,
            cluster_prob=cluster_prob,
            cluster_size_range=cluster_size_range,
            cluster_spread=cluster_spread
        )
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        Generate and augment one training sample.
        
        Data Augmentation Pipeline:
        1. Geometric: Random flips (horizontal/vertical) and 90° rotations
        2. Intensity: Gaussian noise addition
        3. Contrast: Random brightness/contrast adjustment
        4. Normalization: Min-max scaling to [0, 1]
        
        Args:
            idx (int): Sample index (ignored, generation is random)
            
        Returns:
            tuple: (img_tensor, mask_tensor)
                img_tensor: Input image [1, 128, 128]
                mask_tensor: Ground truth mask [1, 128, 128]
        """
        img, mask = self._synthetic.data_sample()
        
        # Geometric augmentation
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        
        if np.random.rand() > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)
        
        # Random 90° rotation
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k)
            mask = np.rot90(mask, k)
        
        # Intensity augmentation
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.05, img.shape)
            img = img + noise
        
        # Contrast augmentation
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # Contrast
            beta = np.random.uniform(-0.1, 0.1)   # Brightness
            img = alpha * img + beta
        
        # Normalization
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        
        # Ensure contiguous arrays
        img = img.copy()
        mask = mask.copy()
        
        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return img_tensor, mask_tensor