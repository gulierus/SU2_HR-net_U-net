import numpy as np
import torch
import torch.utils.data as torch_data
from .simulation import SyntheticDataset
from .config import MIN_CELLS, MAX_CELLS


class SyntheticCCPDataset(SyntheticDataset):
    """Synthetic CCP dataset with ground truth masks and optional clustering."""
    
    def __init__(self, min_n=MIN_CELLS, max_n=MAX_CELLS, 
                 radius_choices=[0.5, 1.0, 1.5, 2.0, 2.5],
                 contrast_fg_range=(0.0, 1.0), contrast_bg_range=(0.0, 1.0),
                 use_clusters=False, cluster_prob=0.6, 
                 cluster_size_range=(2, 6), cluster_spread=8.0):
        super().__init__(contrast_fg_range, contrast_bg_range)
        
        self.min_n, self.max_n = min_n, max_n
        self.radius_choices = radius_choices
        self.thickness = 1.0
        self.max_offset = 8
        self.beta_a, self.beta_b = 2, 1
        
        # Grid pro izolovane CCP (kdyz use_clusters=False)
        yy, xx = np.mgrid[15:self.patch_size - 1:16, 15:self.patch_size - 1:16]
        self.yy = yy.flatten()
        self.xx = xx.flatten()
        self.yyy, self.xxx = np.mgrid[:self.patch_size, :self.patch_size]
        
        # Cluster parametry
        self.use_clusters = use_clusters
        self.cluster_prob = cluster_prob
        self.cluster_size_range = cluster_size_range
        self.cluster_spread = cluster_spread
    
    def __iter__(self):
        while True:
            yield self.data_sample()
    
    def _generate_positions_grid(self, n):
        """Puvodni metoda - pozice na mrizce s offsetem."""
        indices = np.random.choice(len(self.yy), size=n, replace=False)
        offsets = np.random.uniform(-self.max_offset, self.max_offset, (n, 2))
        positions = np.column_stack([self.yy[indices], self.xx[indices]]) + offsets
        return positions
    
    def _generate_positions_clusters(self, n):
        """Nova metoda - pozice s clustery."""
        positions = []
        remaining = n
        
        while remaining > 0:
            if np.random.random() < self.cluster_prob and remaining >= 2:
                # Vytvor cluster
                cluster_size = min(np.random.randint(*self.cluster_size_range), remaining)
                cx = np.random.uniform(15, self.patch_size - 15)
                cy = np.random.uniform(15, self.patch_size - 15)
                
                for _ in range(cluster_size):
                    dx = np.random.normal(0, self.cluster_spread / 2)
                    dy = np.random.normal(0, self.cluster_spread / 2)
                    px = np.clip(cx + dx, 5, self.patch_size - 5)
                    py = np.clip(cy + dy, 5, self.patch_size - 5)
                    positions.append([py, px])
                remaining -= cluster_size
            else:
                # Izolovana CCP
                px = np.random.uniform(10, self.patch_size - 10)
                py = np.random.uniform(10, self.patch_size - 10)
                positions.append([py, px])
                remaining -= 1
        
        return np.array(positions)
    
    def data_sample(self):
        n = np.random.randint(self.min_n, self.max_n)
        
        # Vyber metodu generovani pozic
        if self.use_clusters:
            positions = self._generate_positions_clusters(n)
            n = len(positions)  # Muze se mirne lisit
        else:
            positions = self._generate_positions_grid(n)
        
        classes = np.random.beta(self.beta_a, self.beta_b, n) * 0.9 + 0.1
        radii = np.random.choice(self.radius_choices, size=n)
        
        target_distance = classes * radii
        distance = np.hypot(self.yyy[..., None] - positions[:, 0], 
                           self.xxx[..., None] - positions[:, 1])
        abs_distance = np.abs(distance - target_distance)
        parts = np.where(abs_distance > self.thickness, 0, 
                        np.log(np.interp(abs_distance / self.thickness, [0, 1], [np.e, 1])))
        full_image = np.sum(parts, -1)
        
        distances = np.maximum(classes - distance / ((1 - classes) * 2 + radii + self.thickness * 2), 0)
        y = np.minimum(np.sum(distances, -1), 1)
        
        x = super()._simulate_sim(full_image)
        return x, y


class CCPDatasetWrapper(torch_data.Dataset):
    """PyTorch Dataset that yields synthetic CCP images and masks on-the-fly."""
    
    def __init__(self, length=500, min_n=MIN_CELLS, max_n=MAX_CELLS,
                 use_clusters=False, cluster_prob=0.6, 
                 cluster_size_range=(2, 6), cluster_spread=8.0):
        super().__init__()
        self.length = length
        self._synthetic = SyntheticCCPDataset(
            min_n=min_n, 
            max_n=max_n,
            use_clusters=use_clusters,
            cluster_prob=cluster_prob,
            cluster_size_range=cluster_size_range,
            cluster_spread=cluster_spread
        )
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        img, mask = self._synthetic.data_sample()
        
        # Data Augmentation
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        
        if np.random.rand() > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)
        
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k)
            mask = np.rot90(mask, k)
        
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.05, img.shape)
            img = img + noise
        
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-0.1, 0.1)
            img = alpha * img + beta
        
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        
        img = img.copy()
        mask = mask.copy()
        
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return img_tensor, mask_tensor