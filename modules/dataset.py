import numpy as np
import torch
import torch.utils.data as torch_data
from .simulation import SyntheticDataset
from .config import MIN_CELLS, MAX_CELLS

class SyntheticCCPDataset(SyntheticDataset):
    """Synthetic CCP dataset with ground truth masks."""
    def __init__(self, min_n=MIN_CELLS, max_n=MAX_CELLS, radius=2.5, contrast_fg_range=(0.0, 1.0), contrast_bg_range=(0.0, 1.0)):
        super().__init__(contrast_fg_range, contrast_bg_range)
        self.min_n, self.max_n = min_n, max_n
        self.radius = radius
        self.thickness = 1.0
        self.max_offset = 8
        self.beta_a, self.beta_b = 2, 1
        yy, xx = np.mgrid[15:self.patch_size - 1:16, 15:self.patch_size - 1:16]
        self.yy = yy.flatten()
        self.xx = xx.flatten()
        self.yyy, self.xxx = np.mgrid[:self.patch_size, :self.patch_size]

    def __iter__(self):
        while True:
            yield self.data_sample()

    def data_sample(self):
        n = np.random.randint(self.min_n, self.max_n)
        indices = np.random.choice(len(self.yy), size=n, replace=False)
        offsets = np.random.uniform(-self.max_offset, self.max_offset, (n, 2))
        positions = np.column_stack([self.yy[indices], self.xx[indices]]) + offsets
        classes = np.random.beta(self.beta_a, self.beta_b, n) * 0.9 + 0.1
        target_distance = classes * self.radius
        distance = np.hypot(self.yyy[..., None] - positions[:, 0], self.xxx[..., None] - positions[:, 1])
        abs_distance = np.abs(distance - target_distance)
        parts = np.where(abs_distance > self.thickness, 0, np.log(np.interp(abs_distance / self.thickness, [0, 1], [np.e, 1])))
        full_image = np.sum(parts, -1)
        distances = np.maximum(classes - distance / ((1 - classes) * 2 + self.radius + self.thickness * 2), 0)
        y = np.minimum(np.sum(distances, -1), 1)
        x = super()._simulate_sim(full_image)
        return x, y

class CCPDatasetWrapper(torch_data.Dataset):
    """
    PyTorch Dataset that yields synthetic CCP images and masks on-the-fly.
    """
    def __init__(self, length=500, min_n=MIN_CELLS, max_n=MAX_CELLS):
        super().__init__()
        self.length = length
        self._synthetic = SyntheticCCPDataset(min_n=min_n, max_n=max_n)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img, mask = self._synthetic.data_sample()
        
        # Data Augmentation
        # 1. Random Horizontal Flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
            
        # 2. Random Vertical Flip
        if np.random.rand() > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)
            
        # 3. Random 90-degree Rotation
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k)
            mask = np.rot90(mask, k)
        
        # 4. Gaussian noise 
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.05, img.shape)
            img = img + noise

        # 5. Random brightness/contrast
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # contrast
            beta = np.random.uniform(-0.1, 0.1)  # brightness
            img = alpha * img + beta
        
            
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        
        # Ensure correct copy for torch
        img = img.copy()
        mask = mask.copy()
        
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        return img_tensor, mask_tensor
