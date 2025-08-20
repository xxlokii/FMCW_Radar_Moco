import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch
import logging
import torch.nn.functional as F


# MocoDataset class for loading radar data from npy files
class MocoDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.data_index = []

        for file_name in self.file_list:
            file_path = os.path.join(self.data_dir, file_name)
            data = np.load(file_path, mmap_mode='r') 
            num_samples = data.shape[0]
            self.data_index.extend([(file_name, i) for i in range(num_samples)])

        self.total_samples = len(self.data_index)
        self.aug = RadarDataAugmentor()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_name, data_idx = self.data_index[idx]
        file_path = os.path.join(self.data_dir, file_name)
        data = np.load(file_path, mmap_mode='r')

        item = data[data_idx] 

        patch = torch.tensor(item, dtype=torch.float32)
        patch1 = self.aug(patch.clone())
        patch2 = self.aug(patch.clone())
        return patch1, patch2

# Moco_data_loader function to create DataLoader for MocoDataset
def Moco_data_loader(train_dir, batch_size = 256, random_seed = 42):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    logger = logging.getLogger('moco_v3')

    train_dataset = MocoDataset(train_dir)
    logger.info(f"The number of training samples: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory= True)
    return train_loader


# Radar data augmentation class
class RadarDataAugmentor:
    def __init__(self, 
                 phase_noise_prob=0.5,
                 magnitude_scale_prob=0.6,
                 antenna_swap_prob=0.3,
                 channel_drop_ratio=0.1,
                 doppler_mask_prob=0.4,
                 range_mask_prob=0.4,
                 thermal_noise_std=0.02,
                 clutter_noise_std=0.05,
                 time_jitter_prob=0.3,
                 max_time_shift=3,
                 continuous_mask=True,             
    ):
        """
        Radar data augmentation module.
        """
        # Probability parameters
        self.phase_noise_prob = phase_noise_prob
        self.magnitude_scale_prob = magnitude_scale_prob
        self.antenna_swap_prob = antenna_swap_prob
        self.channel_drop_ratio = channel_drop_ratio
        self.doppler_mask_prob = doppler_mask_prob
        self.range_mask_prob = range_mask_prob
        self.time_jitter_prob = time_jitter_prob
        
        # Intensity parameters
        self.thermal_noise_std = thermal_noise_std
        self.clutter_noise_std = clutter_noise_std
        self.max_time_shift = max_time_shift
        self.continuous_mask = continuous_mask

    def _complex_perturbation(self, x_complex):
        """Complex domain perturbation"""
        T = x_complex.size(0)
        
        # Phase noise 
        if torch.rand(1) < self.phase_noise_prob:
            phase_shift = torch.cumsum(torch.randn(T) * 0.1, dim=0)
            x_complex = x_complex * torch.exp(1j * phase_shift[:, None, None, None])
        
        # Dynamic magnitude scaling (considering antenna differences)
        if torch.rand(1) < self.magnitude_scale_prob:
            # Independent scaling for each antenna
            ant_scale = torch.empty(3).uniform_(0.85, 1.15)
            # Temporal variation
            time_scale = torch.linspace(0.95, 1.05, T)
            x_complex = x_complex * (ant_scale[None, :, None, None] * time_scale[:, None, None, None])
        
        return x_complex

    def _antenna_operations(self, x_aug, x_complex):
        """Antenna operations"""
        
        if not x_complex.is_complex():
            x_complex = x_complex.to(torch.complex64)
        
        # Antenna swapping
        if torch.rand(1) < self.antenna_swap_prob:
            perm = torch.randperm(3)
            x_aug = torch.cat([x_aug[:, 2*p:2*(p+1)] for p in perm], dim=1)
            x_complex = x_complex[:, perm]
        
        # Antenna coupling effect
        if torch.rand(1) < 0.2:
            # Create a complex coupling matrix
            coupling_mat = torch.eye(3, dtype=torch.complex64)
            coupling_mat += torch.randn(3, 3, dtype=torch.float32) * 0.1
            
            # Perform complex matrix multiplication
            x_complex = torch.einsum('ij,tjhw->tihw', coupling_mat, x_complex)
        
        # Channel attenuation
        if torch.rand(1) < self.channel_drop_ratio:
            ant_idx = torch.randint(3, (1,))
            noise_power = x_complex[:, ant_idx].abs().mean() * 0.3
            noise = torch.randn_like(x_complex[:, ant_idx]) * noise_power
            x_complex[:, ant_idx] += noise
        
        return x_aug, x_complex


    def _generate_adaptive_mask(self, length, mask_ratio):
        """Generate adaptive masking"""
        mask_len = int(length * mask_ratio)
        start = torch.randint(0, max(1, length - mask_len + 1), (1,))
        
        # Generate a mask with smooth transitions
        mask = torch.ones(length)
        if mask_len > 0:
            # Smooth transitions at edges
            transition = min(5, mask_len // 2)
            mask[start:start+transition] = torch.linspace(1, 0, transition)
            mask[start+transition:start+mask_len-transition] = 0
            mask[start+mask_len-transition:start+mask_len] = torch.linspace(0, 1, transition)
        
        return mask

    def _spatial_masking(self, x_aug, H, W):
        """Improved spatial masking"""
        # Doppler masking
        if torch.rand(1) < self.doppler_mask_prob:
            if self.continuous_mask:
                mask = self._generate_adaptive_mask(H, 0.2)
                x_aug = x_aug * mask[None, None, :, None]
            else:
                x_aug *= (torch.rand(H) > 0.8)[None, None, :, None]
        
        # Range masking
        if torch.rand(1) < self.range_mask_prob:
            if self.continuous_mask:
                mask = self._generate_adaptive_mask(W, 0.15)
                x_aug = x_aug * mask[None, None, None, :]
            else:
                x_aug *= (torch.rand(W) > 0.85)[None, None, None, :]
        
        return x_aug

    def _add_advanced_noise(self, x_aug, x_complex):
        """Advanced noise model"""
        T, C, H, W = x_aug.shape
        
        # Adaptive thermal noise
        thermal_noise = torch.randn_like(x_aug) * self.thermal_noise_std
        thermal_noise *= torch.empty(C).uniform_(0.8, 1.2)[None, :, None, None]
        x_aug += thermal_noise

        # Clutter model
        if torch.rand(1) < 0.6:
            base_size = max(4, H//8, W//8)  
            
            # Generate base clutter
            clutter_base = torch.randn(1, 1, base_size, base_size)
            
            clutter_noise = F.interpolate(
                clutter_base,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            
            # Spatial correlation handling
            clutter_noise = F.avg_pool2d(
                clutter_noise,
                kernel_size=3,
                padding=1,
                stride=1
            )
            
            # Ensure final noise size matches
            if clutter_noise.shape[2:] != (H, W):
                clutter_noise = F.interpolate(clutter_noise, size=(H, W), mode='bilinear')
            
            # Add noise (ensure proper broadcasting)
            scale = torch.empty(1).uniform_(0.8, 1.2)
            x_aug += clutter_noise.expand_as(x_aug) * self.clutter_noise_std * scale
        
        return x_aug


    def _time_operations(self, x_aug):
        """Temporal operations"""
        if torch.rand(1) < self.time_jitter_prob:
            shift = torch.randint(-self.max_time_shift, self.max_time_shift+1, (1,))
            if shift > 0:
                x_aug = torch.cat([x_aug[:1].expand(shift.item(), -1, -1, -1), x_aug[:-shift.item()]], dim=0)
            elif shift < 0:
                x_aug = torch.cat([x_aug[-shift.item():], x_aug[-1:].expand(-shift.item(), -1, -1, -1)], dim=0)
        
        # Added micro-Doppler disturbance
        if torch.rand(1) < 0.4:
            micro_doppler = torch.cumsum(torch.randn(x_aug.size(0)) * 0.05, dim=0)
            x_aug = x_aug * torch.exp(1j * micro_doppler)[:, None, None, None] 
        return x_aug
    

    def __call__(self, x):
        x_aug = x.clone()
        T, C, H, W = x_aug.shape
        
        # Convert to complex for processing
        x_complex = x_aug[:, 0::2] + 1j * x_aug[:, 1::2]
        # 1. Complex domain enhancement
        x_complex = self._complex_perturbation(x_complex)
        # 2. Antenna operations
        x_aug, x_complex = self._antenna_operations(x_aug, x_complex)
        # 3. Spatial masking
        x_aug = self._spatial_masking(x_aug, H, W)
        # 4. Noise injection
        x_aug = self._add_advanced_noise(x_aug, x_complex)
        # 5. Temporal operations
        x_aug = self._time_operations(x_aug)

        # Ensure output shape matches original shape
        x_aug = torch.cat([x_complex.real, x_complex.imag], dim=1).view(T, C, H, W)
        
        return x_aug
    

def zero_score_normalize(data):
    mean = np.mean(data, axis = (0, 2, 3), keepdims= True)
    std = np.std(data, axis = (0, 2, 3), keepdims= True)
    normalizede_data = np.where(std!=0, (data - mean) / std, 0)
    return normalizede_data