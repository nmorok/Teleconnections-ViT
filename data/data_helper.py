import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CrabDataset(Dataset):
    """
    Two-channel Dataset for crab spawner-recruit data.
    Input (x): [S_t, R_{t-1}]
    Target (y): [R_t]
    """
    def __init__(self, spawner_path, recruit_path, n_years=30, normalize=True):
        # Load the data from paths
        self.spawners = np.load(spawner_path).astype(np.float32)
        self.recruits = np.load(recruit_path).astype(np.float32)
        self.n_years = n_years # 30
        self.normalize = normalize
        
        self.n_total_grids = self.spawners.shape[0] # hopefully 3000
        

        if self.normalize:
            # Normalize each sample individually to [0, 1]
            self.spawn_mean = self.spawners[self.spawners > 0].mean()
            self.spawn_std = self.spawners[self.spawners > 0].std()
            self.recruit_mean = self.recruits[self.recruits > 0].mean()
            self.recruit_std = self.recruits[self.recruits > 0].std()
            
            print(f"Spawner norm: mean={self.spawn_mean:.4f}, std={self.spawn_std:.4f}")
            print(f"Recruit norm: mean={self.recruit_mean:.4f}, std={self.recruit_std:.4f}")
        
        # Identify indices where a "previous year" exists within the same bootstrap.
        # We skip the first year (index 0, 30, 60...) of every bootstrap block.
        #self.valid_indices = [
        #    i for i in range(len(self.spawners)) 
        #    if i % self.n_years != 0
        #]

    #def __len__(self):
    #    return len(self.valid_indices)
    def __len__(self):
        return len(self.spawners)

    def __getitem__(self, idx):

        # Map the dataset index to a valid year that has a t-1 neighbor
        #real_idx = self.valid_indices[idx]
        real_idx = idx
        
        # Channel 1: Current Spawners (S_t)
        s_t = self.spawners[real_idx]
        # Channel 2: Previous Recruits (R_{t-1})
        #r_prev = self.recruits[real_idx - 1]
        
        # Target: Current Recruits (R_t)
        target = self.recruits[real_idx]

        year_idx = idx % self.n_years
        
        # Normalize
        if self.normalize:
            s_t = (s_t - self.spawn_mean) / (self.spawn_std + 1e-8)
            #r_prev = (r_prev - self.recruit_mean) / (self.recruit_std + 1e-8)
            target = (target - self.recruit_mean) / (self.recruit_std + 1e-8)
            
        # Stack channels to create [2, H, W]
        #x = np.stack([s_t, r_prev], axis=0)
        
        
        # Convert to tensors
        #x = torch.from_numpy(x).float()
        x = torch.from_numpy(s_t).float().unsqueeze(0) # [1, H, W]
        y = torch.from_numpy(target).float().unsqueeze(0) # [1, H, W]
        year_idx = torch.tensor(year_idx).long()
        
        return x, y, year_idx

def get_dataloaders(batch_size=5, n_years=30):
    """
    Helper function to initialize the split loaders.
    """
    data_dir = "data/dummy/splits/"

    # Initialize datasets with the file paths
    train_ds = CrabDataset(data_dir + "train_spawners.npy", data_dir + "train_recruits.npy", n_years=n_years, normalize = True)
    val_ds = CrabDataset(data_dir + "val_spawners.npy", data_dir + "val_recruits.npy", n_years=n_years, normalize = False)
    test_ds = CrabDataset(data_dir + "test_spawners.npy", data_dir + "test_recruits.npy", n_years=n_years, normalize = False)

    # Apply training stats to validation and test sets
    val_ds.spawn_mean = train_ds.spawn_mean
    val_ds.spawn_std = train_ds.spawn_std
    val_ds.recruit_mean = train_ds.recruit_mean
    val_ds.recruit_std = train_ds.recruit_std
    test_ds.spawn_mean = train_ds.spawn_mean
    test_ds.spawn_std = train_ds.spawn_std
    test_ds.recruit_mean = train_ds.recruit_mean
    test_ds.recruit_std = train_ds.recruit_std
    val_ds.normalize = True
    test_ds.normalize = True

    # Create loaders 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader