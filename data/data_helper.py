import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CrabDataset(Dataset):
    def __init__(self, spawner_path, recruit_path, n_years, 
                 spawner_max=None, recruit_max=None):
        """
        Args:
            spawner_max (float, optional): Force a specific max value for scaling. 
                                           If None, calculates max from current data.
            recruit_max (float, optional): Force a specific max value for scaling.
                                           If None, calculates max from current data.
        """
        # 1. Load Data
        self.spawners = np.load(spawner_path).astype(np.float32)
        self.recruits = np.load(recruit_path).astype(np.float32)
        self.n_years = n_years
        
        # 2. Handle Spawner Max
        if spawner_max is None:
            # Case A: Training (Calculate from self)
            self.spawner_max = np.max(self.spawners) + 1e-6
            print(f"Computed new Spawner Max: {self.spawner_max:.4f}")
        else:
            # Case B: Validation (Use provided value)
            self.spawner_max = spawner_max
            print(f"Using provided Spawner Max: {self.spawner_max:.4f}")

        # 3. Handle Recruit Max
        if recruit_max is None:
            # Case A: Training (Calculate from self)
            self.recruit_max = np.max(self.recruits) + 1e-6
            print(f"Computed new Recruit Max: {self.recruit_max:.4f}")
        else:
            # Case B: Validation (Use provided value)
            self.recruit_max = recruit_max
            print(f"Using provided Recruit Max: {self.recruit_max:.4f}")

    def __len__(self):
        return len(self.spawners)

    def __getitem__(self, idx):
        raw_spawner = self.spawners[idx]
        raw_recruit = self.recruits[idx]
        year_idx = idx % self.n_years
        
        # Scale to [0, 1]
        spawner_scaled = raw_spawner / self.spawner_max
        recruit_scaled = raw_recruit / self.recruit_max
        
        # Return as [1, 50, 50] tensors
        return (torch.tensor(spawner_scaled, dtype=torch.float32).unsqueeze(0),
                torch.tensor(recruit_scaled, dtype=torch.float32).unsqueeze(0),
                torch.tensor(year_idx, dtype=torch.long))
    

def get_dataloaders(batch_size=5, n_years=30):
    """
    Helper function to initialize the split loaders.
    """
    data_dir = "data/dummy/splits/"

    # Initialize datasets with the file paths
    train_ds = CrabDataset(data_dir + "train_spawners.npy", data_dir + "train_recruits.npy", n_years=n_years, spawner_max=None, recruit_max=None)
    val_ds = CrabDataset(data_dir + "val_spawners.npy", data_dir + "val_recruits.npy", n_years=n_years, spawner_max=train_ds.spawner_max, recruit_max=train_ds.recruit_max)
    test_ds = CrabDataset(data_dir + "test_spawners.npy", data_dir + "test_recruits.npy", n_years=n_years, spawner_max=train_ds.spawner_max, recruit_max=train_ds.recruit_max)

    # Create loaders 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader