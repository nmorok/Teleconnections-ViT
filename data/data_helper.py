import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CrabDataset(Dataset):
    def __init__(self, spawner_path, recruit_path, n_years, memory_years = 5,
                 historical_spawners = None,
                 historical_recruits = None,
                 year_offset=0, mask_path=None,
                 year_mask=None, historical_year_mask=None,
                 include_current_spawner=True):
        """
        """
        # 1. Load Data
        self.year_offset = year_offset
        self.spawners = np.load(spawner_path).astype(np.float32)
        self.recruits = np.load(recruit_path).astype(np.float32)
        self.n_years = n_years
        self.memory_years = memory_years
        # Store historical data from previous split
        self.historical_spawners = historical_spawners  # (n_bootstraps, 5, 50, 50) or None
        self.historical_recruits = historical_recruits  # (n_bootstraps, 5, 50, 50) or None
        self.include_current_spawner = include_current_spawner

        total_samples = len(self.spawners)
        assert total_samples % n_years == 0, \
            f"Total samples ({total_samples}) must be divisible by n_years ({n_years})"
        self.n_bootstraps = total_samples // n_years

        # Verify historical data shape if provided
        if self.historical_spawners is not None:
            assert self.historical_spawners.shape[0] == self.n_bootstraps, \
                f"Historical data must have same number of bootstraps ({self.n_bootstraps})"
            assert self.historical_spawners.shape[1] == memory_years, \
                f"Historical data must have {memory_years} years"
            
        print(f"Loaded {self.n_bootstraps} bootstrap samples, {self.n_years} years each")
        if self.historical_spawners is not None:
            print(f"  Using {memory_years} years of historical data from previous split")

        if mask_path is not None:
            self.mask = np.load(mask_path).astype(np.float32)
            # Zero out land/background immediately
            self.spawners[:, self.mask == 0] = 0.0
            self.recruits[:, self.mask == 0] = 0.0
        else:
            self.mask = np.ones((50, 50), dtype=np.float32)
        
        
        
        print("Applying Log-Scaling: x_scaled = log(1 + x)")
        self.spawners = np.log1p(self.spawners)
        self.recruits = np.log1p(self.recruits)

        if year_mask is not None:
            self.year_mask = year_mask  # [n_years], 0 for 2020
        else:
            self.year_mask = np.ones(n_years, dtype=np.float32)
        
        # Historical year mask (from previous split's last N years)
        if historical_year_mask is not None:
            self.historical_year_mask = historical_year_mask  # [memory_years]
        else:
            self.historical_year_mask = np.ones(memory_years, dtype=np.float32)

    def __len__(self):
        return len(self.spawners)

    def __getitem__(self, idx):
        # Determine which bootstrap and which year
        bootstrap_idx = idx // self.n_years
        relative_year_idx = idx % self.n_years
        year_idx = self.year_offset + relative_year_idx  # Absolute year index in the full timeline
        
        
        # Initialize memory bank arrays with zeros (mask will indicate validity)
        # Using 0.0 instead of -1.0 is cleaner with mask-based handling
        memory_spawners = np.zeros((self.memory_years, 50, 50), dtype=np.float32)
        memory_recruits = np.zeros((self.memory_years, 50, 50), dtype=np.float32)
        # Temporal mask: [current_year, -1yr, -2yr, -3yr, -4yr, -5yr]
        temporal_mask = np.zeros(self.memory_years + 1, dtype=np.float32)

        # Get current spawner and recruit
                # Current spawner — zero out if doing one-year-ahead
        if self.include_current_spawner:
            current_spawner = self.spawners[idx]
            temporal_mask[0] = self.year_mask[relative_year_idx]
        else:
            current_spawner = np.zeros((50, 50), dtype=np.float32)
            temporal_mask[0] = 0.0  # mask tells model this channel is invalid
        
        current_recruit = self.recruits[idx]
        
        # Fill in historical data where available (stay within same bootstrap!)
        for i in range(self.memory_years):
            lookback = i + 1  # 1, 2, 3, 4, 5 years back
            target_relative_year = relative_year_idx - lookback
            
            if target_relative_year >= 0:
                # Within current split — check year_mask
                if self.year_mask[target_relative_year] == 1.0:
                    local_idx = bootstrap_idx * self.n_years + target_relative_year
                    memory_spawners[i] = self.spawners[local_idx]
                    memory_recruits[i] = self.recruits[local_idx]
                    temporal_mask[i+1] = 1.0
                # else: year_mask is 0 (2020), leave as zeros + mask=0
                
            elif self.historical_spawners is not None:
                historical_idx = self.memory_years + target_relative_year
                if historical_idx >= 0:
                    # Check historical year mask
                    if self.historical_year_mask[historical_idx] == 1.0:
                        memory_spawners[i] = self.historical_spawners[bootstrap_idx, historical_idx]
                        memory_recruits[i] = self.historical_recruits[bootstrap_idx, historical_idx]
                        temporal_mask[i+1] = 1.0

        input_tensor = torch.cat([
            torch.tensor(current_spawner, dtype=torch.float32).unsqueeze(0),
            torch.tensor(memory_spawners, dtype=torch.float32),
            torch.tensor(memory_recruits, dtype=torch.float32)
        ], dim=0)

        target_tensor = torch.tensor(current_recruit, dtype=torch.float32).unsqueeze(0)
        temporal_mask_tensor = torch.tensor(temporal_mask, dtype=torch.float32)
        
        # per-sample validity flag (0 for 2020, 1 otherwise)
        valid_year = torch.tensor(self.year_mask[relative_year_idx], dtype=torch.float32)
        if not self.include_current_spawner and temporal_mask.sum() == 0:
            valid_year = torch.tensor(0.0, dtype=torch.float32)

        return (input_tensor, target_tensor, temporal_mask_tensor, 
                torch.tensor(year_idx, dtype=torch.long), 
                torch.tensor(self.mask, dtype=torch.float32),
                valid_year)

def get_last_n_years(spawners, recruits, n_bootstraps, n_years_total, n_years_to_extract):
    """
    Extract the last n_years_to_extract years from a dataset to use as historical context.
    
    Args:
        spawners: Transformed spawner data of shape (n_bootstraps * n_years_total, 50, 50)
        recruits: Transformed recruit data of shape (n_bootstraps * n_years_total 
        n_bootstraps: Number of bootstrap samples
        n_years_total: Total years in this dataset
        n_years_to_extract: How many years to extract (e.g., 5)
        
    Returns:
        historical_spawners: (n_bootstraps, n_years_to_extract, 50, 50)
        historical_recruits: (n_bootstraps, n_years_to_extract, 50, 50)
    """
    
    historical_spawners = np.zeros((n_bootstraps, n_years_to_extract, 50, 50), dtype=np.float32)
    historical_recruits = np.zeros((n_bootstraps, n_years_to_extract, 50, 50), dtype=np.float32)
    
    for bootstrap_idx in range(n_bootstraps):
        # Get the last n_years_to_extract years of this bootstrap
        start_year = n_years_total - n_years_to_extract
        for year_offset in range(n_years_to_extract):
            global_idx = bootstrap_idx * n_years_total + start_year + year_offset
            historical_spawners[bootstrap_idx, year_offset] = spawners[global_idx]
            historical_recruits[bootstrap_idx, year_offset] = recruits[global_idx]
    
    return historical_spawners, historical_recruits


def get_dataloaders(level='easy', batch_size=5, memory_years=5,
                    train_years=22, val_years=5, test_years=3, 
                    data_type='dummy', include_current_spawner=True):
    if data_type == 'real':
        data_dir = f"data/real/splits/real/"
        mask_path = "data/real/output/spatial_mask.npy"
        level = 'real'
    else:
        data_dir = f"data/dummy/splits/{level}/"
        mask_path = None

    # Load year masks
    train_year_mask = np.load(data_dir + f"train_year_mask_{level}.npy")
    val_year_mask = np.load(data_dir + f"val_year_mask_{level}.npy")
    test_year_mask = np.load(data_dir + f"test_year_mask_{level}.npy")

    train_spawners = np.load(data_dir + f"train_spawners_{level}.npy")
    n_bootstraps = len(train_spawners) // train_years

    # 1. Training dataset
    train_ds = CrabDataset(
        data_dir + f"train_spawners_{level}.npy",
        data_dir + f"train_recruits_{level}.npy",
        n_years=train_years, memory_years=memory_years,
        year_offset=0, mask_path=mask_path,
        year_mask=train_year_mask,
        include_current_spawner=include_current_spawner
    )

    # 2. Historical context for val
    train_hist_spawners, train_hist_recruits = get_last_n_years(
        train_ds.spawners, train_ds.recruits,
        n_bootstraps, train_years, memory_years
    )
    train_hist_year_mask = train_year_mask[-memory_years:]  # last 5 years of train

    # 3. Validation dataset
    val_ds = CrabDataset(
        data_dir + f"val_spawners_{level}.npy",
        data_dir + f"val_recruits_{level}.npy",
        n_years=val_years, memory_years=memory_years,
        historical_spawners=train_hist_spawners,
        historical_recruits=train_hist_recruits,
        year_offset=train_years, mask_path=mask_path,
        year_mask=val_year_mask,
        historical_year_mask=train_hist_year_mask,
        include_current_spawner=include_current_spawner
    )

    # 4. Historical context for test
    val_hist_spawners, val_hist_recruits = get_last_n_years(
        val_ds.spawners, val_ds.recruits,
        n_bootstraps, val_years, memory_years
    )
    val_hist_year_mask = val_year_mask[-memory_years:]  # last 5 years of val

    # 5. Test dataset
    test_ds = CrabDataset(
        data_dir + f"test_spawners_{level}.npy",
        data_dir + f"test_recruits_{level}.npy",
        n_years=test_years, memory_years=memory_years,
        historical_spawners=val_hist_spawners,
        historical_recruits=val_hist_recruits,
        year_offset=train_years + val_years, mask_path=mask_path,
        year_mask=test_year_mask,
        historical_year_mask=val_hist_year_mask,
        include_current_spawner=include_current_spawner
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader