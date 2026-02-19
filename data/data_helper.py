import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CrabDataset(Dataset):
    def __init__(self, spawner_path, recruit_path, n_years, memory_years = 5,
                 historical_spawners = None,
                 historical_recruits = None,
                 spawner_max=None, recruit_max=None, transform="max",
                 year_offset=0):
        """
        Args:
            spawner_max (float, optional): Force a specific max value for scaling. 
                                           If None, calculates max from current data.
            recruit_max (float, optional): Force a specific max value for scaling.
                                           If None, calculates max from current data.
        """
        # 1. Load Data
        self.transform = transform
        self.year_offset = year_offset
        self.spawners = np.load(spawner_path).astype(np.float32)
        self.recruits = np.load(recruit_path).astype(np.float32)
        self.n_years = n_years
        self.memory_years = memory_years
        # Store historical data from previous split
        self.historical_spawners = historical_spawners  # (n_bootstraps, 5, 50, 50) or None
        self.historical_recruits = historical_recruits  # (n_bootstraps, 5, 50, 50) or None

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
        
        # 2. Handle Spawner Max or log scale
        if transform == "max":
            if spawner_max is None:
                # Case A: Training (Calculate from self)
                self.spawner_max = np.max(self.spawners) + 1e-6
                self.spawners = self.spawners / self.spawner_max 
                print(f"Computed new Spawner Max: {self.spawner_max:.4f}")
            else:
                # Case B: Validation (Use provided value)
                self.spawner_max = spawner_max
                print(f"Using provided Spawner Max: {self.spawner_max:.4f}")
                self.spawners = self.spawners / self.spawner_max 

            # 3. Handle Recruit Max or log scale
            if recruit_max is None:
                # Case A: Training (Calculate from self)
                self.recruit_max = np.max(self.recruits) + 1e-6
                print(f"Computed new Recruit Max: {self.recruit_max:.4f}")
                self.recruits = self.recruits / self.recruit_max  
            else:
                # Case B: Validation (Use provided value)
                self.recruit_max = recruit_max
                print(f"Using provided Recruit Max: {self.recruit_max:.4f}")
                self.recruits = self.recruits / self.recruit_max  
        elif transform == "log":
            print("Applying Log-Scaling: x_scaled = log(1 + x)")
            self.spawners = np.log1p(self.spawners)
            self.recruits = np.log1p(self.recruits)
            self.spawner_max = 1.0 
            self.recruit_max = 1.0

    def __len__(self):
        return len(self.spawners)

    def __getitem__(self, idx):
        # Determine which bootstrap and which year
        bootstrap_idx = idx // self.n_years
        relative_year_idx = idx % self.n_years
        year_idx = self.year_offset + relative_year_idx  # Absolute year index in the full timeline
        

    
        
        # Get current spawner and recruit
        current_spawner = self.spawners[idx]
        current_recruit = self.recruits[idx] 
        
        # Initialize memory bank arrays with zeros (mask will indicate validity)
        # Using 0.0 instead of -1.0 is cleaner with mask-based handling
        memory_spawners = np.zeros((self.memory_years, 50, 50), dtype=np.float32)
        memory_recruits = np.zeros((self.memory_years, 50, 50), dtype=np.float32)
        # Temporal mask: [current_year, -1yr, -2yr, -3yr, -4yr, -5yr]
        temporal_mask = np.zeros(self.memory_years + 1, dtype=np.float32)
        temporal_mask[0] = 1.0  # Current year is always valid
        
        # Fill in historical data where available (stay within same bootstrap!)
        for i in range(self.memory_years):
            lookback = i + 1  # 1, 2, 3, 4, 5 years back
            historical_year = year_idx - lookback
            
            if historical_year >= 0:
                # Calculate flat index for same bootstrap, historical year
                historical_idx = bootstrap_idx * self.n_years + historical_year
                
                # Get historical data
                memory_spawners[i] = self.spawners[historical_idx] 
                memory_recruits[i] = self.recruits[historical_idx]
                temporal_mask[i+1] = 1.0  # Mark this year as valid
            elif self.historical_spawners is not None and self.historical_recruits is not None:
                # Historical is in previous split
                # historical_year is negative, so -1 means last year of previous split, -2 means second-to-last year, etc.
                historical_idx = self.memory_years + historical_year # convert to positive index (0 to memory_years-1)
                if historical_idx >= 0:
                    memory_spawners[i] = self.historical_spawners[bootstrap_idx, historical_idx]
                    memory_recruits[i] = self.historical_recruits[bootstrap_idx, historical_idx] 
                    temporal_mask[i+1] = 1.0  # Mark this year as valid   
        
        # Stack all channels: [current_spawner, 5x spawner history, 5x recruit history]
        # Total: 1 + 5 + 5 = 11 channels
        input_tensor = torch.cat([
            torch.tensor(current_spawner, dtype=torch.float32).unsqueeze(0),  # [1, 50, 50]
            torch.tensor(memory_spawners, dtype=torch.float32),                # [5, 50, 50]
            torch.tensor(memory_recruits, dtype=torch.float32)                 # [5, 50, 50]
        ], dim=0)  # Result: [11, 50, 50]
        
        target_tensor = torch.tensor(current_recruit, dtype=torch.float32).unsqueeze(0)  # [1, 50, 50]
        temporal_mask_tensor = torch.tensor(temporal_mask, dtype=torch.float32)  # [5]
        
        return (input_tensor, target_tensor, temporal_mask_tensor, torch.tensor(year_idx, dtype=torch.long))
    

def get_last_n_years(spawners, recruits, n_bootstraps, n_years_total, n_years_to_extract):
    """
    Extract the last n_years_to_extract years from a dataset to use as historical context.
    
    Args:
        spawner_path: Path to spawner data
        recruit_path: Path to recruit data  
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
                   train_years=22, val_years=5, test_years=3, transform="log"):
    """
    Helper function to initialize the split loaders with proper temporal continuity.
    
    Args:
        batch_size: Batch size for training
        memory_years: Years of history to include (default 5)
        train_years: Number of years in training set (default 22, years 0-21)
        val_years: Number of years in validation set (default 5, years 22-26)
        test_years: Number of years in test set (default 3, years 27-29)
        transform: Whether to apply log transform (default "log")
    """
    data_dir = f"data/dummy/splits/{level}/"
    
    # First, determine number of bootstraps from training data
    train_spawners = np.load(data_dir + f"train_spawners_{level}.npy")
    n_bootstraps = len(train_spawners) // train_years
    print(f"Detected {n_bootstraps} bootstrap samples")

    # 1. Training dataset (no historical data needed)
    train_ds = CrabDataset(
        data_dir + f"train_spawners_{level}.npy", 
        data_dir + f"train_recruits_{level}.npy",
        n_years=train_years,
        memory_years=memory_years,
        historical_spawners=None,
        historical_recruits=None,
        spawner_max=None, 
        recruit_max=None,
        transform=transform,
        year_offset=0
    )
    
    # 2. Extract last 5 years from training for validation historical context
    print("\nExtracting historical context from training data for validation...")
    train_hist_spawners, train_hist_recruits = get_last_n_years(
        train_ds.spawners, train_ds.recruits,
        n_bootstraps=n_bootstraps,
        n_years_total=train_years,
        n_years_to_extract=memory_years
    )
    
    # 3. Validation dataset (with training history)
    val_ds = CrabDataset(
        data_dir + f"val_spawners_{level}.npy", 
        data_dir + f"val_recruits_{level}.npy",
        n_years=val_years,
        memory_years=memory_years,
        historical_spawners=train_hist_spawners,
        historical_recruits=train_hist_recruits,
        spawner_max=train_ds.spawner_max, 
        recruit_max=train_ds.recruit_max,
        transform=transform,
        year_offset=train_years  # Validation years start after training years
    )
    
    # 4. Extract last 5 years from validation for test historical context
    print("\nExtracting historical context from validation data for test...")
    val_hist_spawners, val_hist_recruits = get_last_n_years(
        val_ds.spawners, val_ds.recruits,
        n_bootstraps=n_bootstraps,
        n_years_total=val_years,
        n_years_to_extract=memory_years
    )
    
    # 5. Test dataset (with validation history)
    test_ds = CrabDataset(
        data_dir + f"test_spawners_{level}.npy", 
        data_dir + f"test_recruits_{level}.npy",
        n_years=test_years,
        memory_years=memory_years,
        historical_spawners=val_hist_spawners,
        historical_recruits=val_hist_recruits,
        spawner_max=train_ds.spawner_max, 
        recruit_max=train_ds.recruit_max,
        transform=transform,
        year_offset=train_years + val_years  # Test years start after training + validation years
    )
    print("\nData shapes:")
    print(train_hist_spawners.shape, train_hist_recruits.shape)
    print(val_hist_spawners.shape, val_hist_recruits.shape, "\n")

    print("Sample input shape (train):", train_ds[0][0].shape)  # Should be [11, 50, 50]
    print("Sample target shape (train):", train_ds[0][1].shape)

    print("Mask shape (train):", train_ds[0][2].shape)  # Should be [6]
    print("Year index (train):", train_ds[0][3].item()) #should be 0-4 for val

    # Create loaders 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader