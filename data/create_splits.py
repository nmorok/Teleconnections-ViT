"""
Split data into train/validation/test sets.
"""

import numpy as np
from pathlib import Path
import os   

def create_temporal_splits(data_type = 'dummy', n_years = 30, train_years = 22, val_years = 5, test_years = 3, n_bootstraps = 100, level = 'easy', lag=0):
    """
    Create temporal splits for spawner and recruit data.
     
     Args:
         data_type: str, "real" or "dummy"
         train_years: int, number of years for training set
         val_years: int, number of years for validation set
         test_years: int, number of years for test set
         n_bootstraps: int, number of bootstrap samples
    """

    

    # Load data
    if data_type == "real":
        data_dir = Path("data/real/output")
        spawners = np.load(data_dir / "gridded_spawners.npy")
        recruits = np.load(data_dir / "gridded_recruits.npy")

        n_boot, n_years_total, h, w = spawners.shape
        
        # PRE-ALIGN THE LAG
        # Spawners: Years 0 to 25. Recruits: Years 4 to 29.
        aligned_spawners = spawners[:, :n_years_total - lag]
        aligned_recruits = recruits[:, lag:]
        
        # Now we have [n_bootstraps, 26, 50, 50] perfectly matched pairs
        
        # Temporal Split
        train_spawn = aligned_spawners[:, :train_years]
        train_recruit = aligned_recruits[:, :train_years]
        
        val_spawn = aligned_spawners[:, train_years : train_years+val_years]
        val_recruit = aligned_recruits[:, train_years : train_years+val_years]
        
        test_spawn = aligned_spawners[:, train_years+val_years : train_years+val_years+test_years]
        test_recruit = aligned_recruits[:, train_years+val_years : train_years+val_years+test_years]
        
        # FLATTEN TO 3D [N_samples, 50, 50] (Matches Dummy Data format)
        train_spawn = train_spawn.reshape(train_spawn.shape[0] * train_spawn.shape[1], h, w)
        train_recruit = train_recruit.reshape(train_recruit.shape[0] * train_recruit.shape[1], h, w)
        val_spawn = val_spawn.reshape(val_spawn.shape[0] * val_spawn.shape[1], h, w)
        val_recruit = val_recruit.reshape(val_recruit.shape[0] * val_recruit.shape[1], h, w)
        test_spawn = test_spawn.reshape(test_spawn.shape[0] * test_spawn.shape[1], h, w)
        test_recruit = test_recruit.reshape(test_recruit.shape[0] * test_recruit.shape[1], h, w)

        splits = {
            "train_spawn": train_spawn, "train_recruit": train_recruit,
            "val_spawn": val_spawn,     "val_recruit": val_recruit,
            "test_spawn": test_spawn,   "test_recruit": test_recruit
        }

    elif data_type == "dummy":
        # (Your existing dummy logic remains exactly the same here)
        data_dir = Path("data/dummy/output")
        spawners = np.load(data_dir / f"gmrf_spawners_50x50_{level}.npy")
        recruits = np.load(data_dir / f"gmrf_recruits_50x50_{level}.npy")
        
        n_samples = len(spawners)
        n_years = n_samples // n_bootstraps
        
        train_indices, val_indices, test_indices = [], [], []
        for b in range(n_bootstraps):
            offset = b * n_years
            train_indices.extend(range(offset, offset + train_years))
            val_indices.extend(range(offset + train_years, offset + train_years + val_years))
            test_indices.extend(range(offset + train_years + val_years, offset + n_years))

        splits = {
            "train_spawn": spawners[train_indices], "train_recruit": recruits[train_indices],
            "val_spawn": spawners[val_indices],     "val_recruit": recruits[val_indices],
            "test_spawn": spawners[test_indices],   "test_recruit": recruits[test_indices]
        }

    for k, v in splits.items():
        print(f"  {k}: {v.shape} samples")

    return splits



def save_splits(splits, level, directory):
    """Save splits to directory."""
    # Save splits

    splits_dir = Path(directory) / level
    splits_dir.mkdir(parents=True, exist_ok=True)

    np.save(splits_dir / f"train_spawners_{level}.npy", splits["train_spawn"])
    np.save(splits_dir / f"train_recruits_{level}.npy", splits["train_recruit"])
    np.save(splits_dir / f"val_spawners_{level}.npy", splits["val_spawn"])
    np.save(splits_dir / f"val_recruits_{level}.npy", splits["val_recruit"])
    np.save(splits_dir / f"test_spawners_{level}.npy", splits["test_spawn"])
    np.save(splits_dir / f"test_recruits_{level}.npy", splits["test_recruit"])

    print(f"\nSaved to: {splits_dir}")
    print("✓ Ready for model training!")

if __name__ == "__main__":
    for level in ['easy', 'medium', 'hard']:
    # Create splits for dummy data
        splits = create_temporal_splits(
            data_type='dummy',
            n_years=30,
            train_years=18,
            val_years=9,
            test_years=3,
            n_bootstraps=100,
            level = level
    )

        # Save splits
        save_splits(splits, level, directory="data/dummy/splits")
    
    real_splits = create_temporal_splits(data_type='real', 
                                         n_years=34,
                                         train_years=22, 
                                         val_years=9, 
                                         test_years=3,
                                         n_bootstraps=100, 
                                         lag=0)
    save_splits(real_splits, level='real', directory="data/real/splits")
        