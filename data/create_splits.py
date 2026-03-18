"""
Split data into train/validation/test sets.
"""

import numpy as np
from pathlib import Path
import os   

def create_temporal_splits(data_type='dummy', n_years=30, train_years=22, 
                           val_years=5, test_years=3, n_bootstraps=100, 
                           level='easy', lag=0):
    if data_type == "real":
        data_dir = Path("data/real/output")
        spawners = np.load(data_dir / "gridded_spawners.npy")
        recruits = np.load(data_dir / "gridded_recruits.npy")
        
        # Load year mask (0 for 2020, 1 for valid years)
        year_mask_path = data_dir / "year_mask.npy"
        if year_mask_path.exists():
            full_year_mask = np.load(year_mask_path).astype(np.float32)
        else:
            # Fallback: construct it (index 32 = 2020)
            full_year_mask = np.ones(spawners.shape[1], dtype=np.float32)
            full_year_mask[32] = 0.0
            print("⚠️  year_mask.npy not found, constructed from known 2020 index")

        n_boot, n_years_total, h, w = spawners.shape

        # PRE-ALIGN THE LAG
        aligned_spawners = spawners[:, :n_years_total - lag]
        aligned_recruits = recruits[:, lag:]
        recruit_mask = full_year_mask[lag:]  # shift mask to align with recruits
        spawner_mask = full_year_mask[:n_years_total - lag]  # same slicing for spawners
        aligned_year_mask = (recruit_mask * spawner_mask).astype(np.float32)  # combined mask for valid years

        # Temporal Split
        train_spawn = aligned_spawners[:, :train_years]
        train_recruit = aligned_recruits[:, :train_years]
        train_year_mask = aligned_year_mask[:train_years]

        val_spawn = aligned_spawners[:, train_years:train_years+val_years]
        val_recruit = aligned_recruits[:, train_years:train_years+val_years]
        val_year_mask = aligned_year_mask[train_years:train_years+val_years]

        test_spawn = aligned_spawners[:, train_years+val_years:train_years+val_years+test_years]
        test_recruit = aligned_recruits[:, train_years+val_years:train_years+val_years+test_years]
        test_year_mask = aligned_year_mask[train_years+val_years:train_years+val_years+test_years]

        # FLATTEN
        train_spawn = train_spawn.reshape(-1, h, w)
        train_recruit = train_recruit.reshape(-1, h, w)
        val_spawn = val_spawn.reshape(-1, h, w)
        val_recruit = val_recruit.reshape(-1, h, w)
        test_spawn = test_spawn.reshape(-1, h, w)
        test_recruit = test_recruit.reshape(-1, h, w)

        splits = {
            "train_spawn": train_spawn, "train_recruit": train_recruit,
            "val_spawn": val_spawn, "val_recruit": val_recruit,
            "test_spawn": test_spawn, "test_recruit": test_recruit,
            "train_year_mask": train_year_mask,
            "val_year_mask": val_year_mask,
            "test_year_mask": test_year_mask,
        }

    elif data_type == "dummy":
        # Your existing dummy logic — year_mask is all 1s
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
            "val_spawn": spawners[val_indices], "val_recruit": recruits[val_indices],
            "test_spawn": spawners[test_indices], "test_recruit": recruits[test_indices],
            "train_year_mask": np.ones(train_years, dtype=np.float32),
            "val_year_mask": np.ones(val_years, dtype=np.float32),
            "test_year_mask": np.ones(test_years, dtype=np.float32),
        }

    for k, v in splits.items():
        print(f"  {k}: {v.shape}")

    return splits


def save_splits(splits, level, directory):
    splits_dir = Path(directory) / level
    splits_dir.mkdir(parents=True, exist_ok=True)

    np.save(splits_dir / f"train_spawners_{level}.npy", splits["train_spawn"])
    np.save(splits_dir / f"train_recruits_{level}.npy", splits["train_recruit"])
    np.save(splits_dir / f"val_spawners_{level}.npy", splits["val_spawn"])
    np.save(splits_dir / f"val_recruits_{level}.npy", splits["val_recruit"])
    np.save(splits_dir / f"test_spawners_{level}.npy", splits["test_spawn"])
    np.save(splits_dir / f"test_recruits_{level}.npy", splits["test_recruit"])
    # Save year masks
    np.save(splits_dir / f"train_year_mask_{level}.npy", splits["train_year_mask"])
    np.save(splits_dir / f"val_year_mask_{level}.npy", splits["val_year_mask"])
    np.save(splits_dir / f"test_year_mask_{level}.npy", splits["test_year_mask"])

    print(f"\nSaved to: {splits_dir}\n")

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
                                         n_years=36,
                                         train_years=24, 
                                         val_years=8, 
                                         test_years=4,
                                         n_bootstraps=100, 
                                         lag=0)
    save_splits(real_splits, level='real', directory="data/real/splits")
        