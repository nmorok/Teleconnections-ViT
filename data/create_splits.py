"""
Split data into train/validation/test sets.
"""

import numpy as np
from pathlib import Path

def create_temporal_splits(data_type = 'dummy', n_years = 30, train_years = 22, val_years = 5, test_years = 3, n_bootstraps = 100):
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
        data_dir = Path("data/processed/output")
        spawners = np.load(data_dir / "spde_spawners.npy")
        recruits = np.load(data_dir / "spde_recruits.npy")

    elif data_type == "dummy":
        data_dir = Path("data/dummy/output")
        spawners = np.load(data_dir / "gmrf_spawners_50x50.npy")
        recruits = np.load(data_dir / "gmrf_recruits_50x50.npy")

    n_samples = len(spawners)

    n_years = n_samples // n_bootstraps

    print(f"Total samples: {n_samples}")
    print(f"Years: {n_years}, Bootstraps: {n_bootstraps}")

    # Temporal split (by year, keeping all bootstraps)
    # Train: years 1-20, Val: years 21-24, Test: years 25-28
    train_years = train_years
    val_years = val_years
    test_years = n_years - train_years - val_years

    # Create indices
    train_indices = []
    val_indices = []
    test_indices = []

    for b in range(n_bootstraps):
        offset = b * n_years
        train_indices.extend(range(offset, offset + train_years))
        val_indices.extend(range(offset + train_years, offset + train_years + val_years))
        test_indices.extend(range(offset + train_years + val_years, offset + n_years))

    # Split data
    train_spawn = spawners[train_indices]
    train_recruit = recruits[train_indices]

    val_spawn = spawners[val_indices]
    val_recruit = recruits[val_indices]

    test_spawn = spawners[test_indices]
    test_recruit = recruits[test_indices]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_spawn)} samples")
    print(f"  Val:   {len(val_spawn)} samples")
    print(f"  Test:  {len(test_spawn)} samples")

    return {
        "train_spawn": train_spawn, "train_recruit": train_recruit,
        "val_spawn": val_spawn, "val_recruit": val_recruit,
        "test_spawn": test_spawn, "test_recruit": test_recruit
    }


def save_splits(splits, directory):
    """Save splits to directory."""
    # Save splits

    splits_dir = Path(directory)
    splits_dir.mkdir(parents=True, exist_ok=True)

    np.save(splits_dir / "train_spawners.npy", splits["train_spawn"])
    np.save(splits_dir / "train_recruits.npy", splits["train_recruit"])
    np.save(splits_dir / "val_spawners.npy", splits["val_spawn"])
    np.save(splits_dir / "val_recruits.npy", splits["val_recruit"])
    np.save(splits_dir / "test_spawners.npy", splits["test_spawn"])
    np.save(splits_dir / "test_recruits.npy", splits["test_recruit"])

    print(f"\nSaved to: {splits_dir}")
    print("✓ Ready for model training!")

def load_splits(directory):
    """Load splits from directory."""
    splits_dir = Path(directory)

    train_spawn = np.load(splits_dir / "train_spawners.npy")
    train_recruit = np.load(splits_dir / "train_recruits.npy")
    val_spawn = np.load(splits_dir / "val_spawners.npy")
    val_recruit = np.load(splits_dir / "val_recruits.npy")
    test_spawn = np.load(splits_dir / "test_spawners.npy")
    test_recruit = np.load(splits_dir / "test_recruits.npy")

    return {
        "train_spawn": train_spawn, "train_recruit": train_recruit,
        "val_spawn": val_spawn, "val_recruit": val_recruit,
        "test_spawn": test_spawn, "test_recruit": test_recruit
    }

if __name__ == "__main__":
    # Create splits for dummy data
    splits = create_temporal_splits(
        data_type='dummy',
        n_years=30,
        train_years=22,
        val_years=5,
        test_years=3,
        n_bootstraps=100
    )

    # Save splits
    save_splits(splits, directory="data/dummy/splits")