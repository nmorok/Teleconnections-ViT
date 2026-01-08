"""
Split data into train/validation/test sets.
"""

import numpy as np
from pathlib import Path

configs = {
    "data_type": "dummy",  # "real" or "dummy"
    "n_bootstraps": 100,
    "train_years": 20,
    "val_years": 4,
    "test_years": 4
}
# Load data


if configs["data_type"] == "real":
    data_dir = Path("data/processed/output")
    spawners = np.load(data_dir / "spde_spawners.npy")
    recruits = np.load(data_dir / "spde_recruits.npy")

elif configs["data_type"] == "dummy":
    data_dir = Path("data/dummy/output")
    spawners = np.load(data_dir / "gmrf_spawners_10x10.npy")
    recruits = np.load(data_dir / "gmrf_recruits_10x10.npy")

n_samples = len(spawners)

n_years = n_samples // configs["n_bootstraps"]

print(f"Total samples: {n_samples}")
print(f"Years: {n_years}, Bootstraps: {configs['n_bootstraps']}")

# Temporal split (by year, keeping all bootstraps)
# Train: years 1-20, Val: years 21-24, Test: years 25-28
train_years = configs["train_years"]
val_years = configs["val_years"]
test_years = n_years - train_years - val_years

# Create indices
train_indices = []
val_indices = []
test_indices = []

for b in range(configs["n_bootstraps"]):
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

# Save splits
if configs["data_type"] == "real":
    splits_dir = Path("data/processed/splits")
elif configs["data_type"] == "dummy":
    splits_dir = Path("data/dummy/splits")
splits_dir.mkdir(parents=True, exist_ok=True)

np.save(splits_dir / "train_spawners.npy", train_spawn)
np.save(splits_dir / "train_recruits.npy", train_recruit)
np.save(splits_dir / "val_spawners.npy", val_spawn)
np.save(splits_dir / "val_recruits.npy", val_recruit)
np.save(splits_dir / "test_spawners.npy", test_spawn)
np.save(splits_dir / "test_recruits.npy", test_recruit)

print(f"\nSaved to: {splits_dir}")
print("✓ Ready for model training!")