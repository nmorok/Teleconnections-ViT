import numpy as np

spawners = np.load("data/real/output/gridded_recruits.npy")
mask = np.load("data/real/output/spatial_mask.npy")
year_mask = np.load("data/real/output/year_mask.npy")

n_boot, n_years = spawners.shape[0], spawners.shape[1]
mask_flat = mask.flatten().astype(bool)

correlations = []
for yr in range(n_years):
    if year_mask[yr] == 0:  # skip 2020
        continue
    
    fields = spawners[:, yr, :, :].reshape(n_boot, -1)[:, mask_flat]
    fields = np.log1p(fields)
    
    for _ in range(50):
        i, j = np.random.choice(n_boot, 2, replace=False)
        r = np.corrcoef(fields[i], fields[j])[0, 1]
        if not np.isnan(r):
            correlations.append(r)

print(f"Mean pairwise correlation: {np.mean(correlations):.3f}")
print(f"Median: {np.median(correlations):.3f}")
print(f"Min: {np.min(correlations):.3f}")
print(f"Max: {np.max(correlations):.3f}")
