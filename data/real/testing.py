import numpy as np
import matplotlib.pyplot as plt
import json

# 1. Define paths (adjust if your Python script is in a different folder)
output_dir = "C:/Users/nmorok/Documents/Thesis/Teleconnections-ViT/data/real/output/"
spawners_file = output_dir + "gridded_spawners.npy"
mask_file = output_dir + "spatial_mask.npy"
meta_file = output_dir + "grid_metadata.json"

# 2. Load the data
spawners = np.load(spawners_file)
mask = np.load(mask_file)

with open(meta_file, 'r') as f:
    meta = json.load(f)

print(f"Raw Python array shape: {spawners.shape}") 
# Expected: (n_bootstraps, n_years, pad_ny, pad_nx)

# 3. Extract the exact same test frame you checked in R (Boot 1, Year 15)
# Remember: Python is 0-indexed, so Boot 1 = 0, Year 15 = 14
test_field = spawners[0, 14, :, :] 

# 4. Check the Max Location to compare with R
# If R said max was at [row, col] = [12, 35], Python should say [11, 34].
# If Python says [34, 11], your X and Y axes got swapped by reticulate!
max_loc = np.unravel_index(np.argmax(test_field), test_field.shape)
print(f"Python max location (Row, Col): {max_loc}")

# 5. Fix the orientation (The "R-to-Python" Fix)
# If your axes are swapped, uncomment the line below to transpose the spatial axes:
# spawners = np.transpose(spawners, (0, 1, 3, 2)) 
# test_field = spawners[0, 14, :, :]

# 6. Mask out zeros so they plot clearly (like NA in R)
test_field_masked = np.where(mask == 1, test_field, np.nan)

# Now apply log1p. (np.log1p handles NaNs gracefully by keeping them as NaNs)
log_field = np.log1p(test_field_masked)

# 7. Plot it!
plt.figure(figsize=(10, 6))

# origin='upper' puts Row 0 at the top (North) - typical for ViT inputs
# origin='lower' puts Row 0 at the bottom (South) - typical for maps
plt.imshow(log_field, cmap='magma', origin='upper', vmin=0, vmax=8)

# Optional: You can set the 'bad' color (NaNs) to a specific background color like light grey
plt.gca().set_facecolor('whitesmoke') 

plt.colorbar(label='log1p(Density)')
plt.title(f"Python View: Spawners (Boot 0, Year {meta['spawner_years'][14]})")
plt.xlabel("Grid X (West -> East)")
plt.ylabel("Grid Y (North -> South)")
plt.show()