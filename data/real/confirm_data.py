"""
Verify gridded EBS data in Python.
Load .npy files from R pipeline and recreate sanity check plots.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import os

# ==============================================================================
# CONFIGURATION — update paths for your setup
# ==============================================================================

DATA_DIR = 'C:/Users/nmorok/Documents/Thesis/Teleconnections-ViT/data/real/output'

# ==============================================================================
# LOAD DATA
# ==============================================================================

def load_gridded_data(data_dir=DATA_DIR):
    spawners = np.load(os.path.join(data_dir, 'gridded_spawners.npy'))
    recruits = np.load(os.path.join(data_dir, 'gridded_recruits.npy'))
    mask = np.load(os.path.join(data_dir, 'spatial_mask.npy'))

    with open(os.path.join(data_dir, 'grid_metadata.json')) as f:
        meta = json.load(f)
    
    meta['spawner_years'] = list(range(36))  # 0 to 35
    meta['recruit_years'] = list(range(36)) 

    print(f"Spawners: {spawners.shape}")  # [n_boot, n_years, pad_ny, pad_nx]
    print(f"Recruits: {recruits.shape}")
    print(f"Mask:     {mask.shape}, valid cells: {int(mask.sum())}")
    print(f"Spawner years: {meta['spawner_years'][0]}-{meta['spawner_years'][-1]} ({meta['n_spawner_years']} yrs)")
    print(f"Recruit years: {meta['recruit_years'][0]}-{meta['recruit_years'][-1]} ({meta['n_recruit_years']} yrs)")
    print(f"Cell size: {meta['cellsize_km']} km")
    print(f"Bootstraps: {meta['n_bootstraps']}")

    return spawners, recruits, mask, meta


# ==============================================================================
# PLOT 1: MASK
# ==============================================================================

def plot_mask(mask):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(mask, cmap='Blues', vmin=0, vmax=1, origin='upper')
    ax.set_title(f'Spatial Mask: {int(mask.sum())}/{mask.size} valid cells ({100*mask.mean():.1f}%)')
    ax.set_xlabel('Column (X)')
    ax.set_ylabel('Row (Y)')
    plt.tight_layout()
    plt.show()


# ==============================================================================
# PLOT 2: SINGLE YEAR FIELD 
# ==============================================================================

def plot_single_year(spawners, recruits, mask, meta, yr_idx=14, boot_idx=0):
    """Plot one year of spawner and recruit fields."""
    yr = meta['spawner_years'][yr_idx]

    s_field = spawners[boot_idx, yr_idx]
    r_field = recruits[boot_idx, yr_idx]

    # Mask out invalid cells for display (set to NaN)
    s_display = np.where(mask > 0, np.log1p(s_field), np.nan)
    r_display = np.where(mask > 0, np.log1p(r_field), np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(s_display, cmap='viridis', origin='upper', vmin=0, vmax=8.0)
    axes[0].set_title(f'Spawner — Year {yr} (Boot {boot_idx})')
    plt.colorbar(im0, ax=axes[0], label='log1p(density)')

    im1 = axes[1].imshow(r_display, cmap='plasma', origin='upper', vmin=0, vmax=8.0)
    axes[1].set_title(f'Recruit — Year {yr} (Boot {boot_idx})')
    plt.colorbar(im1, ax=axes[1], label='log1p(density)')

    for ax in axes:
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_facecolor('whitesmoke') # Gives a nice background to NaN values

    plt.tight_layout()
    plt.show()


# ==============================================================================
# PLOT 3: MEAN FIELD vs SINGLE BOOTSTRAP 
# ==============================================================================

def plot_mean_vs_bootstrap(spawners, recruits, mask, meta, yr_idx=14):
    yr = meta['spawner_years'][yr_idx]
    n_boot = spawners.shape[0]

    np.random.seed(42)
    rand_b = np.random.randint(0, n_boot)

    mean_s = spawners[:, yr_idx].mean(axis=0)
    mean_r = recruits[:, yr_idx].mean(axis=0)
    rand_s = spawners[rand_b, yr_idx]
    rand_r = recruits[rand_b, yr_idx]

    fields = [mean_s, rand_s, mean_r, rand_r]
    titles = [
        f'Spawner Mean — {yr}',
        f'Spawner Boot {rand_b} — {yr}',
        f'Recruit Mean — {yr}',
        f'Recruit Boot {rand_b} — {yr}',
    ]
    cmaps = ['viridis', 'viridis', 'plasma', 'plasma']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, field, title, cmap in zip(axes.flat, fields, titles, cmaps):
        display = np.where(mask > 0, np.log1p(field), np.nan)
        im = ax.imshow(display, cmap=cmap, origin='upper', vmin=0, vmax=8.0)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_facecolor('whitesmoke')
        plt.colorbar(im, ax=ax, label='log1p(density)')

    plt.tight_layout()
    plt.show()


# ==============================================================================
# PLOT 4: TIME SERIES (total abundance per year)
# ==============================================================================

def plot_abundance_timeseries(spawners, recruits, mask, meta):
    years_s = np.array(meta['spawner_years'])
    years_r = np.array(meta['recruit_years'])

    # Vectorized sum of density within mask for each year/bootstrap
    mask_bool = mask > 0
    s_sums = spawners[:, :, mask_bool].sum(axis=2)
    r_sums = recruits[:, :, mask_bool].sum(axis=2)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, sums, years, title, color in [
        (axes[0], s_sums, years_s, 'Total Spawner Abundance', 'green'),
        (axes[1], r_sums, years_r, 'Total Recruit Abundance', 'blue'),
    ]:
        median = np.median(sums, axis=0)
        p25 = np.percentile(sums, 25, axis=0)
        p75 = np.percentile(sums, 75, axis=0)
        p05 = np.percentile(sums, 5, axis=0)
        p95 = np.percentile(sums, 95, axis=0)

        # Plot confidence intervals
        ax.fill_between(years, p05, p95, alpha=0.1, color=color, label='5th-95th Pct')
        ax.fill_between(years, p25, p75, alpha=0.2, color=color, label='25th-75th Pct')
        ax.plot(years, median, '-o', color=color, lw=2, ms=4, label='Median')

        # Delineate Train, Validation, and Test splits
        if len(years) >= 30:
            ax.axvspan(years[0], years[21], color='gray', alpha=0.1, label='Training (0-21)')
            ax.axvspan(years[22], years[26], color='gold', alpha=0.1, label='Validation (22-26)')
            ax.axvspan(years[27], years[-1], color='red', alpha=0.1, label='Testing (27-29)')

        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Total Density')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)

    axes[1].set_xlabel('Year')
    plt.tight_layout()
    plt.show()


# ==============================================================================
# PLOT 5: DATA STATISTICS
# ==============================================================================

def plot_data_stats(spawners, recruits, mask):
    mask_bool = mask > 0

    s_vals = spawners[:, :, mask_bool].flatten()
    r_vals = recruits[:, :, mask_bool].flatten()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(s_vals[s_vals > 0], bins=100, color='green', alpha=0.7, log=True)
    axes[0].set_title(f'Spawner Density (nonzero)\nZero%: {100*np.mean(s_vals==0):.1f}%')
    axes[0].set_xlabel('Density')

    axes[1].hist(r_vals[r_vals > 0], bins=100, color='blue', alpha=0.7, log=True)
    axes[1].set_title(f'Recruit Density (nonzero)\nZero%: {100*np.mean(r_vals==0):.1f}%')
    axes[1].set_xlabel('Density')

    s_log = np.log1p(s_vals)
    r_log = np.log1p(r_vals)
    axes[2].hist(s_log, bins=100, color='green', alpha=0.5, label='Spawner')
    axes[2].hist(r_log, bins=100, color='blue', alpha=0.5, label='Recruit')
    axes[2].set_title('Log1p Distribution')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    print(f"\nSpawner stats (valid cells only):")
    print(f"  Mean: {s_vals.mean():.1f}, Median: {np.median(s_vals):.1f}, "
          f"Max: {s_vals.max():.1f}, Zero%: {100*np.mean(s_vals==0):.1f}%")
    print(f"Recruit stats (valid cells only):")
    print(f"  Mean: {r_vals.mean():.1f}, Median: {np.median(r_vals):.1f}, "
          f"Max: {r_vals.max():.1f}, Zero%: {100*np.mean(r_vals==0):.1f}%")
    
    # ==============================================================================
# PLOT 6: GRAND MEAN (All Years, All Bootstraps)
# ==============================================================================

def plot_grand_mean(spawners, recruits, mask):
    """Plot the average spatial field across all years and all bootstraps."""
    
    # Calculate the mean across axis 0 (bootstraps) AND axis 1 (years)
    # This reduces the 4D array [boot, year, y, x] down to 2D [y, x]
    grand_mean_s = spawners.mean(axis=(0, 1))
    grand_mean_r = recruits.mean(axis=(0, 1))

    # Mask out invalid cells (set to NaN) and apply log1p
    s_display = np.where(mask > 0, np.log1p(grand_mean_s), np.nan)
    r_display = np.where(mask > 0, np.log1p(grand_mean_r), np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Spawner Plot
    im0 = axes[0].imshow(s_display, cmap='viridis', origin='upper', vmin=0, vmax=8.0)
    axes[0].set_title('Grand Mean Spawners\n(All Years & Bootstraps)', fontsize=14, fontweight='bold')
    plt.colorbar(im0, ax=axes[0], label='log1p(Mean Density)')

    # Recruit Plot
    im1 = axes[1].imshow(r_display, cmap='plasma', origin='upper', vmin=0, vmax=8.0)
    axes[1].set_title('Grand Mean Recruits\n(All Years & Bootstraps)', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=axes[1], label='log1p(Mean Density)')

    # Formatting
    for ax in axes:
        ax.set_xlabel('Grid X (West -> East)')
        ax.set_ylabel('Grid Y (North -> South)')
        ax.set_facecolor('whitesmoke') # Grey background for the NaNs outside the mask

    plt.tight_layout()
    plt.show()

import matplotlib.animation as animation

# ==============================================================================
# PLOT 7: TIME-SERIES ANIMATION (GIF)
# ==============================================================================

def create_yearly_animation(spawners, recruits, mask, meta, save_dir=DATA_DIR):
    """
    Creates a GIF animating the mean field year by year.
    """
    # 1. Calculate mean across bootstraps (axis 0)
    mean_s = spawners.mean(axis=0)
    mean_r = recruits.mean(axis=0)

    # 2. Pre-apply mask and log1p transform for all frames
    mask_bool = mask > 0
    s_anim_data = np.where(mask_bool, np.log1p(mean_s), np.nan)
    r_anim_data = np.where(mask_bool, np.log1p(mean_r), np.nan)

    years = meta['spawner_years']
    n_years = len(years)

    # 3. Setup Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    # Initialize first frame
    im0 = axes[0].imshow(s_anim_data[0], cmap='viridis', origin='upper', vmin=0, vmax=8.0)
    axes[0].set_title("Spawners", fontsize=14)
    axes[0].set_facecolor('whitesmoke')
    plt.colorbar(im0, ax=axes[0], label='log1p(Mean Density)')

    im1 = axes[1].imshow(r_anim_data[0], cmap='plasma', origin='upper', vmin=0, vmax=8.0)
    axes[1].set_title("Recruits", fontsize=14)
    axes[1].set_facecolor('whitesmoke')
    plt.colorbar(im1, ax=axes[1], label='log1p(Mean Density)')

    for ax in axes:
        ax.set_xlabel('Grid X (West -> East)')
        ax.set_ylabel('Grid Y (North -> South)')

    # Add extra space at the top so the title doesn't get clipped
    plt.tight_layout(rect=[0, 0.03, 1, 0.90]) 

    # 4. Create a prominent, updating title
    title_text = fig.suptitle(f"Year: {years[0]}  |  Index: 0/{n_years-1}", 
                              fontsize=18, fontweight='bold')

    # 5. Update function for the animation
    def update(frame):
        im0.set_data(s_anim_data[frame])
        im1.set_data(r_anim_data[frame])
        # Update both the exact year and the 0-33 index
        title_text.set_text(f"Year: {years[frame]}  |  Index: {frame}/{n_years-1}")
        return [im0, im1, title_text]

    # 6. Render Animation (blit=False ensures the title text updates every frame)
    anim = animation.FuncAnimation(fig, update, frames=n_years, interval=600, blit=False)

    # 7. Save as GIF
    save_path = os.path.join(save_dir, "ebs_crab_dynamics.gif")
    print(f"Saving animation to {save_path} ...")
    
    anim.save(save_path, writer='pillow', fps=2) 
    print("Animation saved successfully!")
    
    plt.close(fig)

# ==============================================================================
# RUN ALL
# ==============================================================================

if __name__ == '__main__':
    print("Loading data...")
    spawners, recruits, mask, meta = load_gridded_data()

    print("\nGenerating Spatial Mask Plot...")
    plot_mask(mask)

    print("\nGenerating Single Year Plot...")
    plot_single_year(spawners, recruits, mask, meta)

    print("\nGenerating Mean vs Bootstrap Comparison...")
    plot_mean_vs_bootstrap(spawners, recruits, mask, meta)

    print("\nGenerating Abundance Time Series...")
    plot_abundance_timeseries(spawners, recruits, mask, meta)

    print("\nGenerating Data Statistics...")
    plot_data_stats(spawners, recruits, mask)

    print("\nGenerating Grand Mean Plot...")
    plot_grand_mean(spawners, recruits, mask)

    print("\nGenerating Time-Series Animation...")
    create_yearly_animation(spawners, recruits, mask, meta)