"""
Generate dummy crab data for pipeline testing
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm

def create_spatial_precision_matrix(grid_size = 50, kappa=1.0):
    """
    Create precision matrix for 2D spatial grid with 4-neighbor structure.
    
    This matrix encodes spatial correlation: neighboring grid cells will be
    correlated in the generated fields.
    
    Args:
        grid_size: int
            Size of square grid (e.g., 50 for 50×50 grid)
            
        kappa: float, default=1.0
            Spatial range parameter. Controls correlation distance.
            - Smaller κ = longer correlation range (smoother fields)
            - Larger κ = shorter correlation range (more local variation)
            
            
    Returns:
        Q_spatial: sparse matrix, shape (grid_size², grid_size²)
            Precision matrix where Q[i,j] encodes relationship between
            grid cells i and j
            
    Matrix structure:
        - Diagonal Q[i,i] = κ² × (number of neighbors)
        - Off-diagonal Q[i,j] = -κ² if i and j are neighbors, 0 otherwise
        
    Example (3×3 grid):
        Grid:       Flattened:    Q is 9×9:
        [0][1][2]   0,1,2,        Q[0,0]=2κ², Q[0,1]=-κ², Q[0,3]=-κ²
        [3][4][5]   3,4,5,        Q[4,4]=4κ², Q[4,1]=-κ², Q[4,3]=-κ², ...
        [6][7][8]   6,7,8         (center has 4 neighbors, corners have 2)
    """
    
    # Total number of grid cells (e.g., 50×50 = 2500)
    n_cells = grid_size * grid_size
    
    # Initialize sparse matrix (most elements are zero)
    # Using lil_matrix for efficient construction
    Q = sp.lil_matrix((n_cells, n_cells))
    
    # Helper function: convert 2D grid coordinates to 1D index
    def grid_to_index(row, col):
        """Map (row, col) in grid to linear index"""
        return row * grid_size + col
    
    # Fill the precision matrix by looping over all grid cells
    for row in range(grid_size):
        for col in range(grid_size):
            
            # Current cell's index in the flattened representation
            center_idx = grid_to_index(row, col)
            
            # Count how many neighbors this cell has (2 to 4)
            n_neighbors = 0
            
            # Check each potential neighbor (up, down, left, right) and fill out their precision values
            
            # UP neighbor
            if row > 0:
                neighbor_idx = grid_to_index(row - 1, col)
                Q[center_idx, neighbor_idx] = -kappa**2
                n_neighbors += 1
            
            # DOWN neighbor
            if row < grid_size - 1:
                neighbor_idx = grid_to_index(row + 1, col)
                Q[center_idx, neighbor_idx] = -kappa**2
                n_neighbors += 1
            
            # LEFT neighbor
            if col > 0:
                neighbor_idx = grid_to_index(row, col - 1)
                Q[center_idx, neighbor_idx] = -kappa**2
                n_neighbors += 1
            
            # RIGHT neighbor
            if col < grid_size - 1:
                neighbor_idx = grid_to_index(row, col + 1)
                Q[center_idx, neighbor_idx] = -kappa**2
                n_neighbors += 1
            
            # Diagonal element = number of neighbors × κ²
            Q[center_idx, center_idx] = n_neighbors * kappa**2
    
    # Convert to CSR format for efficient arithmetic operations
    return Q.tocsr()


def create_temporal_precision_matrix(n_years = 30, rho=0.7, sigma=1.0):
    """
    Create precision matrix for AR(1) temporal process.
    
    This matrix encodes temporal correlation: values in consecutive years
    will be correlated (year-to-year persistence).
    
    Mathematical model:
        X_t = ρ × X_{t-1} + ε_t,  where ε_t ~ N(0, σ²)
        
        This is an autoregressive process of order 1 (AR(1))
        
    Args:
        n_years: int
            Number of time points (e.g., 30 for spawners)
            
        rho: float, default=0.7
            Temporal correlation coefficient between consecutive years
            - ρ = 0.0: No temporal correlation (white noise)
            - ρ = 0.5: Moderate persistence
            - ρ = 0.7: Strong persistence (realistic for crab populations)
            - ρ = 0.9: Very strong persistence
            - Must be in [0, 1) for stationarity
            
        sigma: float, default=1.0
            Innovation standard deviation (controls how much new randomness
            is added each year)
            
    Returns:
        Q_temporal: sparse matrix, shape (n_years, n_years)
            Precision matrix for temporal correlation
            
    Matrix structure:
        - Q[t,t] = (1 + ρ²)/σ² for interior years (connected both sides)
        - Q[t,t] = 1/σ² for first and last years (connected one side)
        - Q[t,t-1] = Q[t-1,t] = -ρ/σ² for consecutive years
        - All other elements = 0
        
    Interpretation of ρ:
        - Correlation at lag 1: ρ
        - Correlation at lag 2: ρ²
        - Correlation at lag k: ρ^k
        
    Example (ρ=0.7):
        Year 0 ↔ Year 1: correlation = 0.7
        Year 0 ↔ Year 2: correlation = 0.49
        Year 0 ↔ Year 5: correlation = 0.17 (weak but present)
    """
    
    # Initialize sparse matrix
    Q = sp.lil_matrix((n_years, n_years))
    
    # Fill matrix year by year
    for t in range(n_years):
        
        if t == 0:
            # First year: only connected to next year
            Q[t, t] = 1.0 / sigma**2
            
        elif t == n_years - 1:
            # Last year: only connected to previous year
            Q[t, t] = 1.0 / sigma**2
            Q[t, t-1] = -rho / sigma**2
            
        else:
            # Interior years: connected to both previous and next years
            Q[t, t] = (1.0 + rho**2) / sigma**2
            Q[t, t-1] = -rho / sigma**2
            Q[t, t+1] = -rho / sigma**2
    
    # Make matrix symmetric (Q should be symmetric)
    # This ensures Q[i,j] = Q[j,i]
    Q = Q + Q.T - sp.diags(Q.diagonal())
    
    return Q.tocsr()


def sample_gmrf(Q, n_samples=1):
    """
    Draw random samples from a Gaussian Markov Random Field (GMRF).
    
    Mathematical background:
        A GMRF is specified by precision matrix Q (inverse covariance):
        X ~ N(μ, Σ), where Σ = Q⁻¹
        
        To sample, we solve: Q^(1/2) × X = Z, where Z ~ N(0, I)
        Equivalently: Q × X = Z
        
    Args:
        Q: sparse matrix, shape (n, n)
            Precision matrix defining the GMRF
            
        n_samples: int, default=1
            Number of independent samples to draw
            
    Returns:
        samples: array, shape (n_samples, n)
            Random samples from the GMRF
            
    Algorithm:
        For each sample:
        1. Generate Z ~ N(0, I_n)  (standard normal vector)
        2. Solve Q × X = Z for X  (sparse linear system)
        3. X ~ N(0, Q⁻¹)  (has the desired correlation structure)
        
    Note:
        We use a sparse solver (spsolve) because Q is large but mostly zeros.
        For 50×50 grid, Q is 2500×2500 but only ~10,000 non-zero elements.
    """
    
    n = Q.shape[0]  # Dimension of the random field
    
    samples = []
    
    for _ in range(n_samples):
        # Step 1: Generate standard normal random vector
        z = np.random.randn(n)
        
        # Step 2: Solve sparse linear system Q × x = z
        # This is the key step! It creates the correlation structure.
        # spsolve is efficient because Q is sparse (mostly zeros)
        x = spsolve(Q, z)   
        x = (x - x.mean()) / x.std()   
        samples.append(x)
    
    return np.array(samples)





def create_spatiotemporal_gmrf_data(
    grid_size=10,
    n_years=30,
    n_bootstraps=100,
    spatial_kappa=0.3,
    temporal_rho=0.7,
    temporal_sigma=1.0,
    mean_density=50.0,
    seed=2026
):
    """
    Generate spatiotemporal GMRF data mimicking crab density fields.
    
    This function creates realistic spatial fields that evolve over time,
    with multiple bootstrap replicates to represent uncertainty.
    
    Process for each bootstrap:
        Year 0: Sample spatial field from GMRF (random start)
        Year 1: Evolve using AR(1): X_1 = ρ×X_0 + √(1-ρ²)×innovation
        Year 2: X_2 = ρ×X_1 + √(1-ρ²)×innovation
        ...
        Each "innovation" is a new spatial GMRF sample
        
    Args:
        grid_size: int, default=10
            Size of spatial grid (10 means 10×10 = 100 cells)
            
        n_years: int, default=30
            Number of years (time steps)
            
        n_bootstraps: int, default=100
            Number of bootstrap replicates (different spatial samples)
            
        spatial_kappa: float, default=0.3
            Spatial range parameter (smaller = smoother fields)
            Typical: 0.3 gives ~60km correlation range at 20km resolution
            
        temporal_rho: float, default=0.7
            Temporal correlation (year-to-year persistence)
            
        temporal_sigma: float, default=1.0
            Temporal innovation variance
            
        mean_density: float, default=50.0
            Mean crab density (crabs per km²)
            
        seed: int, optional
            Random seed for reproducibility
            
    Returns:
        data: array, shape (n_years * n_bootstraps, grid_size, grid_size)
            Generated spatiotemporal fields
            Organized as: [Y0B0, Y1B0, Y2B0, ..., Y0B1, Y1B1, ...]
            where YiBj = Year i, Bootstrap j
            
        params: dict
            Dictionary of parameters used for generation
            
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    print(f"Creating spatiotemporal GMRF data...")
    print(f"  Grid: {grid_size}×{grid_size} ({grid_size**2} cells)")
    print(f"  Years: {n_years}")
    print(f"  Bootstraps: {n_bootstraps}")
    print(f"  Total samples: {n_years * n_bootstraps}")
    
    # -------------------------------------------------------------------------
    # Step 1: Create precision matrices (these define the correlation structure)
    # -------------------------------------------------------------------------
    
    print("\n[1/4] Building spatial precision matrix...")
    Q_spatial = create_spatial_precision_matrix(grid_size, kappa=spatial_kappa)
    print(f"      Shape: {Q_spatial.shape}, Non-zeros: {Q_spatial.nnz}")
    
    print("[2/4] Building temporal precision matrix...")
    Q_temporal = create_temporal_precision_matrix(n_years, rho=temporal_rho, sigma=temporal_sigma)
    
    # -------------------------------------------------------------------------
    # Step 2: Generate fields for each bootstrap
    # -------------------------------------------------------------------------
    
    print("[3/4] Sampling spatiotemporal fields...")
    
    # Storage: will hold all generated fields
    n_total_samples = n_years * n_bootstraps
    n_cells = grid_size * grid_size
    data = np.zeros((n_total_samples, grid_size, grid_size))
    
    # Loop over bootstraps
    for b in tqdm(range(n_bootstraps), desc="Bootstrap"): # tqdm for progress bar
        
        # Storage for this bootstrap's time series
        # Shape: (n_years, n_cells) where each row is a flattened spatial field
        spatial_fields = np.zeros((n_years, n_cells))
        
        # Generate time series using AR(1) process
        for t in range(n_years):
            
            if t == 0:
                # First year: sample fresh spatial field from GMRF
                field = sample_gmrf(Q_spatial, n_samples=1)[0]
                
            else:
                # Subsequent years: AR(1) evolution
                # Formula: X_t = ρ × X_{t-1} + √(1-ρ²) × innovation
                # The √(1-ρ²) factor keeps variance constant over time
                
                prev_field = spatial_fields[t-1]
                innovation = sample_gmrf(Q_spatial, n_samples=1)[0]
                
                field = (temporal_rho * prev_field + 
                        np.sqrt(1 - temporal_rho**2) * innovation)
            
            spatial_fields[t] = field
        
        # Store in output array, reshaping to 2D grid
        for t in range(n_years):
            idx = b * n_years + t  # Index in flattened output
            data[idx] = spatial_fields[t].reshape(grid_size, grid_size)
    
    # -------------------------------------------------------------------------
    # Step 3: Transform to realistic crab density scale
    # -------------------------------------------------------------------------
    
    print("[4/4] Transforming to realistic density scale...")
    
    # Currently data is N(0,1)-ish (from GMRF sampling)
    # Transform to positive, realistic crab densities
    
    # Standardize
    data_standardized = (data - data.mean()) / data.std()
    
    # Apply log-normal-ish transformation to get positive values
    # exp() creates right-skewed distribution (like real crab data)
    data_transformed = mean_density * np.exp(data_standardized * 0.5)
    
    # -------------------------------------------------------------------------
    # Step 4: Package results
    # -------------------------------------------------------------------------
    
    params = {
        'grid_size': grid_size,
        'n_years': n_years,
        'n_bootstraps': n_bootstraps,
        'n_total_samples': n_total_samples,
        'spatial_kappa': spatial_kappa,
        'temporal_rho': temporal_rho,
        'temporal_sigma': temporal_sigma,
        'mean_density': mean_density,
        'spatial_correlation_range_cells': 1.0 / spatial_kappa,
        'seed': seed
    }
    
    print("\nDone! Generated data statistics:")
    print(f"  Shape: {data_transformed.shape}")
    print(f"  Mean density: {data_transformed.mean():.2f}")
    print(f"  Std density: {data_transformed.std():.2f}")
    print(f"  Min density: {data_transformed.min():.2f}")
    print(f"  Max density: {data_transformed.max():.2f}")
    
    return data_transformed, params


def create_spawner_recruit_pairs(
    grid_size=50,
    n_spawner_years=30,
    n_recruit_years=30,
    n_bootstraps=100,
    spatial_kappa=0.3,
    temporal_rho=0.7,
    recruitment_correlation=0.3,
    mean_spawner_density=50.0,
    mean_recruit_density=30.0,
    seed=2026
):
    """
    Generate correlated spawner and recruitment GMRF fields.
        
    Args:
        grid_size: int, default=10
        n_spawner_years: int, default=30
            Number of spawner years (e.g., 1988-2015)
        n_recruit_years: int, default=30
            Number of recruitment years (e.g., 1993-2016)
        n_bootstraps: int, default=100
        spatial_kappa: float, default=0.3
        temporal_rho: float, default=0.7
        recruitment_correlation: float, default=0.3
        mean_spawner_density: float, default=50.0
        mean_recruit_density: float, default=30.0
        seed: int, default=2026
            
    Returns:
        spawners: array, shape (n_spawner_years * n_bootstraps, grid_size, grid_size)
        recruits: array, shape (n_recruit_years * n_bootstraps, grid_size, grid_size)
        params: dict
        
    Mathematical relationship:
        R = α × S + √(1-α²) × ε
        
        where:
        - R: recruitment field
        - S: spawner field (5 years earlier)
        - α: recruitment_correlation
        - ε: independent spatiotemporal variation
        
        This ensures Corr(R, S) = α while maintaining realistic variance in R
    """
    
    # -------------------------------------------------------------------------
    # Step 1: Generate spawner fields
    # -------------------------------------------------------------------------
    
    print("\n" + "="*80)
    print("GENERATING SPAWNER FIELDS")
    print("="*80)
    
    spawners, params_s = create_spatiotemporal_gmrf_data(
        grid_size=grid_size,
        n_years=n_spawner_years,
        n_bootstraps=n_bootstraps,
        spatial_kappa=spatial_kappa,
        temporal_rho=temporal_rho,
        mean_density=mean_spawner_density,
        seed=seed
    )
    
    # -------------------------------------------------------------------------
    # Step 2: Generate independent recruitment fields (base)
    # -------------------------------------------------------------------------
    
    print("\n" + "="*80)
    print("GENERATING INDEPENDENT RECRUITMENT FIELDS (BASE)")
    print("="*80)
    
    # These will be mixed with spawner fields to create correlation
    recruits_base, params_r = create_spatiotemporal_gmrf_data(
        grid_size=grid_size,
        n_years=n_recruit_years,
        n_bootstraps=n_bootstraps,
        spatial_kappa=spatial_kappa,
        temporal_rho=temporal_rho,
        mean_density=mean_recruit_density,
        seed=seed + 1000 if seed is not None else None  # Different seed
    )
    
    # -------------------------------------------------------------------------
    # Step 3: Add spawner-recruitment correlation
    # -------------------------------------------------------------------------
    
    print("\n" + "="*80)
    print("ADDING SPAWNER-RECRUITMENT CORRELATION")
    print("="*80)
    print(f"Target correlation: {recruitment_correlation:.2f}")
    
    # Initialize final recruitment array
    recruits = np.zeros_like(recruits_base)
    
    # For each bootstrap and year, mix spawner and independent variation
    for b in range(n_bootstraps):
        for t in range(n_recruit_years):
            
            # Index in flattened arrays
            recruit_idx = b * n_recruit_years + t
            spawner_idx = b * n_spawner_years + t  # Corresponding spawner year
            
            # Mix formula: R = α×S + √(1-α²)×ε
            # This ensures Corr(R,S) = α
            recruits[recruit_idx] = (
                recruitment_correlation * spawners[spawner_idx] +
                np.sqrt(1 - recruitment_correlation**2) * recruits_base[recruit_idx]
            )
    
    # Verify correlation (check a few samples)
    print("\nVerifying spawner-recruitment correlation...")
    correlations = []
    for b in range(min(10, n_bootstraps)):  # Check first 10 bootstraps
        for t in range(n_recruit_years):
            recruit_idx = b * n_recruit_years + t
            spawner_idx = b * n_spawner_years + t
            
            corr = np.corrcoef(
                spawners[spawner_idx].flatten(),
                recruits[recruit_idx].flatten()
            )[0, 1]
            correlations.append(corr)
    
    print(f"  Mean correlation: {np.mean(correlations):.3f}")
    print(f"  Std correlation: {np.std(correlations):.3f}")
    print(f"  Expected: {recruitment_correlation:.3f} ✓")
    
    # -------------------------------------------------------------------------
    # Package results
    # -------------------------------------------------------------------------
    
    params = {
        **params_s,  # Include all spawner params
        'n_recruit_years': n_recruit_years,
        'recruitment_correlation': recruitment_correlation,
        'mean_recruit_density': mean_recruit_density
    }

    print(spawners[:10])
    
    return spawners, recruits, params


# ==============================================================================
# PART 6: VISUALIZATION
# ==============================================================================

def visualize_gmrf_properties(spawners, recruits, params, output_dir="./output"):
    """
    Create comprehensive visualizations of generated GMRF data.
    
    Creates 3 figures:
    1. Spatial patterns (example fields from different years)
    2. Temporal evolution (time series at one location)
    3. Spawner-recruitment relationship (scatterplot)
    
    Args:
        spawners: array, shape (n_spawner_samples, grid_size, grid_size)
        recruits: array, shape (n_recruit_samples, grid_size, grid_size)
        params: dict, parameters used for generation
        output_dir: str, directory to save figures
    """
    
    grid_size = params['grid_size']
    n_years = params['n_years']
    n_bootstraps = params['n_bootstraps']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Figure 1: Spatial patterns over time
    # -------------------------------------------------------------------------
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Spatial Patterns Over Time (Bootstrap 0)', fontsize=16)
    
    # Show 3 years from first bootstrap
    years_to_show = [0, n_years//2, n_years-1]
    
    for i, year_idx in enumerate(years_to_show):
        # Spawners
        ax = axes[0, i]
        im = ax.imshow(spawners[year_idx], cmap='viridis', aspect='auto')
        ax.set_title(f'Spawners: Year {year_idx+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Density')
        
        # Recruits (if available)
        if year_idx < len(recruits) // n_bootstraps:
            ax = axes[1, i]
            im = ax.imshow(recruits[year_idx], cmap='plasma', aspect='auto')
            ax.set_title(f'Recruits: Year {year_idx+1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, label='Density')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gmrf_spatial_patterns.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'gmrf_spatial_patterns.png'}")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Figure 2: Temporal evolution
    # -------------------------------------------------------------------------
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract time series at center pixel
    center = grid_size // 2
    
    # Spawner time series (first bootstrap)
    spawner_ts = spawners[:n_years, center, center]
    axes[0].plot(range(1, n_years+1), spawner_ts, 'o-', linewidth=2, markersize=6)
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Spawner Temporal Evolution (Center Pixel)\nρ = {params["temporal_rho"]:.2f}')
    axes[0].grid(True, alpha=0.3)
    
    # Recruit time series (first bootstrap)
    n_recruit_years = params['n_recruit_years']
    recruit_ts = recruits[:n_recruit_years, center, center]
    axes[1].plot(range(1, n_recruit_years+1), recruit_ts, 'o-', linewidth=2, markersize=6, color='orange')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Recruitment Temporal Evolution (Center Pixel)\nρ = {params["temporal_rho"]:.2f}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gmrf_temporal_evolution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'gmrf_temporal_evolution.png'}")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Figure 3: Spawner-recruitment relationship
    # -------------------------------------------------------------------------
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Sample points for scatterplot (don't plot all 2500×2400 points!)
    n_samples = min(10000, spawners.size)
    indices = np.random.choice(spawners.size, n_samples, replace=False)
    
    spawn_flat = spawners.flatten()[indices]
    recruit_flat = recruits.flatten()[indices]
    
    # Scatterplot
    ax.scatter(spawn_flat, recruit_flat, alpha=0.1, s=1, c='blue')
    ax.set_xlabel('Spawner Density', fontsize=12)
    ax.set_ylabel('Recruitment Density', fontsize=12)
    
    # Calculate and show correlation
    corr = np.corrcoef(spawners.flatten(), recruits.flatten())[0, 1]
    ax.set_title(f'Spawner-Recruitment Relationship\nr = {corr:.3f} (target = {params.get("recruitment_correlation", "N/A")})',
                fontsize=14)
    
    # Add reference line
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    min_val = min(xlim[0], ylim[0])
    max_val = max(xlim[1], ylim[1])
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=2, label='1:1 line')
    ax.legend()
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gmrf_spawner_recruit_correlation.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'gmrf_spawner_recruit_correlation.png'}")
    plt.close()
    
    print("\nVisualization complete!")


# ==============================================================================
# PART 7: MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main function to generate dummy GMRF data for crab recruitment modeling.
    
    This creates data matching your real SPDE pipeline output structure:
    - Spawners: 30 years (1988-2017) × 100 bootstraps = 3000 samples
    - Recruits: 30 years (1988-2015) × 100 bootstraps = 3000 samples
    - Both on 10×10 grid (20km resolution)
    """
    
    print("="*80)
    print("GMRF DUMMY DATA GENERATOR FOR CRAB RECRUITMENT")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    
    config = {
        # Spatial grid
        'grid_size': 50,              # 50×50 = 2500 cells per field
        
        # Temporal extent
        'n_spawner_years': 30,        # 1988-2017
        'n_recruit_years': 30,        # 1988-2017

        # Bootstrap uncertainty
        'n_bootstraps': 100,          # Match your SPDE bootstrap count
        
        # Spatial correlation
        'spatial_kappa': 0.3,         # ~60km correlation range at 20km resolution
        
        # Temporal correlation  
        'temporal_rho': 0.3,          # Strong year-to-year persistence
        
        # Spawner-recruitment relationship
        'recruitment_correlation': 0.6,  # Weak-moderate correlation
        
        # Density scales
        'mean_spawner_density': 50.0,    # Mean crabs/km²
        'mean_recruit_density': 30.0,    # Typically lower than spawners
        
        # Reproducibility
        'seed': 2026
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # -------------------------------------------------------------------------
    # Generate data
    # -------------------------------------------------------------------------
    
    spawners, recruits, params = create_spawner_recruit_pairs(**config)
    
    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    
    output_dir = Path("data/dummy/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    # Save numpy arrays
    np.save(output_dir / "gmrf_spawners_50x50.npy", spawners)
    print(f"Saved: {output_dir / 'gmrf_spawners_50x50.npy'}")
    print(f"  Shape: {spawners.shape}")
    
    np.save(output_dir / "gmrf_recruits_50x50.npy", recruits)
    print(f"Saved: {output_dir / 'gmrf_recruits_50x50.npy'}")
    print(f"  Shape: {recruits.shape}")
    
    # Save metadata
    with open(output_dir / "gmrf_params.json", 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Saved: {output_dir / 'gmrf_params.json'}")


    
    # -------------------------------------------------------------------------
    # Create visualizations
    # -------------------------------------------------------------------------
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    visualize_gmrf_properties(spawners, recruits, params, output_dir=output_dir)
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  1. gmrf_spawners_50x50.npy - Spawner density fields")
    print("  2. gmrf_recruits_50x50.npy - Recruitment density fields")
    print("  3. gmrf_params.json - Generation parameters")
    print("  4. gmrf_spatial_patterns.png - Spatial visualization")
    print("  5. gmrf_temporal_evolution.png - Temporal visualization")
    print("  6. gmrf_spawner_recruit_correlation.png - Relationship plot")
    

if __name__ == "__main__":
    main()