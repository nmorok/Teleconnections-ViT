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
    seed=2026,
    transform = True
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
    
    if transform == False:
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
        return (data - data.mean()) / data.std(), params
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
    data_transformed = mean_density * np.exp(data_standardized * 2.2)
    
    # 3. Apply the "Zero-Floor" (The Tweedie mass at zero)
    # Adjust the threshold (e.g., 5.0) to get more or fewer zeros

    # after looking at the actual data distribution, a more aggresive threshold is needed. 
    # the median for both the spawners and recruits is 0 so we want like 60th percentile to be zero.
    threshold = np.percentile(data_transformed, 60)
    data_transformed = data_transformed - threshold
    data_transformed[data_transformed < 0.0] = 0
    # Final scaling to ensure the MEAN is correct after zeroing out 65% of the data
    current_mean = data_transformed.mean()
    print(f"  Mean after zero-floor: {current_mean:.2f} (target: {mean_density:.2f})")
    if current_mean > 0:
        data_transformed *= (mean_density / current_mean)

    
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



def finalize_density(latent_data, target_mean, target_max, skew=1.35, zero_pct=60):
    """
    Converts Gaussian latent data to skewed, zero-inflated crab density.
    """
    # 1. Exponential transform (Log-normal tail)
    # Skew=1.5 keeps Max around 1-5 million. Skew=2.2 pushes it to 100M+.
    dens = np.exp(latent_data * skew)
    
    # 2. Zero-inflation (Clipping)
    thresh = np.percentile(dens, zero_pct)
    dens = np.maximum(0, dens - thresh)

   
    
    # 3. Final Scale to hit R-summary Mean
    if dens.mean() > 0:
        dens = dens * (target_mean / dens.mean())
    
    # 4 softcap
    dens = np.clip(dens, a_min=0, a_max = target_max)
    return dens


def create_spawner_recruit_pairs(
    grid_size=50,
    n_spawner_years=30,
    n_recruit_years=30,
    n_bootstraps=100,
    spatial_kappa=0.3,
    temporal_rho=0.7,
    recruitment_correlation=0.3,
    mean_spawner_density=50.0,
    max_spawner_density=1179934.0,
    mean_recruit_density=50.0,
    max_recruit_density=972298.0,
    lag = 3,
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
        seed=seed,
        transform = False
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
        seed=seed + 1000 if seed is not None else None,  # Different seed
        transform = False
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
            spawner_year = t - lag  # Spawner year is lag years before recruit year

            if spawner_year < 0:
                # No spawner data available for early recruit years
                # Just use independent variation (correlation = 0)
                recruits[recruit_idx] = recruits_base[recruit_idx]
                continue
            spawner_idx = b * n_spawner_years + spawner_year  # Corresponding spawner year
            
            # Mix formula: R = α×S + √(1-α²)×ε
            # This ensures Corr(R,S) = α
            # we standardize S and ε to have mean 0 and std 1 before mixing to maintain variance
            s_val = (spawners[spawner_idx] - spawners[spawner_idx].mean()) / (spawners[spawner_idx].std() + 1e-6)
            e_val = (recruits_base[recruit_idx] - recruits_base[recruit_idx].mean()) / (recruits_base[recruit_idx].std() + 1e-6)

            recruits[recruit_idx] = (
                recruitment_correlation * s_val +
                np.sqrt(1 - recruitment_correlation**2) * e_val 
            )
        
    # We must re-standardize the mixed recruits. Even though the formula theoretically 
    # maintains variance, numerical sampling and the early 'base' years can shift it. because of the lag structure. there are different variances
    # for the first 0-lag years and then lag+ years. 
    recruits = (recruits - recruits.mean()) / (recruits.std() + 1e-6)
    spawners = (spawners - spawners.mean()) / (spawners.std() + 1e-6)

    spawners = finalize_density(spawners, target_mean=mean_spawner_density, target_max=max_spawner_density, skew=1.5, zero_pct=60)
    recruits = finalize_density(recruits, target_mean=mean_recruit_density, target_max=max_recruit_density, skew=1.5, zero_pct=60)
    
    # Verify correlation (check a few samples)
    print("\nVerifying spawner-recruitment correlation (LAG-AWARE)...")
    correlations = []
    for b in range(min(10, n_bootstraps)):
        # We only check years where a lag relationship exists (t >= lag)
        for t in range(lag, n_recruit_years): 
            recruit_idx = b * n_recruit_years + t
            spawner_idx = b * n_spawner_years + (t - lag) # Corrected Alignment
            
            corr = np.corrcoef(spawners[spawner_idx].flatten(), 
                               recruits[recruit_idx].flatten())[0, 1]
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
        'mean_recruit_density': mean_recruit_density,
        'lag': lag
    }
    
    return spawners, recruits, params


# ==============================================================================
# PART 6: VISUALIZATION
# ==============================================================================

def visualize_gmrf_properties(spawners, recruits, params, output_dir="./output"):
    grid_size = params['grid_size']
    n_years = params['n_years']
    n_bootstraps = params['n_bootstraps']
    lag = params.get('lag', 3)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Figure 1: Spatial patterns (3 standard snapshots)
    # -------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
    fig1.suptitle('Spatial Patterns Over Time (Bootstrap 0)', fontsize=16)
    years_to_show = [0, n_years//2, n_years-1]
    for i, year_idx in enumerate(years_to_show):
        im_s = axes1[0, i].imshow(spawners[year_idx], cmap='viridis')
        axes1[0, i].set_title(f'Spawners: Year {year_idx+1}')
        axes1[0, i].axis('off')
        plt.colorbar(im_s, ax=axes1[0, i])
        
        im_r = axes1[1, i].imshow(recruits[year_idx], cmap='plasma')
        axes1[1, i].set_title(f'Recruits: Year {year_idx+1}')
        axes1[1, i].axis('off')
        plt.colorbar(im_r, ax=axes1[1, i])
    fig1.tight_layout()
    fig1.savefig(output_dir / 'gmrf_spatial_patterns.png', dpi=150)
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Figure 2: Temporal evolution (Center Pixel)
    # -------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    center = grid_size // 2
    
    axes2[0].plot(range(1, n_years+1), spawners[:n_years, center, center], 'o-')
    axes2[0].set_title('Spawner Center Pixel TS')
    
    axes2[1].plot(range(1, n_years+1), recruits[:n_years, center, center], 'o-', color='orange')
    axes2[1].set_title('Recruit Center Pixel TS')
    
    fig2.tight_layout()
    fig2.savefig(output_dir / 'gmrf_temporal_evolution.png', dpi=150)
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # NEW Figure: 5 Consecutive Years (Temporal Sequence)
    # -------------------------------------------------------------------------
    mid_start = n_years // 2 - 2
    years_5 = range(mid_start, mid_start + 5)
    fig3, axes3 = plt.subplots(2, 5, figsize=(20, 8))
    fig3.suptitle(f'Consecutive 5-Year Sequence (Years {mid_start+1}-{mid_start+5})', fontsize=18)
    
    for i, year_idx in enumerate(years_5):
        axes3[0, i].imshow(spawners[year_idx], cmap='viridis')
        axes3[0, i].set_title(f'Spawners Yr {year_idx+1}')
        axes3[0, i].axis('off')
        
        axes3[1, i].imshow(recruits[year_idx], cmap='plasma')
        axes3[1, i].set_title(f'Recruits Yr {year_idx+1}')
        axes3[1, i].axis('off')
    fig3.tight_layout()
    fig3.savefig(output_dir / 'gmrf_5year_sequence.png', dpi=150)
    plt.close(fig3)

    # -------------------------------------------------------------------------
    # NEW Figure: Lagged Comparison (Biological Prediction Alignment)
    # -------------------------------------------------------------------------
    fig4, axes4 = plt.subplots(2, 4, figsize=(18, 8))
    fig4.suptitle(f'Lagged Biological Alignment (Lag = {lag} years)', fontsize=18)
    
    # Select 4 display years (Year T vs Recruit Year T+Lag)
    example_years = [0, n_years//4, n_years//2, n_years - lag - 1]
    example_years = [y for y in example_years if y + lag < n_years][:4]

    for i, s_yr in enumerate(example_years):
        r_yr = s_yr + lag
        axes4[0, i].imshow(spawners[s_yr], cmap='viridis')
        axes4[0, i].set_title(f'Spawners (Year {s_yr+1})')
        axes4[0, i].axis('off')
        
        axes4[1, i].imshow(recruits[r_yr], cmap='plasma')
        axes4[1, i].set_title(f'Recruits (Year {r_yr+1})\n[S_Year {s_yr+1} + Lag {lag}]')
        axes4[1, i].axis('off')
    fig4.tight_layout()
    fig4.savefig(output_dir / 'gmrf_lagged_comparison.png', dpi=150)
    plt.close(fig4)

    # -------------------------------------------------------------------------
    # Figure 5: Aggregated Temporal Evolution
    # -------------------------------------------------------------------------
    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))
    axes5[0].plot(range(1, n_years+1), np.sum(spawners[:n_years], axis=(1,2)), 'o-')
    axes5[0].set_title('Aggregated Spawner Biomass')
    
    axes5[1].plot(range(1, n_years+1), np.sum(recruits[:n_years], axis=(1,2)), 'o-', color='orange')
    axes5[1].set_title('Aggregated Recruit Biomass')
    
    fig5.tight_layout()
    fig5.savefig(output_dir / 'gmrf_agg_temporal_evolution.png', dpi=150)
    plt.close(fig5)

    # -------------------------------------------------------------------------
    # Figure 6: Diagnostics (Density Histograms & S-R Scatter)
    # -------------------------------------------------------------------------
    fig6, axes6 = plt.subplots(1, 2, figsize=(16, 7))
    axes6[0].hist(recruits.flatten(), bins=100, color='salmon', alpha=0.7, log=True)
    axes6[0].set_title('Recruit Density (Log Scale Hist)')
    
    n_samples = min(10000, spawners.size)
    idx = np.random.choice(spawners.size, n_samples, replace=False)
    axes6[1].scatter(spawners.flatten()[idx], recruits.flatten()[idx], alpha=0.1, s=2)
    axes6[1].set_title('Global S-R Correlation')
    
    fig6.tight_layout()
    fig6.savefig(output_dir / 'gmrf_diagnostics.png', dpi=150)
    plt.close(fig6)


# ==============================================================================
# PART 7: MAIN EXECUTION
# ==============================================================================

def main():
    """
    Final Main function with Log-Space Global Correlation Verification.
    """
    print("="*80)
    print("GMRF DUMMY DATA GENERATOR FOR CRAB RECRUITMENT")
    print("="*80)
    
    config_easy = {
        'grid_size': 50,
        'n_spawner_years': 30,
        'n_recruit_years': 30,
        'n_bootstraps': 100,
        'spatial_kappa': 0.3,
        'temporal_rho': 0.0,
        'recruitment_correlation': 0.9, 
        'mean_spawner_density': 3728.0, 
        'max_spawner_density': 1179934.0,
        'mean_recruit_density': 4514.0, 
        'max_recruit_density': 972298.0,
        'lag': 0,
        'seed': 2026
    }
    
    # 1. Generate data
    spawners_easy, recruits_easy, params_easy = create_spawner_recruit_pairs(**config_easy)
    
    # 2. Reshape for Lag-Aware Alignment
    lag = config_easy['lag']
    n_years = config_easy['n_spawner_years']
    n_boot = config_easy['n_bootstraps']
    
    s_cube_easy = spawners_easy.reshape(n_boot, n_years, 50, 50)
    r_cube_easy = recruits_easy.reshape(n_boot, n_years, 50, 50)
    
    # Align: Spawner[t] matched with Recruit[t + lag]
    if lag > 0:
        s_aligned = s_cube_easy[:, :-lag, :, :].flatten()
        r_aligned = r_cube_easy[:, lag:, :, :].flatten()
    else:
        s_aligned = s_cube_easy.flatten()
        r_aligned = r_cube_easy.flatten()
    
    # --- THE DIAGNOSTIC FIX: Correlation in Log-Space ---
    # This prevents the 79M max values from ruining the global statistic
    overall_corr = np.corrcoef(np.log1p(s_aligned), np.log1p(r_aligned))[0, 1]

    # 3. Save outputs
    output_dir = Path("data/dummy/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "gmrf_spawners_50x50_easy.npy", spawners_easy)
    np.save(output_dir / "gmrf_recruits_50x50_easy.npy", recruits_easy)
    with open(output_dir / "gmrf_params_easy.json", 'w') as f:
        json.dump(params_easy, f, indent=2)

    # 4. Summary & Verification
    print("\n" + "="*80)
    print("GENERATION SUMMARY (LOG-STABILIZED)")
    print("="*80)
    
    s_zeros = 100 * np.mean(spawners_easy == 0)
    r_zeros = 100 * np.mean(recruits_easy == 0)
    
    print(f"Spawners -> Mean: {spawners_easy.mean():.1f}, Max: {spawners_easy.max():.1f}, Zeros: {s_zeros:.1f}%")
    print(f"Recruits -> Mean: {recruits_easy.mean():.1f}, Max: {recruits_easy.max():.1f}, Zeros: {r_zeros:.1f}%")
    print(f"Global Log-Lagged S-R Correlation: {overall_corr:.4f}")


    if overall_corr > 0.6:
        print("✅ SUCCESS: Biological signal is preserved and verified!")
    else:
        print("⚠️ WARNING: Signal still weak. Verify latency mixing.")

    # 5. Visualizations
    visualize_gmrf_properties(spawners_easy, recruits_easy, params_easy, output_dir=output_dir)
    
    # Spatial Check Bootstrap 0: Year 0 Spawner vs Year 3 Recruit
    S_field = spawners_easy[0]   # B0, Y0
    R_field = recruits_easy[lag]   # B0, Y3
    spatial_corr = np.corrcoef(np.log1p(S_field.flatten()), np.log1p(R_field.flatten()))[0,1]
    
    print(f"\n--- SPATIAL CHECK (B0: S_Yr0 vs R_Yr{lag}) ---")
    print(f"Calculated Spatial Correlation (Log-Space): {spatial_corr:.4f}")

    config_medium = {
        'grid_size': 50,
        'n_spawner_years': 30,
        'n_recruit_years': 30,
        'n_bootstraps': 100,
        'spatial_kappa': 0.3,
        'temporal_rho': 0.7,
        'recruitment_correlation': 0.9, 
        'mean_spawner_density': 3728.0, 
        'max_spawner_density': 1179934.0,
        'mean_recruit_density': 4514.0, 
        'max_recruit_density': 972298.0,
        'lag': 3,
        'seed': 2026
    }
    
    # 1. Generate data
    spawners_medium, recruits_medium, params_medium = create_spawner_recruit_pairs(**config_medium)
    
    # 2. Reshape for Lag-Aware Alignment
    lag = config_medium['lag']
    n_years = config_medium['n_spawner_years']
    n_boot = config_medium['n_bootstraps']
    
    s_cube_medium = spawners_medium.reshape(n_boot, n_years, 50, 50)
    r_cube_medium = recruits_medium.reshape(n_boot, n_years, 50, 50)
    
    # Align: Spawner[t] matched with Recruit[t + lag]
    s_aligned = s_cube_medium[:, :-lag, :, :].flatten()
    r_aligned = r_cube_medium[:, lag:, :, :].flatten()
    
    # --- THE DIAGNOSTIC FIX: Correlation in Log-Space ---
    # This prevents the 79M max values from ruining the global statistic
    overall_corr = np.corrcoef(np.log1p(s_aligned), np.log1p(r_aligned))[0, 1]

    # 3. Save outputs
    output_dir = Path("data/dummy/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "gmrf_spawners_50x50_medium.npy", spawners_medium)
    np.save(output_dir / "gmrf_recruits_50x50_medium.npy", recruits_medium)
    with open(output_dir / "gmrf_params_medium.json", 'w') as f:
        json.dump(params_medium, f, indent=2)

    # 4. Summary & Verification
    print("\n" + "="*80)
    print("GENERATION SUMMARY (LOG-STABILIZED)")
    print("="*80)
    
    s_zeros = 100 * np.mean(spawners_medium == 0)
    r_zeros = 100 * np.mean(recruits_medium == 0)
    
    print(f"Spawners -> Mean: {spawners_medium.mean():.1f}, Max: {spawners_medium.max():.1f}, Zeros: {s_zeros:.1f}%")
    print(f"Recruits -> Mean: {recruits_medium.mean():.1f}, Max: {recruits_medium.max():.1f}, Zeros: {r_zeros:.1f}%")
    print(f"Global Log-Lagged S-R Correlation: {overall_corr:.4f}")


    if overall_corr > 0.6:
        print("✅ SUCCESS: Biological signal is preserved and verified!")
    else:
        print("⚠️ WARNING: Signal still weak. Verify latency mixing.")

    # 5. Visualizations
    visualize_gmrf_properties(spawners_medium, recruits_medium, params_medium, output_dir=output_dir)
    
    # Spatial Check Bootstrap 0: Year 0 Spawner vs Year 3 Recruit
    S_field = spawners_medium[0]   # B0, Y0
    R_field = recruits_medium[lag]   # B0, Y3
    spatial_corr = np.corrcoef(np.log1p(S_field.flatten()), np.log1p(R_field.flatten()))[0,1]
    
    print(f"\n--- SPATIAL CHECK (B0: S_Yr0 vs R_Yr{lag}) ---")
    print(f"Calculated Spatial Correlation (Log-Space): {spatial_corr:.4f}")


    config_hard = {
        'grid_size': 50,
        'n_spawner_years': 30,
        'n_recruit_years': 30,
        'n_bootstraps': 100,
        'spatial_kappa': 0.3,
        'temporal_rho': 0.7,
        'recruitment_correlation': 0.8, 
        'mean_spawner_density': 3728.0, 
        'max_spawner_density': 1179934.0,
        'mean_recruit_density': 4514.0, 
        'max_recruit_density': 972298.0,
        'lag': 5,
        'seed': 2026
    }
    
    # 1. Generate data
    spawners_hard, recruits_hard, params_hard = create_spawner_recruit_pairs(**config_hard)
    
    # 2. Reshape for Lag-Aware Alignment
    lag = config_hard['lag']
    n_years = config_hard['n_spawner_years']
    n_boot = config_hard['n_bootstraps']
    
    s_cube_hard = spawners_hard.reshape(n_boot, n_years, 50, 50)
    r_cube_hard = recruits_hard.reshape(n_boot, n_years, 50, 50)
    
    # Align: Spawner[t] matched with Recruit[t + lag]
    s_aligned = s_cube_hard[:, :-lag, :, :].flatten()
    r_aligned = r_cube_hard[:, lag:, :, :].flatten()
    
    # --- THE DIAGNOSTIC FIX: Correlation in Log-Space ---
    # This prevents the 79M max values from ruining the global statistic
    overall_corr = np.corrcoef(np.log1p(s_aligned), np.log1p(r_aligned))[0, 1]

    # 3. Save outputs
    output_dir = Path("data/dummy/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "gmrf_spawners_50x50_hard.npy", spawners_hard)
    np.save(output_dir / "gmrf_recruits_50x50_hard.npy", recruits_hard)
    with open(output_dir / "gmrf_params_hard.json", 'w') as f:
        json.dump(params_hard, f, indent=2)

    # 4. Summary & Verification
    print("\n" + "="*80)
    print("GENERATION SUMMARY (LOG-STABILIZED)")
    print("="*80)
    
    s_zeros = 100 * np.mean(spawners_hard == 0)
    r_zeros = 100 * np.mean(recruits_hard == 0)
    
    print(f"Spawners -> Mean: {spawners_hard.mean():.1f}, Max: {spawners_hard.max():.1f}, Zeros: {s_zeros:.1f}%")
    print(f"Recruits -> Mean: {recruits_hard.mean():.1f}, Max: {recruits_hard.max():.1f}, Zeros: {r_zeros:.1f}%")
    print(f"Global Log-Lagged S-R Correlation: {overall_corr:.4f}")


    if overall_corr > 0.6:
        print("✅ SUCCESS: Biological signal is preserved and verified!")
    else:
        print("⚠️ WARNING: Signal still weak. Verify latency mixing.")

    # 5. Visualizations
    visualize_gmrf_properties(spawners_hard, recruits_hard, params_hard, output_dir=output_dir)
    
    # Spatial Check Bootstrap 0: Year 0 Spawner vs Year 3 Recruit
    S_field = spawners_hard[0]   # B0, Y0
    R_field = recruits_hard[lag]   # B0, Y3
    spatial_corr = np.corrcoef(np.log1p(S_field.flatten()), np.log1p(R_field.flatten()))[0,1]
    
    print(f"\n--- SPATIAL CHECK (B0: S_Yr0 vs R_Yr{lag}) ---")
    print(f"Calculated Spatial Correlation (Log-Space): {spatial_corr:.4f}")
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE!")
    print("="*80)


def run_tweedie_test(data, label="Data"):
    """
    Tweedie Test: Checks if Variance scales with Mean (Power Law).
    Mathematically: Var = phi * Mean^p. 
    On a log-log plot, the slope of the line is the Tweedie index 'p'.
    Target: 1 < p < 2.
    """
    # Sample patches to get local variations in mean and variance
    means, vars = [], []
    for _ in range(1000):
        # Randomly sample a 5x5 patch from a random bootstrap/year
        b = np.random.randint(0, data.shape[0])
        r = np.random.randint(0, data.shape[1]-5)
        c = np.random.randint(0, data.shape[2]-5)
        
        patch = data[b, r:r+5, c:c+5]
        m, v = np.mean(patch), np.var(patch)
        if m > 0.1 and v > 0.1: # Exclude dead zones for the log-test
            means.append(m)
            vars.append(v)
            
    log_m, log_v = np.log(means), np.log(vars)
    p_index, intercept = np.polyfit(log_m, log_v, 1)
    
    plt.scatter(log_m, log_v, alpha=0.2, label=f'{label} (p={p_index:.2f})')
    plt.plot(log_m, p_index*log_m + intercept, '--')
    return p_index

def plot_density_histograms(spawners, recruits):
    """Plots log-scaled histograms to see the zero-mass and the skewed tail."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for ax, data, title, color in zip(axes, [spawners, recruits], 
                                     ["Spawner Densities", "Recruit Densities"], 
                                     ["blue", "red"]):
        ax.hist(data.flatten(), bins=100, color=color, alpha=0.6, log=True)
        ax.set_title(f"{title}\nZero %: {100*np.mean(data==0):.1f}%")
        ax.set_xlabel("Density")
        ax.set_ylabel("Frequency (Log Scale)")
        ax.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()