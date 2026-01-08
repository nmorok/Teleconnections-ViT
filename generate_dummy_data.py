"""
Generate dummy crab data for pipeline testing
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path

def create_dummy_crab_data(n_years = 23, n_bootstraps = 100, grid_size = 10, spatial_smoothing = True):

    """Generate dummy spawner and recruitment data."""

    np.random.seed(42)

    
def create_smooth_random_field(grid_size = 10, smoothness = 2):
    """Generate spatially autocorrelated fields that mimic SPDE matern covariance functions."""