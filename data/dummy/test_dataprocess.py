"""
Debug script to find where variation is lost in your existing GMRF code.

Run this step-by-step to identify the problem.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Copy your functions here
def create_spatial_precision_matrix(grid_size=50, kappa=1.0):
    """Your existing function"""
    n_cells = grid_size * grid_size
    Q = sparse.lil_matrix((n_cells, n_cells))
    
    def grid_to_index(row, col):
        return row * grid_size + col
    
    for row in range(grid_size):
        for col in range(grid_size):
            center_idx = grid_to_index(row, col)
            n_neighbors = 0
            
            if row > 0:
                neighbor_idx = grid_to_index(row - 1, col)
                Q[center_idx, neighbor_idx] = -kappa**2
                n_neighbors += 1
            
            if row < grid_size - 1:
                neighbor_idx = grid_to_index(row + 1, col)
                Q[center_idx, neighbor_idx] = -kappa**2
                n_neighbors += 1
            
            if col > 0:
                neighbor_idx = grid_to_index(row, col - 1)
                Q[center_idx, neighbor_idx] = -kappa**2
                n_neighbors += 1
            
            if col < grid_size - 1:
                neighbor_idx = grid_to_index(row, col + 1)
                Q[center_idx, neighbor_idx] = -kappa**2
                n_neighbors += 1
            
            Q[center_idx, center_idx] = n_neighbors * kappa**2
    
    return Q.tocsr()


def sample_gmrf(Q, n_samples=1):
    """Your existing function"""
    n = Q.shape[0]
    samples = []
    
    for _ in range(n_samples):
        z = np.random.randn(n)
        x = spsolve(Q, z)      
        x = (x - x.mean()) / x.std()
        samples.append(x)
    
    return np.array(samples)


# =============================================================================
# DEBUGGING STEPS
# =============================================================================

def test_step1_precision_matrix():
    """Test 1: Does precision matrix have correct structure?"""
    print("="*70)
    print("TEST 1: Precision Matrix Structure")
    print("="*70)
    
    grid_size = 10  # Small for testing
    kappa = 0.3
    
    Q = create_spatial_precision_matrix(grid_size, kappa=kappa)
    
    print(f"Matrix shape: {Q.shape}")
    print(f"Expected: ({grid_size**2}, {grid_size**2})")
    print(f"Non-zero elements: {Q.nnz}")
    print(f"Expected: ~{grid_size**2 * 5} (center + 4 neighbors per cell)")
    
    # Check a few diagonal elements
    print(f"\nDiagonal elements (should be 2-4 × kappa²):")
    for i in [0, 50, 99]:  # Corner, middle, corner
        print(f"  Q[{i},{i}] = {Q[i,i]:.4f} (kappa²={kappa**2:.4f})")
    
    # Check matrix is symmetric
    is_symmetric = np.allclose(Q.toarray(), Q.toarray().T)
    print(f"\nIs symmetric? {is_symmetric}")
    
    if Q.nnz > 0 and is_symmetric:
        print("\n✓ Precision matrix looks correct!")
        return True
    else:
        print("\n❌ Problem with precision matrix!")
        return False


def test_step2_single_sample():
    """Test 2: Does a single GMRF sample have variation?"""
    print("\n" + "="*70)
    print("TEST 2: Single GMRF Sample")
    print("="*70)
    
    grid_size = 50
    kappa = 0.3  # Try different values!
    
    print(f"Parameters: grid_size={grid_size}, kappa={kappa}")
    
    Q = create_spatial_precision_matrix(grid_size, kappa=kappa)
    sample = sample_gmrf(Q, n_samples=1)[0]
    
    # Reshape to 2D
    field = sample.reshape(grid_size, grid_size)
    
    print(f"\nRaw GMRF sample stats:")
    print(f"  Min: {field.min():.6f}")
    print(f"  Max: {field.max():.6f}")
    print(f"  Range: {field.max() - field.min():.6f}")
    print(f"  Mean: {field.mean():.6f}")
    print(f"  Std: {field.std():.6f}")
    print(f"  CV: {field.std() / abs(field.mean()):.6f}")
    
    if field.std() > 0.1:
        print("\n✓ Single sample has good variation!")
        return True, field
    else:
        print("\n❌ Single sample has NO variation!")
        print("   Problem: kappa might be too large")
        return False, field


def test_step3_transformation():
    """Test 3: Does transformation preserve variation?"""
    print("\n" + "="*70)
    print("TEST 3: Transformation Pipeline")
    print("="*70)
    
    # Generate a sample with known variation
    grid_size = 50
    Q = create_spatial_precision_matrix(grid_size, kappa=0.3)
    sample = sample_gmrf(Q, n_samples=1)[0]
    field = sample.reshape(grid_size, grid_size)
    
    print("BEFORE transformation:")
    print(f"  Std: {field.std():.6f}")
    print(f"  Range: {field.max() - field.min():.6f}")
    
    # Apply YOUR transformation
    mean_density = 50.0
    field_standardized = (field - field.mean()) / field.std()
    field_transformed = mean_density * np.exp(field_standardized * 0.5)
    
    print("\nAFTER transformation:")
    print(f"  Std: {field_transformed.std():.6f}")
    print(f"  Range: {field_transformed.max() - field_transformed.min():.6f}")
    print(f"  CV: {field_transformed.std() / field_transformed.mean():.6f}")
    
    if field_transformed.std() > 5.0:
        print("\n✓ Transformation preserves variation!")
        return True
    else:
        print("\n❌ Transformation KILLS variation!")
        return False


def test_step4_different_kappas():
    """Test 4: How does kappa affect variation?"""
    print("\n" + "="*70)
    print("TEST 4: Effect of Kappa Parameter")
    print("="*70)
    
    grid_size = 50
    kappas = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    
    print(f"{'Kappa':<10} {'Std':<12} {'Range':<12} {'CV':<12} {'Status'}")
    print("-" * 60)
    
    for kappa in kappas:
        Q = create_spatial_precision_matrix(grid_size, kappa=kappa)
        sample = sample_gmrf(Q, n_samples=1)[0]
        field = sample.reshape(grid_size, grid_size)
        
        std = field.std()
        range_val = field.max() - field.min()
        cv = std / abs(field.mean()) if field.mean() != 0 else 0
        
        status = "✓ Good" if std > 0.5 else "❌ Too smooth"
        
        print(f"{kappa:<10.2f} {std:<12.4f} {range_val:<12.4f} {cv:<12.4f} {status}")
    
    print("\nRecommendation: Use kappa = 0.1-0.5 for good spatial variation")


def test_step5_visualize():
    """Test 5: Visualize what the fields look like"""
    print("\n" + "="*70)
    print("TEST 5: Visual Check")
    print("="*70)
    
    import matplotlib.pyplot as plt
    
    grid_size = 50
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    kappas = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    
    for idx, kappa in enumerate(kappas):
        ax = axes[idx // 3, idx % 3]
        
        Q = create_spatial_precision_matrix(grid_size, kappa=kappa)
        sample = sample_gmrf(Q, n_samples=1)[0]
        field = sample.reshape(grid_size, grid_size)
        
        im = ax.imshow(field, cmap='viridis')
        ax.set_title(f'kappa={kappa:.1f}\nStd={field.std():.3f}')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('gmrf_kappa_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to: gmrf_kappa_comparison.png")
    print("  Check if you see spatial patterns (not uniform color!)")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🔍 " * 35)
    print("DEBUGGING YOUR EXISTING GMRF CODE")
    print("🔍 " * 35 + "\n")
    
    # Run tests in sequence
    test1_pass = test_step1_precision_matrix()
    
    if test1_pass:
        test2_pass, field = test_step2_single_sample()
        
        if test2_pass:
            test3_pass = test_step3_transformation()
            test_step4_different_kappas()
            test_step5_visualize()
        else:
            print("\n⚠️  DIAGNOSIS: Problem in GMRF sampling")
            print("   Try smaller kappa values (0.1-0.3)")
            test_step4_different_kappas()
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
    print("\nIf all tests pass, your code is fine!")
    print("If a test fails, that's where the problem is.")