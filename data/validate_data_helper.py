import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.data_helper import get_dataloaders

import torch
import numpy as np

def validate_temporal_continuity(train_loader, val_loader, test_loader, 
                                train_years=22, val_years=5, test_years=3,
                                memory_years=5):
    """
    Comprehensive validation for temporal dataset with cross-split historical context.
    """
    print("="*80)
    print("COMPREHENSIVE TEMPORAL CONTINUITY VALIDATION")
    print("="*80)
    
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    test_ds = test_loader.dataset
    
    all_tests_passed = True
    
    # ============================================================================
    # TEST 1: Basic Dataset Properties
    # ============================================================================
    print("\n" + "="*80)
    print("TEST 1: Basic Dataset Properties")
    print("="*80)
    
    print(f"\nTraining Dataset:")
    print(f"  Total samples: {len(train_ds)}")
    print(f"  Bootstrap samples: {train_ds.n_bootstraps}")
    print(f"  Years per bootstrap: {train_ds.n_years}")
    print(f"  Expected: {train_ds.n_bootstraps * train_years}")
    assert len(train_ds) == train_ds.n_bootstraps * train_years
    print(f"  ✓ Correct sample count")
    
    print(f"\nValidation Dataset:")
    print(f"  Total samples: {len(val_ds)}")
    print(f"  Bootstrap samples: {val_ds.n_bootstraps}")
    print(f"  Years per bootstrap: {val_ds.n_years}")
    print(f"  Has historical data: {val_ds.historical_spawners is not None}")
    assert len(val_ds) == val_ds.n_bootstraps * val_years
    print(f"  ✓ Correct sample count")
    
    print(f"\nTest Dataset:")
    print(f"  Total samples: {len(test_ds)}")
    print(f"  Bootstrap samples: {test_ds.n_bootstraps}")
    print(f"  Years per bootstrap: {test_ds.n_years}")
    print(f"  Has historical data: {test_ds.historical_spawners is not None}")
    assert len(test_ds) == test_ds.n_bootstraps * test_years
    print(f"  ✓ Correct sample count")
    
    # Verify same number of bootstraps across all splits
    assert train_ds.n_bootstraps == val_ds.n_bootstraps == test_ds.n_bootstraps
    print(f"\n✓ All datasets have same number of bootstraps ({train_ds.n_bootstraps})")
    
    # Verify scaling consistency
    assert train_ds.spawner_max == val_ds.spawner_max == test_ds.spawner_max
    assert train_ds.recruit_max == val_ds.recruit_max == test_ds.recruit_max
    print(f"✓ Scaling factors consistent across splits")
    print(f"  Spawner max: {train_ds.spawner_max:.4f}")
    print(f"  Recruit max: {train_ds.recruit_max:.4f}")
    
    # ============================================================================
    # TEST 2: Temporal Masks - Validation and Test should have FULL history
    # ============================================================================
    print("\n" + "="*80)
    print("TEST 2: Temporal Mask Validation")
    print("="*80)
    
    print("\nTraining (years 0-21) - masks vary by year:")
    test_years_train = [0, 1, 2, 4, 5, 10, 21]
    expected_masks_train = [
        [0, 0, 0, 0, 0],  # Year 0
        [1, 0, 0, 0, 0],  # Year 1
        [1, 1, 0, 0, 0],  # Year 2
        [1, 1, 1, 1, 0],  # Year 4
        [1, 1, 1, 1, 1],  # Year 5
        [1, 1, 1, 1, 1],  # Year 10
        [1, 1, 1, 1, 1],  # Year 21
    ]
    
    train_mask_passed = True
    for year, expected in zip(test_years_train, expected_masks_train):
        if year < train_years:
            _, _, mask, _ = train_ds[year]  # Bootstrap 0
            actual = mask.numpy().tolist()
            match = actual == expected
            status = "✓" if match else "✗"
            print(f"  {status} Year {year:2d}: expected {expected}, got {actual}")
            if not match:
                train_mask_passed = False
    
    if train_mask_passed:
        print("  ✓ Training temporal masks correct")
    else:
        print("  ✗ Training temporal masks FAILED")
        all_tests_passed = False
    
    print("\nValidation (years 22-26) - ALL should have FULL history [1,1,1,1,1]:")
    val_mask_passed = True
    expected_full = [1, 1, 1, 1, 1]
    
    for year_idx in range(val_years):
        for bootstrap_idx in [0, val_ds.n_bootstraps // 2, val_ds.n_bootstraps - 1]:
            sample_idx = bootstrap_idx * val_years + year_idx
            _, _, mask, _ = val_ds[sample_idx]
            actual = mask.numpy().tolist()
            match = actual == expected_full
            
            if year_idx == 0 or year_idx == val_years - 1:  # Print first and last year
                status = "✓" if match else "✗"
                print(f"  {status} Bootstrap {bootstrap_idx}, Year {year_idx}: {actual}")
            
            if not match:
                print(f"  ✗ FAILED: Bootstrap {bootstrap_idx}, Year {year_idx}: expected {expected_full}, got {actual}")
                val_mask_passed = False
    
    if val_mask_passed:
        print("  ✓ ALL validation samples have full 5-year history")
    else:
        print("  ✗ Validation temporal masks FAILED")
        all_tests_passed = False
    
    print("\nTest (years 27-29) - ALL should have FULL history [1,1,1,1,1]:")
    test_mask_passed = True
    
    for year_idx in range(test_years):
        for bootstrap_idx in [0, test_ds.n_bootstraps // 2, test_ds.n_bootstraps - 1]:
            sample_idx = bootstrap_idx * test_years + year_idx
            _, _, mask, _ = test_ds[sample_idx]
            actual = mask.numpy().tolist()
            match = actual == expected_full
            
            status = "✓" if match else "✗"
            print(f"  {status} Bootstrap {bootstrap_idx}, Year {year_idx}: {actual}")
            
            if not match:
                test_mask_passed = False
    
    if test_mask_passed:
        print("  ✓ ALL test samples have full 5-year history")
    else:
        print("  ✗ Test temporal masks FAILED")
        all_tests_passed = False
    
    # ============================================================================
    # TEST 3: Historical Data Continuity Across Splits
    # ============================================================================
    print("\n" + "="*80)
    print("TEST 3: Historical Data Continuity Across Splits")
    print("="*80)
    
    print("\nTesting that validation year 22 has access to training years 17-21...")
    
    continuity_passed = True
    bootstrap_to_test = 0  # Test first bootstrap
    
    # Get validation year 0 (which is year 22 globally)
    val_idx = bootstrap_to_test * val_years + 0
    val_input, _, val_mask, _ = val_ds[val_idx]
    
    # val_input channels:
    # 0: current spawner (year 22)
    # 1-5: historical spawners (years 21, 20, 19, 18, 17)
    # 6-10: historical recruits (years 21, 20, 19, 18, 17)
    
    # Get training years 17-21 for same bootstrap
    train_years_to_check = [17, 18, 19, 20, 21]
    
    print(f"\nChecking Bootstrap {bootstrap_to_test}:")
    for i, train_year in enumerate(train_years_to_check):
        # Get training data
        train_idx = bootstrap_to_test * train_years + train_year
        train_input, train_target, _, _ = train_ds[train_idx]
        train_spawner = train_input[0]  # Current spawner
        train_recruit = train_target[0]  # Current recruit
        
        # Get historical data from validation
        # Memory is stored as [t-1, t-2, t-3, t-4, t-5]
        # So for year 22, we want years [21, 20, 19, 18, 17]
        # train_year 21 should be in position 0 (t-1)
        # train_year 20 should be in position 1 (t-2), etc.
        mem_position = 21 - train_year
        val_hist_spawner = val_input[1 + mem_position]  # Channels 1-5
        val_hist_recruit = val_input[6 + mem_position]  # Channels 6-10
        
        # Compare
        spawner_match = torch.allclose(train_spawner, val_hist_spawner, atol=1e-6)
        recruit_match = torch.allclose(train_recruit, val_hist_recruit, atol=1e-6)
        
        spawner_status = "✓" if spawner_match else "✗"
        recruit_status = "✓" if recruit_match else "✗"
        
        print(f"  Year {train_year}: Spawner {spawner_status}, Recruit {recruit_status}")
        
        if not spawner_match:
            print(f"    ✗ Spawner mismatch! Max diff: {(train_spawner - val_hist_spawner).abs().max():.6f}")
            continuity_passed = False
        if not recruit_match:
            print(f"    ✗ Recruit mismatch! Max diff: {(train_recruit - val_hist_recruit).abs().max():.6f}")
            continuity_passed = False
    
    if continuity_passed:
        print("  ✓ Validation has correct historical data from training")
    else:
        print("  ✗ Historical data continuity FAILED")
        all_tests_passed = False
    
    # Similar test for test dataset
    print("\nTesting that test year 27 has access to validation years 22-26...")
    
    test_continuity_passed = True
    
    # Get test year 0 (which is year 27 globally)
    test_idx = bootstrap_to_test * test_years + 0
    test_input, _, test_mask, _ = test_ds[test_idx]
    
    # Get validation years 22-26 for same bootstrap
    val_years_to_check = list(range(val_years))
    
    print(f"\nChecking Bootstrap {bootstrap_to_test}:")
    for i, val_year in enumerate(val_years_to_check):
        # Get validation data
        val_idx = bootstrap_to_test * val_years + val_year
        val_input, val_target, _, _ = val_ds[val_idx]
        val_spawner = val_input[0]
        val_recruit = val_target[0]
        
        # Memory position for year 27 looking back at validation
        # [26, 25, 24, 23, 22] = [t-1, t-2, t-3, t-4, t-5]
        mem_position = (val_years - 1) - val_year
        test_hist_spawner = test_input[1 + mem_position]
        test_hist_recruit = test_input[6 + mem_position]
        
        spawner_match = torch.allclose(val_spawner, test_hist_spawner, atol=1e-6)
        recruit_match = torch.allclose(val_recruit, test_hist_recruit, atol=1e-6)
        
        spawner_status = "✓" if spawner_match else "✗"
        recruit_status = "✓" if recruit_match else "✗"
        
        print(f"  Year {22 + val_year}: Spawner {spawner_status}, Recruit {recruit_status}")
        
        if not spawner_match:
            print(f"    ✗ Spawner mismatch! Max diff: {(val_spawner - test_hist_spawner).abs().max():.6f}")
            test_continuity_passed = False
        if not recruit_match:
            print(f"    ✗ Recruit mismatch! Max diff: {(val_recruit - test_hist_recruit).abs().max():.6f}")
            test_continuity_passed = False
    
    if test_continuity_passed:
        print("  ✓ Test has correct historical data from validation")
    else:
        print("  ✗ Test historical data continuity FAILED")
        all_tests_passed = False
    
    # ============================================================================
    # TEST 4: Bootstrap Boundaries Still Respected
    # ============================================================================
    print("\n" + "="*80)
    print("TEST 4: Bootstrap Boundaries")
    print("="*80)
    
    print("\nVerifying bootstraps don't share data across boundaries...")
    
    boundary_passed = True
    
    # Check training: last year of bootstrap 0 vs first year of bootstrap 1
    if train_ds.n_bootstraps > 1:
        last_of_b0 = train_years - 1
        first_of_b1 = train_years
        
        input_b0, _, mask_b0, _ = train_ds[last_of_b0]
        input_b1, _, mask_b1, _ = train_ds[first_of_b1]
        
        # Bootstrap 1 year 0 should have no history
        expected_mask = [0, 0, 0, 0, 0]
        actual_mask = mask_b1.numpy().tolist()
        
        if actual_mask == expected_mask:
            print(f"  ✓ Training: Bootstrap 1 year 0 has no history (correct)")
        else:
            print(f"  ✗ Training: Bootstrap 1 year 0 has history (WRONG): {actual_mask}")
            boundary_passed = False
    
    # Check validation: bootstraps should not share data
    if val_ds.n_bootstraps > 1:
        # But they should ALL have full history from training
        first_of_b0_val = 0
        first_of_b1_val = val_years
        
        _, _, mask_b0_val, _ = val_ds[first_of_b0_val]
        _, _, mask_b1_val, _ = val_ds[first_of_b1_val]
        
        # Both should have full history from training
        expected_full = [1, 1, 1, 1, 1]
        
        if mask_b0_val.numpy().tolist() == expected_full and mask_b1_val.numpy().tolist() == expected_full:
            print(f"  ✓ Validation: Both bootstrap 0 and 1 have full history from training")
        else:
            print(f"  ✗ Validation: Mask mismatch")
            print(f"    Bootstrap 0: {mask_b0_val.numpy().tolist()}")
            print(f"    Bootstrap 1: {mask_b1_val.numpy().tolist()}")
            boundary_passed = False
    
    if boundary_passed:
        print("  ✓ Bootstrap boundaries respected")
    else:
        print("  ✗ Bootstrap boundary test FAILED")
        all_tests_passed = False
    
    # ============================================================================
    # TEST 5: Data Scaling
    # ============================================================================
    print("\n" + "="*80)
    print("TEST 5: Data Scaling")
    print("="*80)
    
    scaling_passed = True
    
    # Check a batch from each loader
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))
    
    for name, batch in [("Training", train_batch), ("Validation", val_batch), ("Test", test_batch)]:
        inputs, targets, _, _ = batch
        
        print(f"\n{name}:")
        print(f"  Input range: [{inputs.min():.4f}, {inputs.max():.4f}]")
        print(f"  Target range: [{targets.min():.4f}, {targets.max():.4f}]")
        
        if inputs.min() >= 0 and inputs.max() <= 1:
            print(f"  ✓ Inputs in [0, 1]")
        else:
            print(f"  ✗ Inputs out of range!")
            scaling_passed = False
        
        if targets.min() >= 0 and targets.max() <= 1:
            print(f"  ✓ Targets in [0, 1]")
        else:
            print(f"  ✗ Targets out of range!")
            scaling_passed = False
    
    if not scaling_passed:
        all_tests_passed = False
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if all_tests_passed:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nYour temporal dataset is correctly configured:")
        print("  ✓ Training (years 0-21) uses its own history")
        print("  ✓ Validation (years 22-26) has full 5-year history from training")
        print("  ✓ Test (years 27-29) has full 5-year history from validation")
        print("  ✓ Bootstrap boundaries respected")
        print("  ✓ Data scaling consistent across all splits")
    else:
        print("\n❌ SOME TESTS FAILED ❌")
        print("\nPlease review the failures above and fix the issues.")
    
    print("="*80)
    
    return all_tests_passed


# Example usage:
if __name__ == "__main__":
    from data.data_helper import get_dataloaders
    
    print("Loading dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=5,
        memory_years=5,
        train_years=22,
        val_years=5,
        test_years=3
    )
    
    print("\nStarting validation...")
    success = validate_temporal_continuity(
        train_loader, val_loader, test_loader,
        train_years=22, val_years=5, test_years=3,
        memory_years=5
    )
    
    if success:
        print("\n✅ Dataset ready for training!")
    else:
        print("\n❌ Please fix issues before training!")