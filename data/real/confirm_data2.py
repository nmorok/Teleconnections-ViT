"""
=============================================================================
Real Data Diagnostic: EBS Snow Crab Survey Data Verification
=============================================================================
Run this BEFORE training to verify data integrity, ordering, and gap handling.

Checks:
  1. Raw array shapes and dtypes
  2. Calendar year mapping and 2020 gap
  3. NaN/Inf/negative values
  4. Spatial mask alignment
  5. Per-year statistics (mean, max, nonzero fraction)
  6. Flattened ordering verification (B0Y0, B0Y1, ...)
  7. Bootstrap consistency (should differ but be spatially correlated)
  8. Split file verification (if splits exist)
  9. Memory bank / temporal mask verification for 2020 gap
  10. Time series visualization

Usage:
    python diagnose_real_data.py
"""

import numpy as np
import os
import json

# =========================================================================
# CONFIG
# =========================================================================

DATA_DIR = "data/real/output"
SPLITS_DIR = "data/real/splits/real"
CALENDAR_YEARS = list(range(1988, 2024)) # 35 years

TRAIN_YEARS = 22
VAL_YEARS = 9
TEST_YEARS = 4

# =========================================================================
# HELPERS
# =========================================================================

def section(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def ok(msg):
    print(f"  ✓ {msg}")


def warn(msg):
    print(f"  ⚠️  {msg}")


def fail(msg):
    print(f"  ✗ {msg}")


# =========================================================================
# 1. RAW DATA SHAPES
# =========================================================================

def check_raw_shapes():
    section("1. RAW DATA SHAPES")

    for name in ["gridded_spawners.npy", "gridded_recruits.npy", "spatial_mask.npy"]:
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            fail(f"Missing: {path}")
            continue
        arr = np.load(path)
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, "
              f"min={arr.min():.4f}, max={arr.max():.4f}")

    spawners = np.load(os.path.join(DATA_DIR, "gridded_spawners.npy"))
    recruits = np.load(os.path.join(DATA_DIR, "gridded_recruits.npy"))

    if spawners.shape != recruits.shape:
        fail(f"Shape mismatch: spawners {spawners.shape} vs recruits {recruits.shape}")
    else:
        ok(f"Spawners and recruits shapes match: {spawners.shape}")

    n_boot, n_years, h, w = spawners.shape
    expected_shape = (100, 35, 50, 50)
    if spawners.shape == expected_shape:
        ok(f"Shape matches expected {expected_shape}")
    else:
        warn(f"Expected {expected_shape}, got {spawners.shape}")

    if n_years != len(CALENDAR_YEARS):
        fail(f"Data has {n_years} years but calendar has {len(CALENDAR_YEARS)}")
    else:
        ok(f"Year count matches calendar: {n_years}")

    return spawners, recruits


# =========================================================================
# 2. CALENDAR YEAR MAPPING
# =========================================================================

def check_calendar_years():
    section("2. CALENDAR YEAR MAPPING")

    print(f"  Full calendar: {CALENDAR_YEARS[0]}–{CALENDAR_YEARS[-1]} ({len(CALENDAR_YEARS)} years)")

    # Check for gaps
    for i in range(1, len(CALENDAR_YEARS)):
        gap = CALENDAR_YEARS[i] - CALENDAR_YEARS[i - 1]
        if gap > 1:
            warn(f"Gap between index {i-1}→{i}: "
                 f"{CALENDAR_YEARS[i-1]}→{CALENDAR_YEARS[i]} "
                 f"(missing {gap - 1} year{'s' if gap > 2 else ''})")

    # Print split mapping
    train_cal = CALENDAR_YEARS[:TRAIN_YEARS]
    val_cal = CALENDAR_YEARS[TRAIN_YEARS:TRAIN_YEARS + VAL_YEARS]
    test_cal = CALENDAR_YEARS[TRAIN_YEARS + VAL_YEARS:TRAIN_YEARS + VAL_YEARS + TEST_YEARS]

    print(f"\n  Split mapping:")
    print(f"    TRAIN ({TRAIN_YEARS} yrs): indices 0–{TRAIN_YEARS-1} → {train_cal[0]}–{train_cal[-1]}")
    print(f"    VAL   ({VAL_YEARS} yrs):  indices {TRAIN_YEARS}–{TRAIN_YEARS+VAL_YEARS-1} → {val_cal[0]}–{val_cal[-1]}")
    print(f"    TEST  ({TEST_YEARS} yrs):  indices {TRAIN_YEARS+VAL_YEARS}–{TRAIN_YEARS+VAL_YEARS+TEST_YEARS-1} → {test_cal}")

    total_used = TRAIN_YEARS + VAL_YEARS + TEST_YEARS
    if total_used < len(CALENDAR_YEARS):
        warn(f"Using {total_used} of {len(CALENDAR_YEARS)} years "
             f"({len(CALENDAR_YEARS) - total_used} dropped)")
    elif total_used == len(CALENDAR_YEARS):
        ok(f"All {total_used} years used")

    # Check which split contains the 2020 gap
    for name, cal in [("TRAIN", train_cal), ("VAL", val_cal), ("TEST", test_cal)]:
        for i in range(1, len(cal)):
            if cal[i] - cal[i - 1] > 1:
                warn(f"2020 gap falls in {name} split: between {cal[i-1]} and {cal[i]}")

    return train_cal, val_cal, test_cal


# =========================================================================
# 3. NaN / Inf / NEGATIVE CHECK
# =========================================================================

def check_values(spawners, recruits):
    section("3. VALUE INTEGRITY")

    for name, arr in [("Spawners", spawners), ("Recruits", recruits)]:
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        n_neg = (arr < 0).sum()

        if n_nan == 0:
            ok(f"{name}: no NaN values")
        else:
            fail(f"{name}: {n_nan} NaN values!")

        if n_inf == 0:
            ok(f"{name}: no Inf values")
        else:
            fail(f"{name}: {n_inf} Inf values!")

        if n_neg == 0:
            ok(f"{name}: no negative values")
        else:
            warn(f"{name}: {n_neg} negative values (min={arr.min():.4f})")


# =========================================================================
# 4. SPATIAL MASK
# =========================================================================

def check_mask(spawners, recruits):
    section("4. SPATIAL MASK")

    mask_path = os.path.join(DATA_DIR, "spatial_mask.npy")
    if not os.path.exists(mask_path):
        fail("spatial_mask.npy not found")
        return None

    mask = np.load(mask_path)
    valid = mask > 0
    n_valid = valid.sum()
    n_total = mask.size
    print(f"  Valid cells: {n_valid} / {n_total} ({100*n_valid/n_total:.1f}%)")

    # Check that data outside mask is zero
    sp_outside = spawners[:, :, ~valid]
    rc_outside = recruits[:, :, ~valid]

    if np.abs(sp_outside).max() < 1e-6:
        ok("Spawners are zero outside mask")
    else:
        warn(f"Spawners have nonzero values outside mask (max={sp_outside.max():.4f})")

    if np.abs(rc_outside).max() < 1e-6:
        ok("Recruits are zero outside mask")
    else:
        warn(f"Recruits have nonzero values outside mask (max={rc_outside.max():.4f})")

    # Check mask is contiguous-ish (not scattered random pixels)
    n_rows_with_valid = (valid.any(axis=1)).sum()
    n_cols_with_valid = (valid.any(axis=0)).sum()
    print(f"  Mask spans {n_rows_with_valid} rows × {n_cols_with_valid} cols")

    return valid


# =========================================================================
# 5. PER-YEAR STATISTICS
# =========================================================================

def check_per_year(spawners, recruits, valid):
    section("5. PER-YEAR STATISTICS (mean across bootstraps, valid cells)")

    n_boot, n_years, h, w = spawners.shape

    header = f"  {'Idx':>3} {'Year':>6} {'Sp_mean':>10} {'Sp_max':>10} {'Rc_mean':>10} {'Rc_max':>10} {'Sp_nz%':>8} {'Rc_nz%':>8}"
    print(header)
    print("  " + "-" * len(header))

    prev_year = None
    for yi in range(n_years):
        sp = spawners[:, yi][:, valid]  # [n_boot, n_valid]
        rc = recruits[:, yi][:, valid]
        year = CALENDAR_YEARS[yi] if yi < len(CALENDAR_YEARS) else f"?{yi}"

        # Flag gaps
        gap_marker = ""
        if prev_year is not None and isinstance(year, int) and year - prev_year > 1:
            gap_marker = " ← GAP"

        print(f"  {yi:>3} {year:>6} {sp.mean():>10.2f} {sp.max():>10.2f} "
              f"{rc.mean():>10.2f} {rc.max():>10.2f} "
              f"{(sp > 0).mean()*100:>7.1f}% {(rc > 0).mean()*100:>7.1f}%{gap_marker}")

        prev_year = year if isinstance(year, int) else prev_year

    # Check for suspicious years (all zeros, extreme values)
    for yi in range(n_years):
        sp = spawners[:, yi][:, valid]
        if sp.max() == 0:
            warn(f"Year {CALENDAR_YEARS[yi]}: ALL spawner values are zero!")
        rc = recruits[:, yi][:, valid]
        if rc.max() == 0:
            warn(f"Year {CALENDAR_YEARS[yi]}: ALL recruit values are zero!")


# =========================================================================
# 6. FLATTENED ORDERING
# =========================================================================

def check_flattened_ordering(spawners):
    section("6. FLATTENED ORDERING (B0Y0, B0Y1, ...)")

    n_boot, n_years, h, w = spawners.shape
    flat = spawners.reshape(n_boot * n_years, h, w)
    print(f"  4D shape: {spawners.shape} → flattened: {flat.shape}")

    errors = 0
    checks = [(0, 0), (0, 1), (0, n_years - 1),
              (1, 0), (1, 1),
              (n_boot - 1, 0), (n_boot - 1, n_years - 1)]

    for b, y in checks:
        flat_idx = b * n_years + y
        orig = spawners[b, y]
        from_flat = flat[flat_idx]
        match = np.allclose(orig, from_flat)
        status = "✓" if match else "✗"
        if not match:
            errors += 1
        cal = CALENDAR_YEARS[y] if y < len(CALENDAR_YEARS) else "?"
        print(f"  {status} B{b}Y{y} ({cal}) → flat[{flat_idx}]")

    if errors == 0:
        ok("All ordering checks passed")
    else:
        fail(f"{errors} ordering mismatches!")


# =========================================================================
# 7. BOOTSTRAP CONSISTENCY
# =========================================================================

def check_bootstrap_consistency(spawners, valid):
    section("7. BOOTSTRAP CONSISTENCY")

    n_boot, n_years, h, w = spawners.shape
    print("  Different bootstraps should differ (subsampled stations)")
    print("  but share similar spatial patterns (same underlying field)")

    test_years_idx = [0, n_years // 2, n_years - 1]
    for yi in test_years_idx:
        cal = CALENDAR_YEARS[yi]
        # Compare bootstrap 0 vs 1
        b0 = spawners[0, yi][valid]
        b1 = spawners[1, yi][valid]
        b50 = spawners[50, yi][valid]

        identical_01 = np.allclose(b0, b1)
        identical_050 = np.allclose(b0, b50)
        corr_01 = np.corrcoef(b0, b1)[0, 1] if b0.std() > 0 and b1.std() > 0 else 0

        if identical_01:
            warn(f"Year {cal}: B0 and B1 are IDENTICAL (bootstraps may not be independent)")
        else:
            ok(f"Year {cal}: B0≠B1 (corr={corr_01:.4f}), B0≠B50: {not identical_050}")


# =========================================================================
# 8. SPLIT FILE VERIFICATION
# =========================================================================

def check_splits(spawners, recruits):
    section("8. SPLIT FILES")

    if not os.path.exists(SPLITS_DIR):
        warn(f"Splits directory not found: {SPLITS_DIR}")
        print("  Run create_splits.py first, then re-run this diagnostic.")
        return

    n_boot, n_years, h, w = spawners.shape

    split_configs = [
        ("train", TRAIN_YEARS, 0),
        ("val", VAL_YEARS, TRAIN_YEARS),
        ("test", TEST_YEARS, TRAIN_YEARS + VAL_YEARS),
    ]

    for split_name, n_yrs, year_start in split_configs:
        sp_path = os.path.join(SPLITS_DIR, f"{split_name}_spawners_real.npy")
        rc_path = os.path.join(SPLITS_DIR, f"{split_name}_recruits_real.npy")

        if not os.path.exists(sp_path):
            warn(f"{split_name} split not found: {sp_path}")
            continue

        sp_split = np.load(sp_path)
        rc_split = np.load(rc_path)

        expected_samples = n_boot * n_yrs
        if sp_split.shape[0] != expected_samples:
            fail(f"{split_name}: expected {expected_samples} samples, got {sp_split.shape[0]}")
            continue

        ok(f"{split_name}: {sp_split.shape} ({n_yrs} years × {n_boot} bootstraps)")

        # Verify B0Y0 of this split matches raw data
        raw_b0y0 = spawners[0, year_start]
        split_b0y0 = sp_split[0]
        if np.allclose(raw_b0y0, split_b0y0):
            cal = CALENDAR_YEARS[year_start]
            ok(f"  {split_name}[0] matches raw B0 Year {cal} (index {year_start})")
        else:
            fail(f"  {split_name}[0] does NOT match raw data at index {year_start}!")

        # Verify last entry
        raw_last = spawners[n_boot - 1, year_start + n_yrs - 1]
        split_last = sp_split[-1]
        if np.allclose(raw_last, split_last):
            cal = CALENDAR_YEARS[year_start + n_yrs - 1]
            ok(f"  {split_name}[-1] matches raw B{n_boot-1} Year {cal}")
        else:
            fail(f"  {split_name}[-1] does NOT match raw data!")

    # Check calendar years JSON
    cal_path = os.path.join(SPLITS_DIR, "calendar_years.json")
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            cal_data = json.load(f)
        print(f"\n  Calendar years from JSON:")
        for k, v in cal_data.items():
            print(f"    {k}: {v}")
        ok("calendar_years.json present")
    else:
        warn("calendar_years.json not found — run updated create_splits.py")


# =========================================================================
# 9. MEMORY BANK / TEMPORAL MASK SIMULATION
# =========================================================================

def check_memory_bank():
    section("9. MEMORY BANK SIMULATION (2020 gap handling)")

    test_cal = CALENDAR_YEARS[TRAIN_YEARS + VAL_YEARS:TRAIN_YEARS + VAL_YEARS + TEST_YEARS]
    val_cal = CALENDAR_YEARS[TRAIN_YEARS:TRAIN_YEARS + VAL_YEARS]
    val_hist_cal = val_cal[-5:]  # Last 5 val years as historical for test

    test_year_to_rel = {y: i for i, y in enumerate(test_cal)}
    hist_year_to_idx = {y: i for i, y in enumerate(val_hist_cal)}

    print(f"  Test calendar years: {test_cal}")
    print(f"  Val history for test: {val_hist_cal}")
    print()

    for rel_idx, cal_year in enumerate(test_cal):
        print(f"  Year {cal_year} (test relative idx {rel_idx}):")
        mask = [1.0]  # Current year

        for lag in range(1, 6):
            target = cal_year - lag
            source = "???"
            found = False

            if target in test_year_to_rel:
                source = f"test[{test_year_to_rel[target]}]"
                found = True
            elif target in hist_year_to_idx:
                source = f"val_hist[{hist_year_to_idx[target]}]"
                found = True

            mask.append(1.0 if found else 0.0)
            status = "✓" if found else "✗ MISSING"
            print(f"    t-{lag} = {target}: {status} ({source if found else 'not in any split'})")

        print(f"    temporal_mask = {mask}")
        n_valid = sum(mask)
        if n_valid < 6:
            warn(f"    {cal_year} has {int(6 - n_valid)} masked channel(s) due to 2020 gap")
        else:
            ok(f"    {cal_year} has full 5-year history")
        print()


# =========================================================================
# 10. VISUALIZATION
# =========================================================================

def plot_diagnostics(spawners, recruits, valid):
    section("10. VISUALIZATION")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        warn("matplotlib not available, skipping plots")
        return

    n_boot, n_years, h, w = spawners.shape

    # --- Time series ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    sp_totals = spawners[:, :, valid].sum(axis=2)  # [100, 35]
    rc_totals = recruits[:, :, valid].sum(axis=2)

    ax = axes[0]
    ax.fill_between(CALENDAR_YEARS,
                    np.percentile(sp_totals, 5, axis=0),
                    np.percentile(sp_totals, 95, axis=0), alpha=0.2, color='green')
    ax.plot(CALENDAR_YEARS, np.median(sp_totals, axis=0), 'g-o', ms=3, lw=2, label='Spawners')
    ax.axvspan(2019.5, 2020.5, color='red', alpha=0.3)
    ax.axvline(CALENDAR_YEARS[TRAIN_YEARS - 1] + 0.5, color='grey', ls=':', lw=2)
    ax.axvline(CALENDAR_YEARS[TRAIN_YEARS + VAL_YEARS - 1] + 0.5, color='grey', ls=':', lw=2)
    ax.set_title('Total Spawner Abundance (valid cells, median + 5th–95th)')
    ax.set_ylabel('Sum')
    ax.legend()

    ax = axes[1]
    ax.fill_between(CALENDAR_YEARS,
                    np.percentile(rc_totals, 5, axis=0),
                    np.percentile(rc_totals, 95, axis=0), alpha=0.2, color='blue')
    ax.plot(CALENDAR_YEARS, np.median(rc_totals, axis=0), 'b-o', ms=3, lw=2, label='Recruits')
    ax.axvspan(2019.5, 2020.5, color='red', alpha=0.3)
    ax.axvline(CALENDAR_YEARS[TRAIN_YEARS - 1] + 0.5, color='grey', ls=':', lw=2)
    ax.axvline(CALENDAR_YEARS[TRAIN_YEARS + VAL_YEARS - 1] + 0.5, color='grey', ls=':', lw=2)
    ax.set_title('Total Recruit Abundance (valid cells, median + 5th–95th)')
    ax.set_ylabel('Sum')
    ax.legend()

    # Bootstrap spread
    ax = axes[2]
    cv_sp = sp_totals.std(axis=0) / (sp_totals.mean(axis=0) + 1e-6)
    cv_rc = rc_totals.std(axis=0) / (rc_totals.mean(axis=0) + 1e-6)
    ax.plot(CALENDAR_YEARS, cv_sp, 'g-o', ms=3, label='Spawner CV')
    ax.plot(CALENDAR_YEARS, cv_rc, 'b-o', ms=3, label='Recruit CV')
    ax.axvspan(2019.5, 2020.5, color='red', alpha=0.3)
    ax.set_title('Bootstrap Coefficient of Variation by Year')
    ax.set_ylabel('CV (std/mean)')
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(DATA_DIR, "data_diagnostic_timeseries.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()

    # --- Spatial snapshots ---
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    sample_years = [0, 8, 16, 24, 34]  # Spread across timeline
    for col, yi in enumerate(sample_years):
        cal = CALENDAR_YEARS[yi]
        sp_mean = spawners[:, yi].mean(axis=0)  # Mean across bootstraps
        rc_mean = recruits[:, yi].mean(axis=0)

        # Mask out land
        sp_display = sp_mean.copy()
        rc_display = rc_mean.copy()
        sp_display[~valid] = np.nan
        rc_display[~valid] = np.nan

        axes[0, col].imshow(sp_display, cmap='viridis')
        axes[0, col].set_title(f'Spawner {cal}', fontsize=10)
        axes[0, col].axis('off')

        axes[1, col].imshow(rc_display, cmap='plasma')
        axes[1, col].set_title(f'Recruit {cal}', fontsize=10)
        axes[1, col].axis('off')

    fig.suptitle('Spatial Snapshots (bootstrap mean, selected years)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(DATA_DIR, "data_diagnostic_spatial.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


# =========================================================================
# MAIN
# =========================================================================

def run_all():
    print("\n" + "=" * 65)
    print("  EBS SNOW CRAB DATA DIAGNOSTIC")
    print("=" * 65)

    spawners, recruits = check_raw_shapes()
    train_cal, val_cal, test_cal = check_calendar_years()
    check_values(spawners, recruits)
    valid = check_mask(spawners, recruits)
    if valid is None:
        valid = np.ones((50, 50), dtype=bool)
    check_per_year(spawners, recruits, valid)
    check_flattened_ordering(spawners)
    check_bootstrap_consistency(spawners, valid)
    check_splits(spawners, recruits)
    check_memory_bank()
    plot_diagnostics(spawners, recruits, valid)

    section("DONE")
    print("  Review output above for any ⚠️ or ✗ markers.")
    print("  If all checks pass, data is ready for create_splits.py → training.")


if __name__ == "__main__":
    run_all()