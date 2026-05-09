"""
baseline_evaluation_dummy.py
============================
Per-bootstrap climatological baseline evaluated against the test set recruits
for each synthetic difficulty level (easy, medium, hard).

For each bootstrap:
  1. Compute per-cell mean from that bootstrap's training years (log1p space,
     back-transformed to count space) — one mean grid per bootstrap.
  2. Evaluate that grid against that bootstrap's test years only.
  3. Collect one MAE and one Spearman per (bootstrap x test year).

This matches the model evaluation structure in run_batch_evaluation.py,
where each bootstrap produces one prediction per year.

Split logic mirrors create_splits.py exactly:
    - 100 bootstraps x 30 years, flat array [3000, 50, 50]
    - Train / Val / Test : 18 / 9 / 3 years
    - No year mask (all years valid)
    - No spatial mask (no land cells)

Usage
-----
    python baseline_evaluation_dummy.py [--data_dir data/dummy]
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr


# =============================================================================
# CONFIG
# =============================================================================

TRAIN_YEARS  = 18
VAL_YEARS    = 9
TEST_YEARS   = 3
N_BOOTSTRAPS = 100
LEVELS       = ['easy', 'medium', 'hard']


# =============================================================================
# DATA LOADING
# =============================================================================

def load_level(data_dir: Path, level: str):
    """
    Load recruits for one difficulty level and reshape from flat to
    [n_boot, n_years, 50, 50].
    """
    path = data_dir / "output" / f"gmrf_recruits_50x50_{level}.npy"
    flat = np.load(path).astype(np.float32)
    n_years = flat.shape[0] // N_BOOTSTRAPS
    return flat.reshape(N_BOOTSTRAPS, n_years, 50, 50)


def split_recruits(recruits):
    te_start = TRAIN_YEARS + VAL_YEARS
    te_end   = te_start + TEST_YEARS
    return (
        recruits[:, :TRAIN_YEARS],
        recruits[:, TRAIN_YEARS:te_start],
        recruits[:, te_start:te_end],
    )


# =============================================================================
# CORE: PER-BOOTSTRAP MEAN GRIDS
# =============================================================================

def make_per_bootstrap_pred_grids(train_data):
    """
    Compute one per-cell mean prediction grid per bootstrap.

    Mean computed in log1p space (matching CrabDataset), back-transformed
    with expm1 to count space.

    Parameters
    ----------
    train_data : [n_boot, n_train_years, 50, 50]  count space

    Returns
    -------
    pred_grids : [n_boot, 50, 50]  count space
    """
    log_data = np.log1p(train_data)              # [n_boot, n_train, 50, 50]
    log_mean = log_data.mean(axis=1)             # [n_boot, 50, 50]
    return np.expm1(log_mean).astype(np.float32) # [n_boot, 50, 50]


# =============================================================================
# METRICS
# =============================================================================

def evaluate_per_bootstrap(pred_grids, test_data):
    """
    For each bootstrap, evaluate its own mean grid against its own test years.

    Parameters
    ----------
    pred_grids : [n_boot, 50, 50]
    test_data  : [n_boot, n_test_years, 50, 50]

    Returns
    -------
    records : list of dicts, one per (bootstrap, test year)
    """
    n_boot, n_test, h, w = test_data.shape
    records = []

    for b in range(n_boot):
        pred_flat = pred_grids[b].ravel()

        for t in range(n_test):
            obs_flat = test_data[b, t].ravel()
            mae      = float(np.mean(np.abs(pred_flat - obs_flat)))

            if np.std(pred_flat) > 0 and np.std(obs_flat) > 0:
                r, _ = spearmanr(pred_flat, obs_flat)
            else:
                r = 0.0

            records.append({
                'bootstrap': b,
                'year_slot': t,
                'mae':       mae,
                'spearman':  float(r),
            })

    return records


# =============================================================================
# EVALUATE ONE LEVEL
# =============================================================================

def evaluate_level(data_dir: Path, level: str):
    recruits = load_level(data_dir, level)
    n_boot, n_years, h, w = recruits.shape

    print(f"\n  Recruits shape : {recruits.shape}")
    print(f"  Count stats    : min={recruits.min():.1f}  "
          f"max={recruits.max():.1f}  mean={recruits.mean():.1f}")

    train_data, val_data, test_data = split_recruits(recruits)

    # Per-bootstrap prediction grids
    pred_grids = make_per_bootstrap_pred_grids(train_data)
    print(f"  pred_grids shape : {pred_grids.shape}")

    # Evaluate
    records = evaluate_per_bootstrap(pred_grids, test_data)
    df      = pd.DataFrame(records)

    # Per-year summary
    per_year = []
    print(f"\n  {'Year slot':<12} {'MAE':>8} {'±':>6} {'Spearman':>10} {'±':>7} {'n':>5}")
    print(f"  {'-'*50}")
    for t in range(TEST_YEARS):
        sub      = df[df['year_slot'] == t]
        mae_mean = sub['mae'].mean()
        mae_sd   = sub['mae'].std()
        sp_mean  = sub['spearman'].mean()
        sp_sd    = sub['spearman'].std()
        print(f"  Test year {t+1:<5}  {mae_mean:>8.1f} {mae_sd:>6.1f} "
              f"{sp_mean:>10.4f} {sp_sd:>7.4f} {len(sub):>5}")
        per_year.append({
            'level':         level,
            'test_year':     t + 1,
            'mae_mean':      round(mae_mean, 1),
            'mae_sd':        round(mae_sd,   1),
            'spearman_mean': round(sp_mean,  4),
            'spearman_sd':   round(sp_sd,    4),
            'n_bootstraps':  len(sub),
        })

    overall_mae = df['mae'].mean()
    overall_sp  = df['spearman'].mean()
    print(f"\n  Overall MAE = {overall_mae:.1f}   Overall Spearman = {overall_sp:.4f}")
    print(f"  (n = {len(df)} bootstrap x year evaluations)")

    return {
        'level':            level,
        'mae':              overall_mae,
        'spearman':         overall_sp,
        'mae_sd':           df['mae'].std(),
        'spearman_sd':      df['spearman'].std(),
        'per_year':         per_year,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/dummy")
    args, _ = parser.parse_known_args()

    data_dir = Path(args.data_dir)

    print("=" * 64)
    print("Dummy data — per-bootstrap mean baseline")
    print("Mean in log1p space per bootstrap; metrics in count space")
    print(f"Train / Val / Test : {TRAIN_YEARS} / {VAL_YEARS} / {TEST_YEARS} years")
    print(f"Bootstraps         : {N_BOOTSTRAPS}")
    print("=" * 64)

    all_results  = []
    all_per_year = []

    for level in LEVELS:
        print(f"\n{'─' * 64}")
        print(f"LEVEL: {level.upper()}")
        print(f"{'─' * 64}")
        res = evaluate_level(data_dir, level)
        all_results.append(res)
        all_per_year.extend(res['per_year'])

    print("\n" + "=" * 76)
    print("SUMMARY (count space, mean across all bootstrap x year evaluations)")
    print("=" * 76)
    print(f"{'Level':<10} {'MAE':>10} {'MAE SD':>8} {'Spearman':>12} {'Spear SD':>10}")
    print("-" * 56)
    for res in all_results:
        print(f"{res['level']:<10} {res['mae']:>10.1f} {res['mae_sd']:>8.1f} "
              f"{res['spearman']:>12.4f} {res['spearman_sd']:>10.4f}")
    print("=" * 76)

    # Save per-year CSV
    per_year_df = pd.DataFrame(all_per_year)
    per_year_df.to_csv("baseline_per_year_dummy.csv", index=False)
    print(f"\nPer-year results written to baseline_per_year_dummy.csv")

    # Save summary txt
    out_path = Path("baseline_results_dummy.txt")
    with open(out_path, "w") as f:
        f.write("Dummy data — per-bootstrap mean baseline\n")
        f.write("Mean in log1p space per bootstrap; metrics in count space\n")
        f.write(f"Train / Val / Test : {TRAIN_YEARS} / {VAL_YEARS} / {TEST_YEARS} years\n")
        f.write(f"Bootstraps         : {N_BOOTSTRAPS}\n\n")
        f.write(f"{'Level':<10} {'MAE':>10} {'MAE SD':>8} "
                f"{'Spearman':>12} {'Spear SD':>10}\n")
        f.write("-" * 56 + "\n")
        for res in all_results:
            f.write(f"{res['level']:<10} {res['mae']:>10.1f} {res['mae_sd']:>8.1f} "
                    f"{res['spearman']:>12.4f} {res['spearman_sd']:>10.4f}\n")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
    