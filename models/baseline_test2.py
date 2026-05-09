"""
baseline_evaluation.py
======================
Per-bootstrap climatological baseline evaluated against the test set recruits
for the real EBS survey data.

For each bootstrap:
  1. Compute per-cell mean from that bootstrap's training years (log1p space,
     back-transformed to count space) — one mean grid per bootstrap.
  2. Evaluate that grid against that bootstrap's test years only.
  3. Collect one MAE and one Spearman per (bootstrap x valid test year).

This matches the model evaluation structure in run_batch_evaluation.py,
where each bootstrap produces one prediction per year.

The pooled mean grid (mean across all bootstraps) is also saved for use
in top_model_figure.py spatial panels.

Usage
-----
    python baseline_evaluation.py [--lag 0|5] [--data_dir data/real]
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr


# =============================================================================
# CONFIG
# =============================================================================

TRAIN_YEARS    = 24
VAL_YEARS      = 8
TEST_YEARS     = 4
LAG            = 0
ZERO_THRESHOLD = 14.23


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_align(data_dir: Path, lag: int):
    recruits = np.load(data_dir / "output" / "gridded_recruits.npy")
    n_boot, n_years_total, h, w = recruits.shape

    mask_path = data_dir / "output" / "year_mask.npy"
    if mask_path.exists():
        full_year_mask = np.load(mask_path).astype(np.float32)
    else:
        full_year_mask = np.ones(n_years_total, dtype=np.float32)
        full_year_mask[32] = 0.0
        print("  year_mask.npy not found; using hardcoded 2020 index.")

    recruits_aligned  = recruits[:, lag:]
    recruit_mask      = full_year_mask[lag:]

    if lag > 0:
        spawner_mask      = full_year_mask[:n_years_total - lag]
        year_mask_aligned = (recruit_mask * spawner_mask).astype(np.float32)
    else:
        year_mask_aligned = recruit_mask.astype(np.float32)

    return recruits_aligned, year_mask_aligned


def load_spatial_mask(data_dir: Path):
    mask_path = data_dir / "output" / "spatial_mask.npy"
    if mask_path.exists():
        return np.load(mask_path).astype(bool)
    print("  spatial_mask.npy not found; treating all cells as valid.")
    return np.ones((50, 50), dtype=bool)


def split_recruits(recruits, year_mask, train_years, val_years, test_years):
    te_start = train_years + val_years
    te_end   = te_start + test_years
    return (
        {"data": recruits[:, :train_years],         "mask": year_mask[:train_years]},
        {"data": recruits[:, train_years:te_start], "mask": year_mask[train_years:te_start]},
        {"data": recruits[:, te_start:te_end],      "mask": year_mask[te_start:te_end]},
    )


# =============================================================================
# CORE: PER-BOOTSTRAP MEAN GRIDS
# =============================================================================

def make_per_bootstrap_pred_grids(train_data, year_mask, spatial_mask):
    """
    Compute one per-cell mean prediction grid per bootstrap.

    Mean computed in log1p space (matching CrabDataset), back-transformed
    with expm1 to count space.

    Parameters
    ----------
    train_data   : [n_boot, n_train_years, 50, 50]  count space
    year_mask    : [n_train_years]  float32  (0 = invalid year e.g. 2020)
    spatial_mask : [50, 50]  bool

    Returns
    -------
    pred_grids : [n_boot, 50, 50]  count space; land cells set to 0
    """
    valid    = year_mask.astype(bool)
    log_data = np.log1p(train_data[:, valid, :, :])   # [n_boot, n_valid, 50, 50]
    log_mean = log_data.mean(axis=1)                   # [n_boot, 50, 50]
    pred     = np.expm1(log_mean)                      # [n_boot, 50, 50]
    pred[:, ~spatial_mask] = 0.0
    return pred.astype(np.float32)


# =============================================================================
# METRICS
# =============================================================================

def evaluate_per_bootstrap(pred_grids, test_data, spatial_mask, test_year_mask):
    """
    For each bootstrap, evaluate its own mean grid against its own test years.

    Parameters
    ----------
    pred_grids    : [n_boot, 50, 50]  one mean grid per bootstrap
    test_data     : [n_boot, n_test_years, 50, 50]  count space
    spatial_mask  : [50, 50]  bool
    test_year_mask: [n_test_years]  float32

    Returns
    -------
    records : list of dicts, one per (bootstrap, valid test year)
        keys: bootstrap, year_slot, mae, spearman
    """
    valid_years = test_year_mask.astype(bool)
    n_boot      = pred_grids.shape[0]
    records     = []

    for b in range(n_boot):
        pred = pred_grids[b].copy()
        pred[pred < ZERO_THRESHOLD] = 0.0
        pred_flat = pred[spatial_mask]

        for t_idx in np.where(valid_years)[0]:
            obs = test_data[b, t_idx].copy()
            obs[obs < ZERO_THRESHOLD] = 0.0
            obs_flat = obs[spatial_mask]

            mae = float(np.mean(np.abs(pred_flat - obs_flat)))

            if np.std(pred_flat) > 0 and np.std(obs_flat) > 0:
                r, _ = spearmanr(pred_flat, obs_flat)
            else:
                r = 0.0

            records.append({
                'bootstrap': b,
                'year_slot': int(t_idx),
                'mae':       mae,
                'spearman':  float(r),
            })

    return records


# =============================================================================
# REPORTING
# =============================================================================

def report_results(records, test_year_mask, train_years, val_years, lag):
    """Print and return per-year summary matching run_batch_evaluation.py format."""
    df          = pd.DataFrame(records)
    valid_slots = np.where(test_year_mask.astype(bool))[0]
    per_year    = []

    print(f"\n  {'Year':<8} {'MAE':>8} {'±':>6} {'Spearman':>10} {'±':>7} {'n':>5}")
    print(f"  {'-'*48}")

    for slot in valid_slots:
        cal_year = 1988 + train_years + val_years + int(slot) + lag
        sub      = df[df['year_slot'] == slot]
        mae_mean = sub['mae'].mean()
        mae_sd   = sub['mae'].std()
        sp_mean  = sub['spearman'].mean()
        sp_sd    = sub['spearman'].std()
        n        = len(sub)

        print(f"  {cal_year:<8} {mae_mean:>8.1f} {mae_sd:>6.1f} "
              f"{sp_mean:>10.4f} {sp_sd:>7.4f} {n:>5}")

        per_year.append({
            'lag':           lag,
            'calendar_year': cal_year,
            'mae_mean':      round(mae_mean, 1),
            'mae_sd':        round(mae_sd,   1),
            'spearman_mean': round(sp_mean,  4),
            'spearman_sd':   round(sp_sd,    4),
            'n_bootstraps':  n,
        })

    overall_mae = df['mae'].mean()
    overall_sp  = df['spearman'].mean()
    print(f"\n  Overall MAE = {overall_mae:.1f}   Overall Spearman = {overall_sp:.4f}")
    print(f"  (n = {len(df)} bootstrap x year evaluations)")

    return per_year, overall_mae, overall_sp


# =============================================================================
# MAIN
# =============================================================================

def main(lag=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lag",      type=int, default=LAG)
    parser.add_argument("--data_dir", type=str, default="data/real")
    args, _ = parser.parse_known_args()
    if lag is not None:
        args.lag = lag

    if args.lag == 5:
        train_years, val_years, test_years = 21, 6, 4
    else:
        train_years, val_years, test_years = TRAIN_YEARS, VAL_YEARS, TEST_YEARS

    data_dir     = Path(args.data_dir)
    spatial_mask = load_spatial_mask(data_dir)

    print("=" * 64)
    print(f"Lag               : {args.lag}")
    print(f"Train / Val / Test: {train_years} / {val_years} / {test_years} years")
    print(f"Mean computed in  : log1p space per bootstrap (back-transformed to counts)")
    print(f"Valid spatial cells: {spatial_mask.sum()} / {spatial_mask.size}")
    print("=" * 64)

    recruits, year_mask = load_and_align(data_dir, lag=args.lag)
    print(f"\nAligned recruits shape : {recruits.shape}")
    print(f"Aligned year mask      : {year_mask}  ({int(year_mask.sum())} valid years)")

    train_split, val_split, test_split = split_recruits(
        recruits, year_mask, train_years, val_years, test_years
    )
    print(f"\nTrain : {train_split['data'].shape}  "
          f"valid years = {int(train_split['mask'].sum())}")
    print(f"Val   : {val_split['data'].shape}  "
          f"valid years = {int(val_split['mask'].sum())}")
    print(f"Test  : {test_split['data'].shape}  "
          f"valid years = {int(test_split['mask'].sum())}")

    test_data = test_split["data"].astype(np.float32)

    # ── Per-bootstrap prediction grids ───────────────────────────────────────
    print("\nComputing per-bootstrap mean grids (training years only)...")
    pred_grids = make_per_bootstrap_pred_grids(
        train_split["data"], train_split["mask"], spatial_mask
    )
    print(f"  pred_grids shape : {pred_grids.shape}")
    print(f"  Mean across bootstraps: "
          f"min={pred_grids[:, spatial_mask].min():.1f}  "
          f"max={pred_grids[:, spatial_mask].max():.1f}  "
          f"mean={pred_grids[:, spatial_mask].mean():.1f}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating per-bootstrap grids against test years...")
    records = evaluate_per_bootstrap(
        pred_grids, test_data, spatial_mask, test_split["mask"]
    )

    print("\n" + "-" * 64)
    print("PER-YEAR RESULTS (mean ± SD across 100 bootstraps)")
    print("-" * 64)
    per_year, overall_mae, overall_sp = report_results(
        records, test_split["mask"], train_years, val_years, args.lag
    )

    # ── Save per-year CSV ─────────────────────────────────────────────────────
    per_year_df   = pd.DataFrame(per_year)
    per_year_path = Path(f"baseline_per_year_lag{args.lag}.csv")
    per_year_df.to_csv(per_year_path, index=False)
    print(f"\nPer-year results written to {per_year_path}")

    # ── Save pooled mean grid for top_model_figure.py ────────────────────────
    # Pool across bootstraps for spatial map display only (not used for metrics)
    pooled_mean_grid = pred_grids.mean(axis=0)
    np.save(Path(f"baseline_mean_grid_lag{args.lag}.npy"), pooled_mean_grid)

    obs_mean_grid = test_split["data"][:, test_split["mask"].astype(bool)].mean(axis=(0, 1))
    obs_mean_grid[~spatial_mask] = 0.0
    np.save(Path(f"baseline_obs_mean_lag{args.lag}.npy"), obs_mean_grid)
    print(f"Saved baseline_mean_grid_lag{args.lag}.npy and "
          f"baseline_obs_mean_lag{args.lag}.npy")

    # ── Write summary txt ─────────────────────────────────────────────────────
    out_path = Path(f"baseline_results_lag{args.lag}.txt")
    with open(out_path, "w") as f:
        f.write(f"Lag: {args.lag}\n")
        f.write(f"Train / Val / Test: {train_years} / {val_years} / {test_years} years\n")
        f.write(f"Per-bootstrap mean grids; metrics in count space\n")
        f.write(f"Valid spatial cells: {spatial_mask.sum()} / {spatial_mask.size}\n\n")
        f.write(f"Overall MAE: {overall_mae:.1f}\n")
        f.write(f"Overall Spearman: {overall_sp:.4f}\n\n")
        f.write(f"{'Year':<8} {'MAE':>8} {'MAE SD':>8} "
                f"{'Spearman':>10} {'Spear SD':>10} {'n':>6}\n")
        f.write("-" * 56 + "\n")
        for row in per_year:
            f.write(f"{row['calendar_year']:<8} {row['mae_mean']:>8.1f} "
                    f"{row['mae_sd']:>8.1f} {row['spearman_mean']:>10.4f} "
                    f"{row['spearman_sd']:>10.4f} {row['n_bootstraps']:>6}\n")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    for _lag in [0, 5]:
        print("\n" + "=" * 64)
        print(f"Running baseline for lag={_lag}")
        main(lag=_lag)