"""
baseline_evaluation.py
======================
Two per-cell climatological baselines evaluated against the test set recruits
for the real EBS survey data.

The mean prediction grid is computed in log1p space (matching CrabDataset's
transform and the scale the model was trained on), then back-transformed with
expm1 to count space before metric computation.  This matches the
back-transformation applied to model predictions in run_batch_evaluation.py
(exp(log_pred) * bias_correction - 1; for the baseline there is no bias
correction so the equivalent is expm1).

Baseline 1 — Per-cell mean, training years only.
Baseline 2 — Per-cell mean, training + validation years.

Metrics (all in original count space)
--------------------------------------
MAE:
    Global over all valid (bootstrap x year) grids and valid spatial cells.
    Directly comparable to the MAE column in the results tables.

Spatial Spearman r:
    Per (bootstrap x year) grid — Spearman r over valid spatial cells.
    Mean and SD across 300 grids.
    Matches the per-sample computation in run_batch_evaluation.py.

Temporal Spearman r:
    Per bootstrap — Spearman r of total-abundance time series across valid
    test years (n=3).  Undefined for the baseline (constant prediction).

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
ZERO_THRESHOLD = 14.23   # matches run_batch_evaluation.py


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

    recruits_aligned = recruits[:, lag:]
    recruit_mask     = full_year_mask[lag:]

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
        {"data": recruits[:, :train_years],          "mask": year_mask[:train_years]},
        {"data": recruits[:, train_years:te_start],  "mask": year_mask[train_years:te_start]},
        {"data": recruits[:, te_start:te_end],       "mask": year_mask[te_start:te_end]},
    )


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def make_pred_grid(data, year_mask, spatial_mask):
    """
    Per-cell mean prediction in count space.

    Mean is computed in log1p space (matching CrabDataset), then
    back-transformed with expm1 to count space — matching the
    back-transformation in run_batch_evaluation.py.

    data         : [n_boot, n_years, 50, 50]  count space
    year_mask    : [n_years]  float32
    spatial_mask : [50, 50]  bool

    Returns
    -------
    pred_grid : [50, 50]  count space; land cells set to 0
    """
    valid      = year_mask.astype(bool)
    log_data   = np.log1p(data[:, valid, :, :])      # log1p space
    log_mean   = log_data.mean(axis=(0, 1))           # [50, 50]
    pred_grid  = np.expm1(log_mean)                   # back to count space
    pred_grid[~spatial_mask] = 0.0
    return pred_grid


def compute_mae(pred_grid, test_data, spatial_mask, test_year_mask):
    """Global MAE in count space."""
    valid_years = test_year_mask.astype(bool)
    obs         = test_data[:, valid_years, :, :]
    pred        = np.broadcast_to(pred_grid[np.newaxis, np.newaxis, :, :], obs.shape)
    pred_flat   = pred[:, :, spatial_mask].ravel()
    obs_flat    = obs[:, :, spatial_mask].ravel()
    return float(np.mean(np.abs(pred_flat - obs_flat)))


def compute_spatial_spearman(pred_grid, test_data, spatial_mask, test_year_mask):
    """Per-(bootstrap x year) spatial Spearman r over valid cells."""
    valid_years = test_year_mask.astype(bool)
    obs         = test_data[:, valid_years, :, :]
    n_boot, n_valid_test, h, w = obs.shape
    pred        = np.broadcast_to(pred_grid[np.newaxis, np.newaxis, :, :], obs.shape)

    rhos = []
    for b in range(n_boot):
        for t in range(n_valid_test):
            p = pred[b, t][spatial_mask]
            o = obs[b, t][spatial_mask]
            if np.std(p) > 0 and np.std(o) > 0:
                r, _ = spearmanr(p, o)
            else:
                r = 0.0
            rhos.append(r)
    return np.array(rhos)


def compute_temporal_spearman(pred_grid, test_data, spatial_mask, test_year_mask):
    """Per-bootstrap temporal Spearman r across valid test years."""
    valid_years = test_year_mask.astype(bool)
    obs         = test_data[:, valid_years, :, :]
    n_boot, n_valid_test, h, w = obs.shape

    pred_total  = float(pred_grid[spatial_mask].sum())
    pred_totals = np.full(n_valid_test, pred_total)   # constant — will be NaN

    rhos = []
    for b in range(n_boot):
        obs_totals = obs[b, :, :, :][:, spatial_mask].sum(axis=1)
        if np.std(pred_totals) < 1e-10:
            rhos.append(np.nan)
        else:
            r, _ = spearmanr(pred_totals, obs_totals)
            rhos.append(r)
    return np.array(rhos)


def report_baseline(label, pred_grid, test_data, spatial_mask, test_split):
    year_mask = test_split["mask"]
    n_valid   = int(year_mask.sum())
    n_boot    = test_data.shape[0]
    n_grids   = n_boot * n_valid

    # Apply zero threshold to both prediction and observations,
    # matching the thresh_* variant in run_batch_evaluation.py
    pred_grid_thr          = pred_grid.copy()
    pred_grid_thr[pred_grid_thr < ZERO_THRESHOLD] = 0.0

    test_data_thr          = test_data.copy()
    test_data_thr[test_data_thr < ZERO_THRESHOLD] = 0.0

    mae           = compute_mae(pred_grid_thr, test_data_thr, spatial_mask, year_mask)
    spatial_rhos  = compute_spatial_spearman(pred_grid_thr, test_data_thr, spatial_mask, year_mask)
    temporal_rhos = compute_temporal_spearman(pred_grid_thr, test_data_thr, spatial_mask, year_mask)
    n_temp_valid  = int(np.sum(~np.isnan(temporal_rhos)))

    print(f"\n  Prediction grid (valid cells, count space, threshold={ZERO_THRESHOLD}):")
    print(f"    min  = {pred_grid_thr[spatial_mask].min():.1f}")
    print(f"    max  = {pred_grid_thr[spatial_mask].max():.1f}")
    print(f"    mean = {pred_grid_thr[spatial_mask].mean():.1f}")
    print(f"  MAE                  : {mae:.1f}  "
          f"(n = {n_grids} grids x {spatial_mask.sum()} cells)")
    print(f"  Spatial Spearman r   : {spatial_rhos.mean():.4f}  "
          f"(SD = {spatial_rhos.std():.4f}, n = {n_grids} grids)")
    if n_temp_valid == 0:
        print(f"  Temporal Spearman r  : undefined (constant prediction across years)")
    else:
        print(f"  Temporal Spearman r  : {np.nanmean(temporal_rhos):.4f}  "
              f"(SD = {np.nanstd(temporal_rhos):.4f}, n = {n_temp_valid} bootstraps)")

    return {
        "label":             label,
        "mae":               mae,
        "spatial_rho_mean":  float(spatial_rhos.mean()),
        "spatial_rho_sd":    float(spatial_rhos.std()),
        "temporal_rho_mean": float(np.nanmean(temporal_rhos)),
        "temporal_rho_sd":   float(np.nanstd(temporal_rhos)),
        "n_grids":           n_grids,
        "n_bootstraps":      n_temp_valid,
    }


# =============================================================================
# MAIN
# =============================================================================

def main(lag=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lag",      type=int, default=LAG)
    parser.add_argument("--data_dir", type=str, default="data/real")
    args, _ = parser.parse_known_args()   # ignore Jupyter kernel args
    if lag is not None:
        args.lag = lag  # allow programmatic override

    if args.lag == 5:
        train_years, val_years, test_years = 21, 6, 4
    else:
        train_years, val_years, test_years = TRAIN_YEARS, VAL_YEARS, TEST_YEARS

    data_dir     = Path(args.data_dir)
    spatial_mask = load_spatial_mask(data_dir)

    print("=" * 64)
    print(f"Lag               : {args.lag}")
    print(f"Train / Val / Test: {train_years} / {val_years} / {test_years} years")
    print(f"Mean computed in  : log1p space (back-transformed to counts for metrics)")
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

    # test_data in count space; thresholding is applied inside report_baseline
    # matching run_batch_evaluation.py thresh_* variant
    test_data = test_split["data"].astype(np.float32)

    # ── Baseline 1: training years only ───────────────────────────────────────
    mean_grid_1 = make_pred_grid(
        train_split["data"], train_split["mask"], spatial_mask
    )

    # Save grids for top_model_figure.py
    np.save(Path(f"baseline_mean_grid_lag{args.lag}.npy"), mean_grid_1)
    obs_mean_grid = test_split["data"][:, test_split["mask"].astype(bool)].mean(axis=(0, 1))
    obs_mean_grid[~spatial_mask] = 0.0
    np.save(Path(f"baseline_obs_mean_lag{args.lag}.npy"), obs_mean_grid)
    print(f"Saved baseline_mean_grid_lag{args.lag}.npy and "
          f"baseline_obs_mean_lag{args.lag}.npy")
    print("\n" + "-" * 64)
    print("BASELINE 1 — Per-cell mean (training years only)")
    print("-" * 64)
    res1 = report_baseline("train_only", mean_grid_1, test_data, spatial_mask, test_split)

    # ── Baseline 2: training + validation years ───────────────────────────────
    trainval_data = np.concatenate(
        [train_split["data"], val_split["data"]], axis=1
    )
    trainval_mask = np.concatenate(
        [train_split["mask"], val_split["mask"]]
    )
    mean_grid_2 = make_pred_grid(trainval_data, trainval_mask, spatial_mask)
    print("\n" + "-" * 64)
    print("BASELINE 2 — Per-cell mean (training + validation years)")
    print("-" * 64)
    res2 = report_baseline("train_val", mean_grid_2, test_data, spatial_mask, test_split)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("SUMMARY (count space)")
    print("=" * 76)
    print(f"{'Baseline':<44} {'MAE':>8} {'Spatial r':>10} {'Temporal r':>11}")
    print("-" * 76)
    for res in [res1, res2]:
        t_r = (f"{res['temporal_rho_mean']:.4f}"
               if not np.isnan(res['temporal_rho_mean']) else "  undef")
        print(f"{res['label']:<44} {res['mae']:>8.1f} "
              f"{res['spatial_rho_mean']:>10.4f} {t_r:>11}")
    print("=" * 76)

    out_path = Path(f"baseline_results_lag{args.lag}.txt")
    with open(out_path, "w") as f:
        f.write(f"Lag: {args.lag}\n")
        f.write(f"Train / Val / Test: {train_years} / {val_years} / {test_years} years\n")
        f.write(f"Mean in log1p space; metrics in count space\n")
        f.write(f"Valid spatial cells: {spatial_mask.sum()} / {spatial_mask.size}\n\n")
        f.write(f"{'Baseline':<44} {'MAE':>8} {'Spatial r (mean±SD)':>22} "
                f"{'Temporal r':>12}\n")
        f.write("-" * 90 + "\n")
        for res in [res1, res2]:
            t_str = (f"{res['temporal_rho_mean']:.4f} ± {res['temporal_rho_sd']:.4f}"
                     if not np.isnan(res['temporal_rho_mean']) else "undefined")
            f.write(f"{res['label']:<44} {res['mae']:>8.1f} "
                    f"{res['spatial_rho_mean']:.4f} ± {res['spatial_rho_sd']:.4f}  "
                    f"{t_str}\n")
    print(f"\nResults written to {out_path}")


    # ── Per-year metrics (training-years-only baseline) ───────────────────────
    print("\nPer-year breakdown (Baseline 1 — training years only):")

    valid_test_years  = test_split["mask"].astype(bool)
    obs_test          = test_data[:, valid_test_years, :, :]
    pred_grid_thr     = mean_grid_1.copy()
    pred_grid_thr[pred_grid_thr < 14.23] = 0.0

    valid_slot_indices = np.where(valid_test_years)[0]
    cal_years = [1988 + train_years + val_years + slot + args.lag
                 for slot in valid_slot_indices]

    per_year_rows = []
    n_boot = obs_test.shape[0]

    for t_idx, cal_year in enumerate(cal_years):
        maes = []
        rhos = []
        for b in range(n_boot):
            obs_thr  = obs_test[b, t_idx][spatial_mask].copy()
            obs_thr[obs_thr < 14.23] = 0.0
            pred_thr = pred_grid_thr[spatial_mask]
            maes.append(float(np.mean(np.abs(pred_thr - obs_thr))))
            if np.std(pred_thr) > 0 and np.std(obs_thr) > 0:
                from scipy.stats import spearmanr as _sr
                r, _ = _sr(pred_thr, obs_thr)
            else:
                r = 0.0
            rhos.append(r)

        per_year_rows.append({
            'lag':           args.lag,
            'calendar_year': cal_year,
            'mae_mean':      round(float(np.mean(maes)), 1),
            'mae_sd':        round(float(np.std(maes)),  1),
            'spearman_mean': round(float(np.mean(rhos)), 4),
            'spearman_sd':   round(float(np.std(rhos)),  4),
            'n_bootstraps':  n_boot,
        })
        print(f"  {cal_year}: MAE={np.mean(maes):.1f} ± {np.std(maes):.1f}  "
              f"Spearman={np.mean(rhos):.4f} ± {np.std(rhos):.4f}")

    per_year_df = pd.DataFrame(per_year_rows)
    per_year_path = Path(f"baseline_per_year_lag{args.lag}.csv")
    per_year_df.to_csv(per_year_path, index=False)
    print(f"Per-year results written to {per_year_path}")


if __name__ == "__main__":
    for _lag in [0, 5]:
        print("\n" + "=" * 64)
        print(f"Running baseline for lag={_lag}")
        main(lag=_lag)