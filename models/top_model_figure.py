"""
top_model_figure.py
====================
Creates a 7-column spatial + line-plot summary figure for the top
CrabTransformer runs.

Column layout (left → right)
------------------------------
  0  Spawner history   — stacked cards: t-1 (front) … t-5 (back)
  1  Spawner current   — t, or blank if config excludes it
  2  Temp history      — stacked cards: t-1 … t-5
  3  Temp current      — t, or blank if config excludes it
  4  Recruit history   — stacked cards: t-1 … t-5
  5  True recruit      — target for the displayed test sample
  6  Predicted recruit — model output for the same sample

One row per entry in TARGET_RUNS.  Each row has a unique border colour that
is also used for that config's line in the aggregate-recruit line plot below
the grid.

Usage
-----
  Adjust REPO_DIR, DRIVE_BASE, SAVE_DIR, and TARGET_RUNS at the top, then:

      python models/top_model_figure.py

  Or paste the whole file into a Colab notebook cell and call make_figure().
  GPU is used if available but the script runs on CPU too.
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ── Paths — adjust for your environment ────────────────────────────────────
REPO_DIR   = '/content/Teleconnections-ViT'
DRIVE_BASE = '/content/drive/MyDrive/Teleconnection_ViT/model_outputs'
SAVE_DIR   = '/content/drive/MyDrive/Teleconnection_ViT/analysis'

MEMORY_YEARS = 5
BATCH_SIZE   = 8

sys.path.insert(0, REPO_DIR)
from models.model     import CrabTransformer
from data.data_helper import get_dataloaders

os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── One colour per row — extend if you have more than 8 runs ───────────────
ROW_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    '#ff7f00', '#a65628', '#f781bf', '#555555',
]

# ── Runs to display — (model_size, level, channel_cfg, pred_mode, criterion)
TARGET_RUNS = [
    ('normal', 'real', 'all', 'normal', 'MSE'),
]

# ── Column header strings ───────────────────────────────────────────────────
COL_TITLES = [
    'Spawner history\n(t-1 … t-5)',
    'Spawner current\n(t)',
    'Temp history\n(t-1 … t-5)',
    'Temp current\n(t)',
    'Recruit history\n(t-1 … t-5)',
    'True recruit\n(target)',
    'Predicted recruit',
]

# ── Colour maps per channel group ───────────────────────────────────────────
CMAPS = {'spawner': 'YlOrRd', 'temp': 'RdBu_r', 'recruit': 'Blues'}


# ============================================================
#  HELPERS
# ============================================================

def get_year_splits(data_type, lag):
    if data_type == 'real':
        return (24, 8, 4) if lag == 0 else (21, 6, 4)
    return (18, 9, 3)


def load_run(model_size, level, channel_cfg, pred_mode, criterion):
    """
    Load checkpoint + metadata, return model and dataloaders.
    Channel metadata is read from training_history.json so this function
    works without re-running create_splits.
    """
    data_type = 'real' if level == 'real' else 'dummy'
    run_dir   = os.path.join(DRIVE_BASE, model_size, level,
                             channel_cfg, pred_mode, criterion)
    hist_path = os.path.join(run_dir, 'training_history.json')
    ckpt_path = os.path.join(run_dir, 'best_model.pt')

    with open(hist_path) as f:
        hist = json.load(f)

    meta = hist['channel_cfg_meta']
    bias = hist.get('bias_correction', 1.0)

    t_yr, v_yr, te_yr = get_year_splits(data_type, meta['lag'])

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, memory_years=MEMORY_YEARS,
        train_years=t_yr, val_years=v_yr, test_years=te_yr,
        level=level, data_type=data_type,
        include_current_spawner=meta['incl_curr'],
        lag=meta['lag'],
        use_temp=meta['use_temp'],
        use_spawners=meta['use_spawners'],
        use_recruits=meta['use_recruits'],
    )

    model = CrabTransformer(
        grid_size=50, patch_size=5,
        in_channels=meta['in_channels'],
        embed_dim=meta['embed_dim'],
        num_heads=meta['num_heads'],
        num_layers=meta['num_layers'],
        d_ff=meta['d_ff'],
        dropout=0.0,
        channel_mask_indices=meta['channel_mask_indices'],
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    return model, meta, bias, t_yr, v_yr, te_yr, train_loader, val_loader, test_loader


def extract_channels(inp, meta):
    """
    Split a [C, 50, 50] input tensor into named channel groups.

    Returns a dict:
      sp_hist   — list of 5 arrays  [t-1, t-2, t-3, t-4, t-5], or None
      sp_curr   — array (t),  or None
      temp_hist — list of 5 arrays, or None
      temp_curr — array (t),  or None
      rec_hist  — list of 5 arrays, or None
    """
    ch = {k: None for k in ('sp_curr', 'sp_hist',
                              'temp_curr', 'temp_hist', 'rec_hist')}
    idx  = 0
    incl = meta['incl_curr']

    if meta['use_spawners']:
        if incl:
            ch['sp_curr'] = inp[idx];  idx += 1
        ch['sp_hist'] = [inp[idx + k] for k in range(5)];  idx += 5

    if meta['use_recruits']:
        ch['rec_hist'] = [inp[idx + k] for k in range(5)];  idx += 5

    if meta['use_temp']:
        if incl:
            ch['temp_curr'] = inp[idx];  idx += 1
        ch['temp_hist'] = [inp[idx + k] for k in range(5)];  idx += 5

    return ch


# ── Drawing helpers ─────────────────────────────────────────────────────────

def _blank(ax, msg='N/A'):
    """Fill an axes cell with a grey placeholder."""
    ax.set_facecolor('#eeeeee')
    ax.text(0.5, 0.5, msg, ha='center', va='center',
            transform=ax.transAxes, fontsize=8, color='#888888',
            style='italic')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_stack(ax, images, cmap, vmin, vmax, border_color):
    """
    Draw a list of 2-D arrays as a deck of stacked cards.

    images[0] = t-1  (most recent  →  front card, no offset)
    images[-1] = t-5 (oldest       →  back card,  most offset)

    Older cards peek out from the top-right corner.  Each card is drawn
    as an inset axis within `ax`; drawing proceeds back-to-front so the
    front card covers the older ones.
    """
    ax.axis('off')

    # Filter to non-None images, preserving t-1-first ordering
    valid = [(k, img) for k, img in enumerate(images) if img is not None]
    if not valid:
        _blank(ax)
        return

    n    = len(valid)
    step = 0.03                          # per-layer offset (axes fraction)
    card_w = 1.0 - step * (n - 1)
    card_h = 1.0 - step * (n - 1)

    # Iterate oldest → newest so newest is drawn last (on top)
    for rank in range(n - 1, -1, -1):
        _, img = valid[rank]
        # rank 0 = t-1 (front): dx=0, dy=0
        # rank n-1 = t-5 (back): dx=(n-1)*step, dy=(n-1)*step
        dx = rank * step
        dy = rank * step

        sub = ax.inset_axes([dx, dy, card_w, card_h])
        sub.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', aspect='auto')
        sub.set_xticks([]); sub.set_yticks([])

        is_front = (rank == 0)
        for sp in sub.spines.values():
            sp.set_edgecolor(border_color)
            sp.set_linewidth(2.5 if is_front else 0.8)


def draw_single(ax, image, cmap, vmin, vmax, border_color):
    """Draw a single 2-D array, or a grey placeholder if image is None."""
    if image is None:
        _blank(ax)
        return
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation='nearest', aspect='auto')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(border_color)
        sp.set_linewidth(2.5)
        sp.set_visible(True)


# ── Data collection ─────────────────────────────────────────────────────────

def get_test_sample(test_loader, model, bias, valid_mask):
    """
    Find the first valid (non-2020) sample in the test loader.
    Returns (inp_np [C,50,50], target_log [50,50], pred_display [50,50]).
    pred_display is bias-corrected, clipped, and re-logged for visual parity
    with target_log (both in log1p space).
    """
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets, temporal_mask, year_idx, spatial_mask, valid_year = batch
            for i in range(inputs.shape[0]):
                if valid_year[i] == 0:
                    continue

                pred = model(
                    inputs[i:i+1].to(DEVICE),
                    year_idx[i:i+1].to(DEVICE),
                    temporal_mask[i:i+1].to(DEVICE),
                    spatial_mask=spatial_mask[i:i+1].to(DEVICE),
                )
                pred_log = pred[0, 0].cpu().numpy()

                # Back-transform with bias correction, then re-log for display
                pred_display = np.log1p(
                    np.clip(np.exp(pred_log) * bias - 1.0, 0.0, None)
                )
                pred_display[~valid_mask] = 0.0

                target_log = targets[i, 0].numpy().copy()
                target_log[~valid_mask] = 0.0

                return inputs[i].numpy(), target_log, pred_display
    return None, None, None


def collect_yearly_aggregates(loaders, model, bias, valid_mask):
    """
    Run inference over all three splits and accumulate the total-abundance
    sum per (year_idx, bootstrap).

    Returns {year_idx: {'pred': [float, …], 'obs': [float, …]}}.
    """
    from collections import defaultdict
    agg = defaultdict(lambda: {'pred': [], 'obs': []})

    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                inputs, targets, temporal_mask, year_idx, spatial_mask, valid_year = batch
                preds = model(
                    inputs.to(DEVICE),
                    year_idx.to(DEVICE),
                    temporal_mask.to(DEVICE),
                    spatial_mask=spatial_mask.to(DEVICE),
                )
                for i in range(preds.shape[0]):
                    if valid_year[i] == 0:
                        continue
                    yr    = int(year_idx[i])
                    p_raw = np.clip(
                        np.exp(preds[i, 0].cpu().numpy()) * bias - 1.0, 0.0, None
                    )
                    t_raw = np.expm1(targets[i, 0].numpy())
                    p_raw[~valid_mask] = 0.0
                    t_raw[~valid_mask] = 0.0
                    agg[yr]['pred'].append(float(p_raw[valid_mask].sum()))
                    agg[yr]['obs'].append(float(t_raw[valid_mask].sum()))
    return agg


# ============================================================
#  MAIN FIGURE
# ============================================================

def make_figure(
    runs=None,
    save_path=None,
    title='CrabTransformer — Top Model Summary',
):
    """
    runs : list of (model_size, level, channel_cfg, pred_mode, criterion)
           Defaults to TARGET_RUNS defined at the top of the file.
    save_path : output PNG path.  Defaults to SAVE_DIR/top_model_figure.png.
    """
    if runs is None:
        runs = TARGET_RUNS

    n_rows = len(runs)
    n_cols = 7

    # Figure height: spatial rows + line-plot panel
    cell_h  = 2.8
    line_h  = 3.2
    fig_h   = cell_h * n_rows + line_h + 0.6   # 0.6 for suptitle gap
    cell_w  = 2.5
    fig_w   = cell_w * n_cols + 1.0            # 1.0 for row labels

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=130)

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[cell_h * n_rows, line_h],
        hspace=0.12,
    )
    grid_top = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols,
        subplot_spec=outer[0],
        hspace=0.12, wspace=0.07,
    )
    ax_line = fig.add_subplot(outer[1])

    # Spatial mask
    mask_path = os.path.join(REPO_DIR, 'data/real/output/spatial_mask.npy')
    valid_mask = (np.load(mask_path) > 0) if os.path.exists(mask_path) \
                 else np.ones((50, 50), dtype=bool)

    # Track per-run outputs for the line plot
    line_data   = {}   # row_idx → agg dict
    row_meta    = []   # (label, color, t_yr, v_yr, te_yr, data_type, lag)

    for row_idx, run_spec in enumerate(runs):
        model_size, level, channel_cfg, pred_mode, criterion = run_spec
        color = ROW_COLORS[row_idx % len(ROW_COLORS)]
        label = f"{channel_cfg} | {pred_mode} | {criterion} ({model_size})"

        print(f"\n[{row_idx+1}/{n_rows}]  {label}")

        try:
            (model, meta, bias,
             t_yr, v_yr, te_yr,
             tr_ld, va_ld, te_ld) = load_run(*run_spec)
        except Exception as exc:
            print(f"  ❌  Load failed: {exc}")
            for col in range(n_cols):
                ax = fig.add_subplot(grid_top[row_idx, col])
                _blank(ax, 'LOAD\nERROR')
            row_meta.append((label, color, 24, 8, 4, 'real', 0))
            continue

        data_type = 'real' if level == 'real' else 'dummy'
        row_meta.append((label, color, t_yr, v_yr, te_yr, data_type, meta['lag']))

        # ── Spatial sample ───────────────────────────────────────────
        print('  Finding representative test sample …')
        inp_np, tgt_log, pred_log = get_test_sample(te_ld, model, bias, valid_mask)

        if inp_np is None:
            print('  ⚠  No valid test sample — skipping spatial panels.')
            for col in range(n_cols):
                ax = fig.add_subplot(grid_top[row_idx, col])
                _blank(ax, 'NO DATA')
            continue

        ch = extract_channels(inp_np, meta)

        # ── Scale computations ───────────────────────────────────────
        def _stack(lst):
            return np.stack([x for x in lst if x is not None]) if lst else None

        sp_arr  = _stack((ch['sp_hist'] or []) + ([ch['sp_curr']] if ch['sp_curr'] is not None else []))
        tmp_arr = _stack((ch['temp_hist'] or []) + ([ch['temp_curr']] if ch['temp_curr'] is not None else []))
        rh_arr  = _stack(ch['rec_hist'] or [])
        rec_arr = np.stack([a for a in [tgt_log, pred_log] if a is not None])

        sp_vmin  = float(sp_arr.min())  if sp_arr  is not None else 0.0
        sp_vmax  = float(sp_arr.max())  if sp_arr  is not None else 8.0
        rh_vmin  = float(rh_arr.min())  if rh_arr  is not None else 0.0
        rh_vmax  = float(rh_arr.max())  if rh_arr  is not None else 8.0
        rec_vmin = float(rec_arr.min()) if rec_arr is not None else 0.0
        rec_vmax = float(rec_arr.max()) if rec_arr is not None else 8.0

        if tmp_arr is not None:
            tmp_abs  = max(abs(float(tmp_arr.min())), abs(float(tmp_arr.max()))) or 1.0
            tmp_vmin, tmp_vmax = -tmp_abs, tmp_abs
        else:
            tmp_vmin, tmp_vmax = -2.0, 2.0

        # ── Draw 7 columns ──────────────────────────────────────────
        # Col 0 — spawner history
        ax = fig.add_subplot(grid_top[row_idx, 0])
        if ch['sp_hist']:
            draw_stack(ax, ch['sp_hist'], CMAPS['spawner'], sp_vmin, sp_vmax, color)
        else:
            _blank(ax, 'no spawners')

        # Row label on leftmost axis
        ax.set_ylabel(label, fontsize=7, labelpad=6)

        # Col 1 — spawner current
        ax = fig.add_subplot(grid_top[row_idx, 1])
        draw_single(ax, ch['sp_curr'], CMAPS['spawner'], sp_vmin, sp_vmax, color)

        # Col 2 — temp history
        ax = fig.add_subplot(grid_top[row_idx, 2])
        if ch['temp_hist']:
            draw_stack(ax, ch['temp_hist'], CMAPS['temp'], tmp_vmin, tmp_vmax, color)
        else:
            _blank(ax, 'no temp')

        # Col 3 — temp current
        ax = fig.add_subplot(grid_top[row_idx, 3])
        draw_single(ax, ch['temp_curr'], CMAPS['temp'], tmp_vmin, tmp_vmax, color)

        # Col 4 — recruit history
        ax = fig.add_subplot(grid_top[row_idx, 4])
        if ch['rec_hist']:
            draw_stack(ax, ch['rec_hist'], CMAPS['recruit'], rh_vmin, rh_vmax, color)
        else:
            _blank(ax, 'no rec hist')

        # Col 5 — true recruit
        ax = fig.add_subplot(grid_top[row_idx, 5])
        draw_single(ax, tgt_log, CMAPS['recruit'], rec_vmin, rec_vmax, color)

        # Col 6 — predicted recruit
        ax = fig.add_subplot(grid_top[row_idx, 6])
        draw_single(ax, pred_log, CMAPS['recruit'], rec_vmin, rec_vmax, color)

        # Column titles on the first row only
        if row_idx == 0:
            for col, ttl in enumerate(COL_TITLES):
                sub = fig.add_subplot(grid_top[0, col])
                sub.set_title(ttl, fontsize=8, fontweight='bold', pad=5)

        # ── Yearly aggregates for line plot ──────────────────────────
        print('  Collecting yearly aggregates …')
        agg = collect_yearly_aggregates(
            [tr_ld, va_ld, te_ld], model, bias, valid_mask
        )
        line_data[row_idx] = agg

    # ── Line plot ────────────────────────────────────────────────────────────
    print('\nBuilding line plot …')
    for row_idx, (label, color, t_yr, v_yr, te_yr, data_type, lag) in enumerate(row_meta):
        if row_idx not in line_data:
            continue
        agg          = line_data[row_idx]
        years_sorted = sorted(agg.keys())

        pred_med = [np.median(agg[y]['pred']) for y in years_sorted]
        obs_med  = [np.median(agg[y]['obs'])  for y in years_sorted]
        pred_p25 = [np.percentile(agg[y]['pred'], 25) for y in years_sorted]
        pred_p75 = [np.percentile(agg[y]['pred'], 75) for y in years_sorted]

        ax_line.fill_between(years_sorted, pred_p25, pred_p75,
                             color=color, alpha=0.15)
        ax_line.plot(years_sorted, pred_med, '-',  color=color, lw=2.0,
                     label=f'Pred — {label}')
        ax_line.plot(years_sorted, obs_med,  '--', color=color, lw=1.5,
                     alpha=0.8, label=f'Obs — {label}')

    # Phase shading — use splits from first successful run
    if row_meta:
        _, _, t_yr, v_yr, te_yr, _, _ = row_meta[0]
        all_years = sorted({y for d in line_data.values() for y in d})
        if all_years:
            y0, y1 = all_years[0], all_years[-1]
            xform = ax_line.get_xaxis_transform()  # data-x, axes-y
            ax_line.axvspan(y0,            t_yr - 0.5,            color='seagreen',   alpha=0.07)
            ax_line.axvspan(t_yr - 0.5,    t_yr + v_yr - 0.5,    color='darkorange', alpha=0.07)
            ax_line.axvspan(t_yr + v_yr - 0.5, y1,               color='crimson',    alpha=0.07)
            ax_line.axvline(t_yr - 0.5,         color='grey', lw=1.0, ls=':')
            ax_line.axvline(t_yr + v_yr - 0.5,  color='grey', lw=1.0, ls=':')
            ax_line.text((y0 + t_yr) / 2,                      0.92, 'TRAIN',
                         transform=xform, ha='center', fontsize=8,
                         color='seagreen', fontweight='bold')
            ax_line.text((t_yr + t_yr + v_yr) / 2,            0.92, 'VAL',
                         transform=xform, ha='center', fontsize=8,
                         color='darkorange', fontweight='bold')
            ax_line.text((t_yr + v_yr + y1) / 2,              0.92, 'TEST',
                         transform=xform, ha='center', fontsize=8,
                         color='crimson', fontweight='bold')

    # Legend: one patch per run + solid/dashed key
    legend_handles = [
        mpatches.Patch(color=color, label=label[:55])
        for label, color, *_ in row_meta
    ]
    legend_handles += [
        plt.Line2D([0], [0], color='k', lw=2.0, ls='-',  label='Predicted (solid)'),
        plt.Line2D([0], [0], color='k', lw=1.5, ls='--', label='Observed (dashed)'),
    ]
    ax_line.legend(handles=legend_handles, fontsize=7, loc='upper left',
                   framealpha=0.85, ncol=min(2, len(runs) + 1))
    ax_line.set_xlabel('Year index', fontsize=9)
    ax_line.set_ylabel('Total recruit abundance', fontsize=9)
    ax_line.set_title(
        'Aggregate yearly recruits — median across bootstraps  '
        '(shaded band = IQR of predictions)',
        fontsize=8,
    )
    ax_line.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.003)

    out = save_path or os.path.join(SAVE_DIR, 'top_model_figure.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n✅  Figure saved → {out}')
    return out


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == '__main__':
    make_figure()
