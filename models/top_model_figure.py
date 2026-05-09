"""
top_model_figure.py
====================
Creates a 7-column spatial + line-plot summary figure for top CrabTransformer runs.

Row order
---------
  0  Climatological baseline  (lag-0)
  1  All channels | Now-cast | MSE | Base-size
  2  Temperature only | Now-cast | MSE | Reduced-size
  3  Recruits + Temp | One-year-ahead | MSE | Reduced-size
  --- section divider + new column headers ---
  4  Climatological baseline  (lag-5)
  5  Spawners + Temp | Lag-5 | MSE | Base-size

Column headers differ between sections:
  Lag-0 section : Spawner history (t-1,...,t-5) / Spawner current (t) / ...
  Lag-5 section : Spawner history (t-6,...,t-10) / Spawner current (t-5) / ...

Changes from previous version
------------------------------
- Row order updated as above
- temp_only now-cast uses MSE (not Tweedie)
- Section divider + second set of column headers for lag-5 rows
- Line plot legend: removed dashed-line entry
- Line plot title updated to include "Median across bootstrap"
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
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR   = '/content/Teleconnections-ViT'
DRIVE_BASE = '/content/drive/MyDrive/Teleconnection_ViT/model_outputs'
SAVE_DIR   = '/content/drive/MyDrive/Teleconnection_ViT/analysis'

MEMORY_YEARS    = 5
BATCH_SIZE      = 8
DATA_START_YEAR = 1988

sys.path.insert(0, REPO_DIR)
from models.model     import CrabTransformer
from data.data_helper import get_dataloaders

os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Row colours ───────────────────────────────────────────────────────────────
ROW_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    '#ff7f00', '#a65628', '#f781bf', '#555555',
]

# ── Runs to display ───────────────────────────────────────────────────────────
TARGET_RUNS = [
    ('BASELINE', 'real', 'climatological', 'normal',         'N/A'),  # lag-0 baseline
    ('normal',   'real', 'all',            'normal',         'MSE'),  # all channels now-cast
    ('small',    'real', 'temp_only',      'normal',         'MSE'),  # temp only now-cast MSE
    ('small',    'real', 'rec_temp',       'one_year_ahead', 'MSE'),  # rec+temp 1yr-ahead
    ('BASELINE', 'real', 'climatological', 'lag5',           'N/A'),  # lag-5 baseline
    ('normal',   'real', 'sp_temp',        'lag5',           'MSE'),  # sp+temp lag-5
]

# First row index of the lag-5 section (section divider drawn above this row)
LAG5_SECTION_START = 4

RUN_DISPLAY_NAMES = {
    ('BASELINE', 'real', 'climatological', 'normal',         'N/A'): 'Climatological baseline | Now-cast + 1-yr ahead',
    ('normal',   'real', 'all',            'normal',         'MSE'): 'All channels | Now-cast | MSE | Base-size',
    ('small',    'real', 'temp_only',      'normal',         'MSE'): 'Bottom Temp only | Now-cast | MSE | Reduced-size',
    ('small',    'real', 'rec_temp',       'one_year_ahead', 'MSE'): 'Recruits + Bottom Temp | 1-yr ahead | MSE | Reduced-size',
    ('BASELINE', 'real', 'climatological', 'lag5',           'N/A'): 'Climatological baseline | Lag-5',
    ('normal',   'real', 'sp_temp',        'lag5',           'MSE'): 'Spawners + Bottom Temp | Lag-5 | MSE | Base-size',
}

# Column titles for lag-0 section
COL_TITLES_LAG0 = [
    'Spawner history\n(t-1,...,t-5)',
    'Spawner\ncurrent (t)',
    'Bottom Temperature\nhistory (t-1,...,t-5)',
    'Bottom Temperature\ncurrent (t)',
    'Recruit history\n(t-1,...,t-5)',
    'Observed\nrecruit',
    'Predicted\nrecruit',
]

# Column titles for lag-5 section
COL_TITLES_LAG5 = [
    'Spawner history\n(t-6,...,t-10)',
    'Spawner\ncurrent (t-5)',
    'Bottom Temperature\nhistory (t-6,...,t-10)',
    'Bottom Temperature\ncurrent (t-5)',
    'Recruit history\n(t-1,...,t-5)',
    'Observed\nrecruit',
    'Predicted\nrecruit',
]

CMAPS = {'spawner': 'YlOrRd', 'temp': 'RdBu_r', 'recruit': 'Blues'}


# ============================================================
#  PIPELINE HELPERS
# ============================================================

def get_year_splits(data_type, lag):
    if data_type == 'real':
        return (24, 8, 4) if lag == 0 else (21, 6, 4)
    return (18, 9, 3)


def load_run(model_size, level, channel_cfg, pred_mode, criterion):
    data_type = 'real' if level == 'real' else 'dummy'
    run_dir   = os.path.join(DRIVE_BASE, model_size, level,
                             channel_cfg, pred_mode, criterion)

    with open(os.path.join(run_dir, 'training_history.json')) as f:
        hist = json.load(f)

    meta = hist['channel_cfg_meta']
    bias = hist.get('bias_correction', 1.0)
    t_yr, v_yr, te_yr = get_year_splits(data_type, meta['lag'])

    tr_ld, va_ld, te_ld = get_dataloaders(
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
    model.load_state_dict(
        torch.load(os.path.join(run_dir, 'best_model.pt'), map_location=DEVICE)
    )
    model.eval()

    return model, meta, bias, t_yr, v_yr, te_yr, tr_ld, va_ld, te_ld


def extract_channels(inp, meta):
    ch   = {k: None for k in ('sp_curr', 'sp_hist',
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


def get_test_sample(test_loader, model, bias, valid_mask, target_year_idx=None):
    best_yr   = -1
    best_data = (None, None, None)

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets, temporal_mask, year_idx, spatial_mask, valid_year = batch
            for i in range(inputs.shape[0]):
                if valid_year[i] == 0:
                    continue
                yi = int(year_idx[i])
                if target_year_idx is not None and yi != target_year_idx:
                    continue

                pred = model(
                    inputs[i:i+1].to(DEVICE),
                    year_idx[i:i+1].to(DEVICE),
                    temporal_mask[i:i+1].to(DEVICE),
                    spatial_mask=spatial_mask[i:i+1].to(DEVICE),
                )
                pred_log     = pred[0, 0].cpu().numpy()
                pred_display = np.log1p(
                    np.clip(np.exp(pred_log) * bias - 1.0, 0.0, None)
                )
                pred_display[~valid_mask] = 0.0
                target_log = targets[i, 0].numpy().copy()
                target_log[~valid_mask] = 0.0
                print(f'    Sample year_idx={yi}  '
                      f'inp mean={inputs[i].mean():.3f}  '
                      f'tgt mean={target_log[valid_mask].mean():.3f}  '
                      f'pred mean={pred_display[valid_mask].mean():.3f}')

                if target_year_idx is not None:
                    return inputs[i].numpy(), target_log, pred_display

                if yi > best_yr:
                    best_yr   = yi
                    best_data = (inputs[i].numpy(), target_log, pred_display)

    return best_data


def collect_yearly_aggregates(loaders, model, bias, valid_mask):
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
#  DRAWING HELPERS
# ============================================================

def _cell_pos(fig, gs, row, col):
    ax  = fig.add_subplot(gs[row, col])
    pos = ax.get_position()
    fig.delaxes(ax)
    return pos


def _blank_ax(fig, bbox, msg='N/A'):
    ax = fig.add_axes([bbox.x0, bbox.y0, bbox.width, bbox.height])
    ax.set_facecolor('#e8e8e8')
    ax.text(0.5, 0.5, msg, ha='center', va='center',
            transform=ax.transAxes, fontsize=11, color='#000000', style='italic')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def _make_cmap(name):
    import copy
    cmap = copy.copy(plt.get_cmap(name))
    cmap.set_bad('black')
    return cmap


def _masked(image, valid_mask):
    if valid_mask is None:
        return image
    return np.ma.array(image, mask=~valid_mask)


def draw_single_ax(fig, bbox, image, cmap, vmin, vmax, border_color,
                   valid_mask=None):
    if image is None:
        _blank_ax(fig, bbox)
        return
    ax = fig.add_axes([bbox.x0, bbox.y0, bbox.width, bbox.height])
    ax.set_facecolor('black')
    ax.imshow(_masked(image, valid_mask), cmap=_make_cmap(cmap),
              vmin=vmin, vmax=vmax, interpolation='nearest', aspect='auto')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(border_color)
        sp.set_linewidth(2.5)
        sp.set_visible(True)


def draw_stack_ax(fig, bbox, images, cmap, vmin, vmax, border_color,
                  valid_mask=None):
    valid = [(k, img) for k, img in enumerate(images) if img is not None]
    if not valid:
        _blank_ax(fig, bbox)
        return

    n      = len(valid)
    step_x = bbox.width  * 0.06
    step_y = bbox.height * 0.06
    card_w = bbox.width  - (n - 1) * step_x
    card_h = bbox.height - (n - 1) * step_y
    cmap_obj = _make_cmap(cmap)

    for rank in range(n - 1, -1, -1):
        _, img  = valid[rank]
        offset  = (n - 1 - rank)
        x0 = bbox.x0 + offset * step_x
        y0 = bbox.y0 + offset * step_y

        ax = fig.add_axes([x0, y0, card_w, card_h])
        ax.set_facecolor('black')
        ax.imshow(_masked(img, valid_mask), cmap=cmap_obj,
                  vmin=vmin, vmax=vmax, interpolation='nearest', aspect='auto')
        ax.set_xticks([]); ax.set_yticks([])

        is_front = (rank == 0)
        for sp in ax.spines.values():
            sp.set_edgecolor(border_color)
            sp.set_linewidth(2.5 if is_front else 0.8)
            sp.set_visible(True)


def _load_baseline_grid(lag: int):
    for search_dir in [SAVE_DIR, '.']:
        path = os.path.join(search_dir, f'baseline_mean_grid_lag{lag}.npy')
        if os.path.exists(path):
            return np.load(path).astype(np.float32)
    print(f'  Warning: baseline_mean_grid_lag{lag}.npy not found.')
    return None


def _load_baseline_obs_mean(lag: int):
    for search_dir in [SAVE_DIR, '.']:
        path = os.path.join(search_dir, f'baseline_obs_mean_lag{lag}.npy')
        if os.path.exists(path):
            return np.load(path).astype(np.float32)
    return None


def _get_baseline_obs_single(lag, target_year_idx, valid_mask):
    """Load a single-year single-bootstrap observed target for a baseline row."""
    obs_single = None
    try:
        t_yr_b, v_yr_b, te_yr_b = get_year_splits('real', lag)
        _, _, te_ld_base = get_dataloaders(
            batch_size=BATCH_SIZE, memory_years=MEMORY_YEARS,
            train_years=t_yr_b, val_years=v_yr_b, test_years=te_yr_b,
            level='real', data_type='real',
            include_current_spawner=True,
            lag=lag,
            use_temp=True,
            use_spawners=True,
            use_recruits=True,
        )
        req_yr  = (target_year_idx - lag if target_year_idx is not None else None)
        best_yr = -1
        with torch.no_grad():
            for batch in te_ld_base:
                _, targets, _, year_idx, spatial_mask_b, valid_year = batch
                for i in range(targets.shape[0]):
                    if valid_year[i] == 0:
                        continue
                    yi = int(year_idx[i])
                    if req_yr is not None and yi != req_yr:
                        continue
                    if req_yr is None and yi <= best_yr:
                        continue
                    tgt = targets[i, 0].numpy().copy()
                    tgt[~valid_mask] = 0.0
                    obs_single = tgt
                    best_yr    = yi
                    if req_yr is not None:
                        break
                if req_yr is not None and obs_single is not None:
                    break
    except Exception as e:
        print(f'  Warning: Could not load baseline obs target: {e}')
    return obs_single


# ============================================================
#  MAIN FIGURE
# ============================================================

def make_figure(runs=None, save_path=None, display_year=None,
                title='CrabTransformer — Top Model Summary'):
    if runs is None:
        runs = TARGET_RUNS

    target_year_idx = None
    if display_year is not None:
        target_year_idx = display_year - DATA_START_YEAR
        title = f'{title} ({display_year})'

    n_rows = len(runs)
    n_cols = 7

    # ── Font sizes ────────────────────────────────────────────────────────────
    FS_SUPTITLE    = 18
    FS_SECTION_HDR = 16
    FS_COL_HDR     = 14
    FS_ROW_LABEL   = 14
    FS_PHASE_LABEL = 14
    FS_AXIS_LABEL  = 14
    FS_AXIS_TITLE  = 14
    FS_LEGEND      = 14
    FS_DIVIDER     = 11   # section divider label font size

    # ── Figure size ───────────────────────────────────────────────────────────
    cell_w   = 2.4
    cell_h   = 2.6
    label_w  = 2.0
    line_h   = 5.2
    top_pad  = 0.55
    spacer_w = 0.30
    fig_w    = label_w + cell_w * n_cols + cell_w * spacer_w * 2
    fig_h    = top_pad + cell_h * n_rows + cell_h * 0.45 + line_h + 0.4

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=130)

    left_margin = label_w / fig_w

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[top_pad + cell_h * n_rows, line_h],
        hspace=0.08,
        top=0.87, bottom=0.10,
        left=left_margin, right=0.99,
    )

    def _gcol(c):
        if c < 5:  return c
        if c == 5: return c + 1
        return c + 2

    # The gridspec has n_rows + 1 rows: a narrow gap row is inserted at
    # LAG5_SECTION_START to provide physical space for the second column headers.
    # GAP_H controls how tall the gap row is relative to a data row.
    GAP_H       = 0.45
    n_gs_rows   = n_rows + 1   # +1 for the gap row
    gap_gs_row  = LAG5_SECTION_START   # gridspec row index of the gap

    # Build height_ratios: 1.0 for data rows, GAP_H for the gap row
    height_ratios = []
    for gs_r in range(n_gs_rows):
        height_ratios.append(GAP_H if gs_r == gap_gs_row else 1.0)

    grid_top = gridspec.GridSpecFromSubplotSpec(
        n_gs_rows, n_cols + 2,
        subplot_spec=outer[0],
        hspace=0.10, wspace=0.06,
        width_ratios=[1, 1, 1, 1, 1, spacer_w, 1, spacer_w, 1],
        height_ratios=height_ratios,
    )
    ax_line = fig.add_subplot(outer[1])

    # Map data row index → gridspec row index (rows at or after the gap shift by 1)
    def _grows(data_row):
        return data_row if data_row < LAG5_SECTION_START else data_row + 1

    # ── Spatial mask ──────────────────────────────────────────────────────────
    mask_path  = os.path.join(REPO_DIR, 'data/real/output/spatial_mask.npy')
    valid_mask = (np.load(mask_path) > 0) if os.path.exists(mask_path) \
                 else np.ones((50, 50), dtype=bool)

    # ── Pre-compute cell positions using mapped gridspec rows ─────────────────
    cell_bboxes = {}
    for r in range(n_rows):
        for c in range(n_cols):
            cell_bboxes[(r, c)] = _cell_pos(fig, grid_top, _grows(r), _gcol(c))

    # Also get the bbox of the gap row itself (for placing lag-5 column headers)
    gap_bboxes = {}
    for c in range(n_cols):
        gap_bboxes[c] = _cell_pos(fig, grid_top, gap_gs_row, _gcol(c))

    # ── Header geometry ───────────────────────────────────────────────────────
    top_row_top = cell_bboxes[(0, 0)].y1
    fig_hdr_top = 0.96
    hdr_space   = fig_hdr_top - top_row_top

    col_hdr_y = top_row_top + hdr_space * 0.28
    sec_hdr_y = top_row_top + hdr_space * 0.78
    rule_y    = top_row_top + hdr_space * 0.53

    inputs_x0 = cell_bboxes[(0, 0)].x0
    inputs_x1 = cell_bboxes[(0, 4)].x1
    output_x0 = cell_bboxes[(0, 5)].x0
    output_x1 = cell_bboxes[(0, 5)].x1
    target_x0 = cell_bboxes[(0, 6)].x0
    target_x1 = cell_bboxes[(0, 6)].x1
    inputs_cx  = (inputs_x0 + inputs_x1) / 2
    output_cx  = (output_x0 + output_x1) / 2
    target_cx  = (target_x0 + target_x1) / 2
    div_x      = (cell_bboxes[(0, 4)].x1 + cell_bboxes[(0, 5)].x0) / 2
    div2_x     = (cell_bboxes[(0, 5)].x1 + cell_bboxes[(0, 6)].x0) / 2
    grid_bot   = cell_bboxes[(n_rows - 1, 0)].y0

    # ── Column headers — lag-0 section (top) ─────────────────────────────────
    for col, ttl in enumerate(COL_TITLES_LAG0):
        bbox = cell_bboxes[(0, col)]
        fig.text(
            bbox.x0 + bbox.width / 2, col_hdr_y, ttl,
            ha='center', va='center',
            fontsize=FS_COL_HDR, fontweight='bold',
            transform=fig.transFigure,
        )

    # ── Column headers — lag-5 section ───────────────────────────────────────
    # Centred vertically inside the gap row
    for col, ttl in enumerate(COL_TITLES_LAG5):
        bbox = gap_bboxes[col]
        gap_cy = bbox.y0 + bbox.height / 2
        fig.text(
            bbox.x0 + bbox.width / 2, gap_cy, ttl,
            ha='center', va='center',
            fontsize=FS_COL_HDR, fontweight='bold',
            transform=fig.transFigure,
        )

    # ── Per-run processing ────────────────────────────────────────────────────
    line_data = {}
    row_meta  = []

    for row_idx, run_spec in enumerate(runs):
        model_size, level, channel_cfg, pred_mode, criterion = run_spec
        color = ROW_COLORS[row_idx % len(ROW_COLORS)]
        label = RUN_DISPLAY_NAMES.get(
            run_spec,
            f"{channel_cfg} | {pred_mode} | {criterion} ({model_size})",
        )
        print(f'\n[{row_idx+1}/{n_rows}]  {label}')

        # Row label on left margin — wrap on ' | ' so each component is its own line
        bbox0  = cell_bboxes[(row_idx, 0)]
        row_cy = bbox0.y0 + bbox0.height / 2
        label_wrapped = label.replace(' | ', '\n')
        fig.text(
            left_margin - 0.01, row_cy, label_wrapped,
            ha='right', va='center', fontsize=FS_ROW_LABEL,
            transform=fig.transFigure, color=color, fontweight='bold',
            linespacing=1.3,
        )

        # ── BASELINE ROW ──────────────────────────────────────────────────────
        if model_size == 'BASELINE':
            lag = 5 if pred_mode == 'lag5' else 0

            for c in range(5):
                _blank_ax(fig, cell_bboxes[(row_idx, c)], '')

            baseline_grid = _load_baseline_grid(lag)
            obs_mean_grid = _load_baseline_obs_mean(lag)

            if baseline_grid is not None:
                # Col 5 — single-year observed target matching display_year
                obs_single = _get_baseline_obs_single(lag, target_year_idx, valid_mask)

                if obs_single is not None:
                    rec_vmin = float(obs_single[valid_mask].min())
                    rec_vmax = float(obs_single[valid_mask].max())
                    draw_single_ax(fig, cell_bboxes[(row_idx, 5)],
                                   obs_single, CMAPS['recruit'],
                                   rec_vmin, rec_vmax, color, valid_mask)
                elif obs_mean_grid is not None:
                    rec_vmin = float(np.log1p(np.maximum(0, obs_mean_grid[valid_mask])).min())
                    rec_vmax = float(np.log1p(obs_mean_grid[valid_mask]).max())
                    draw_single_ax(fig, cell_bboxes[(row_idx, 5)],
                                   np.log1p(np.maximum(0, obs_mean_grid)),
                                   CMAPS['recruit'], rec_vmin, rec_vmax,
                                   color, valid_mask)
                else:
                    _blank_ax(fig, cell_bboxes[(row_idx, 5)], 'obs target')

                # Col 6 — baseline mean grid (log1p for display)
                pred_log = np.log1p(np.maximum(0, baseline_grid))
                rec_vmin = float(pred_log[valid_mask].min())
                rec_vmax = float(pred_log[valid_mask].max())
                draw_single_ax(fig, cell_bboxes[(row_idx, 6)],
                               pred_log, CMAPS['recruit'],
                               rec_vmin, rec_vmax, color, valid_mask)

                # Line data — flat predicted total across full year range
                pred_total = float(baseline_grid[valid_mask].sum())
                all_year_idxs = []
                for other_idx, other_agg in line_data.items():
                    if runs[other_idx][0] != 'BASELINE':
                        other_lag = row_meta[other_idx][5] if other_idx < len(row_meta) else 0
                        if other_lag == lag:
                            all_year_idxs = sorted(other_agg.keys())
                            break
                if not all_year_idxs:
                    n_years_total = (21 + 6 + 4) if lag == 5 else (24 + 8 + 4)
                    all_year_idxs = list(range(n_years_total))

                line_data[row_idx] = {
                    yi: {'pred': [pred_total], 'obs': [0.0]}
                    for yi in all_year_idxs
                }
            else:
                _blank_ax(fig, cell_bboxes[(row_idx, 5)], 'run baseline_evaluation.py')
                _blank_ax(fig, cell_bboxes[(row_idx, 6)], 'run baseline_evaluation.py')

            t_yr  = 21 if lag == 5 else 24
            v_yr  = 6  if lag == 5 else 8
            te_yr = 4
            row_meta.append((label, color, t_yr, v_yr, te_yr, lag))
            continue
        # ── END BASELINE ROW ──────────────────────────────────────────────────

        # ── MODEL ROW ─────────────────────────────────────────────────────────
        try:
            (model, meta, bias, t_yr, v_yr, te_yr,
             tr_ld, va_ld, te_ld) = load_run(*run_spec)
        except Exception as exc:
            print(f'  Load failed: {exc}')
            for c in range(n_cols):
                _blank_ax(fig, cell_bboxes[(row_idx, c)], 'LOAD ERR')
            row_meta.append((label, color, 24, 8, 4, 0))
            continue

        row_meta.append((label, color, t_yr, v_yr, te_yr, meta.get('lag', 0)))

        run_lag      = meta.get('lag', 0)
        eff_year_idx = (target_year_idx - run_lag
                        if target_year_idx is not None else None)
        print('  Finding representative test sample ...')
        inp_np, tgt_log, pred_log = get_test_sample(
            te_ld, model, bias, valid_mask,
            target_year_idx=eff_year_idx,
        )

        if inp_np is None:
            print('  No valid test sample found; drawing blanks.')
            for c in range(n_cols):
                _blank_ax(fig, cell_bboxes[(row_idx, c)], 'NO DATA')
        else:
            ch = extract_channels(inp_np, meta)

            def arr_of(lst, single=None):
                items = list(lst or []) + ([single] if single is not None else [])
                items = [x for x in items if x is not None]
                return np.stack(items) if items else None

            def safe_range(arr, symmetric=False):
                if arr is None:
                    return (0.0, 8.0)
                lo, hi = float(arr.min()), float(arr.max())
                if symmetric:
                    ab = max(abs(lo), abs(hi)) or 1.0
                    return (-ab, ab)
                return (lo, hi)

            sp_arr  = arr_of(ch['sp_hist'],  ch['sp_curr'])
            rh_arr  = arr_of(ch['rec_hist'])
            tmp_arr = arr_of(ch['temp_hist'], ch['temp_curr'])
            rec_arr = arr_of([tgt_log, pred_log])

            sp_vmin,  sp_vmax  = safe_range(sp_arr)
            rh_vmin,  rh_vmax  = safe_range(rh_arr)
            tmp_vmin, tmp_vmax = safe_range(tmp_arr, symmetric=True)
            rec_vmin, rec_vmax = safe_range(rec_arr)

            # Col 0 — spawner history
            if ch['sp_hist']:
                draw_stack_ax(fig, cell_bboxes[(row_idx, 0)],
                              ch['sp_hist'], CMAPS['spawner'],
                              sp_vmin, sp_vmax, color, valid_mask)
            else:
                _blank_ax(fig, cell_bboxes[(row_idx, 0)], 'not used')

            # Col 1 — spawner current
            if ch['sp_curr'] is not None:
                draw_single_ax(fig, cell_bboxes[(row_idx, 1)],
                               ch['sp_curr'], CMAPS['spawner'],
                               sp_vmin, sp_vmax, color, valid_mask)
            elif pred_mode == 'normal':
                _blank_ax(fig, cell_bboxes[(row_idx, 1)], 'not used')
            else:
                _blank_ax(fig, cell_bboxes[(row_idx, 1)], 'excluded\n(forecast mode)')

            # Col 2 — temp history
            if ch['temp_hist']:
                draw_stack_ax(fig, cell_bboxes[(row_idx, 2)],
                              ch['temp_hist'], CMAPS['temp'],
                              tmp_vmin, tmp_vmax, color, valid_mask)
            else:
                _blank_ax(fig, cell_bboxes[(row_idx, 2)], 'not used')

            # Col 3 — temp current
            if ch['temp_curr'] is not None:
                draw_single_ax(fig, cell_bboxes[(row_idx, 3)],
                               ch['temp_curr'], CMAPS['temp'],
                               tmp_vmin, tmp_vmax, color, valid_mask)
            elif pred_mode == 'one_year_ahead':
                _blank_ax(fig, cell_bboxes[(row_idx, 3)], 'excluded\n(forecast mode)')
            else:
                _blank_ax(fig, cell_bboxes[(row_idx, 3)], 'not used')

            # Col 4 — recruit history
            if ch['rec_hist']:
                draw_stack_ax(fig, cell_bboxes[(row_idx, 4)],
                              ch['rec_hist'], CMAPS['recruit'],
                              rh_vmin, rh_vmax, color, valid_mask)
            else:
                _blank_ax(fig, cell_bboxes[(row_idx, 4)], 'not used')

            # Col 5 — observed target
            draw_single_ax(fig, cell_bboxes[(row_idx, 5)],
                           tgt_log, CMAPS['recruit'],
                           rec_vmin, rec_vmax, color, valid_mask)

            # Col 6 — predicted
            draw_single_ax(fig, cell_bboxes[(row_idx, 6)],
                           pred_log, CMAPS['recruit'],
                           rec_vmin, rec_vmax, color, valid_mask)

        print('  Collecting yearly aggregates ...')
        agg = collect_yearly_aggregates(
            [tr_ld, va_ld, te_ld], model, bias, valid_mask
        )
        line_data[row_idx] = agg

    # ── Section divider — drawn as top border of the gap row ─────────────────
    if LAG5_SECTION_START < n_rows:
        divider_y = gap_bboxes[0].y1   # top edge of the gap row

        fig.add_artist(Line2D(
            [inputs_x0, target_x1], [divider_y, divider_y],
            transform=fig.transFigure,
            color='#333333', lw=2.0, ls='-',
        ))
        

    # ── Section headers + vertical dividers ───────────────────────────────────
    _wbg = dict(facecolor='white', edgecolor='none', alpha=0.90, pad=2)

    for cx, lbl in [(inputs_cx, 'Inputs'),
                    (output_cx, 'Target'),
                    (target_cx, 'Output')]:
        fig.text(cx, sec_hdr_y, lbl,
                 ha='center', va='center', fontsize=FS_SECTION_HDR,
                 fontweight='bold', transform=fig.transFigure,
                 color='#111111', bbox=_wbg)

    for x0, x1 in [(inputs_x0, inputs_x1),
                   (output_x0, output_x1),
                   (target_x0, target_x1)]:
        fig.add_artist(Line2D([x0, x1], [rule_y, rule_y],
                              transform=fig.transFigure,
                              color='#aaaaaa', lw=1.0))

    for dx in [div_x, div2_x]:
        fig.add_artist(Line2D([dx, dx], [grid_bot, fig_hdr_top],
                              transform=fig.transFigure,
                              color='#666666', lw=1.5, ls='--'))

    # ── Line plot ──────────────────────────────────────────────────────────────
    print('\nBuilding line plot ...')
    obs_plotted = False
    for row_idx, (label, color, t_yr, v_yr, te_yr, lag) in enumerate(row_meta):
        if row_idx not in line_data:
            continue
        agg          = line_data[row_idx]
        years_sorted = sorted(agg.keys())
        years_plot   = [y + lag + DATA_START_YEAR for y in years_sorted]
        pred_med     = [np.median(agg[y]['pred']) for y in years_sorted]
        obs_med      = [np.median(agg[y]['obs'])  for y in years_sorted]

        is_baseline = (runs[row_idx][0] == 'BASELINE')

        if not is_baseline:
            pred_p25 = [np.percentile(agg[y]['pred'], 25) for y in years_sorted]
            pred_p75 = [np.percentile(agg[y]['pred'], 75) for y in years_sorted]
            ax_line.fill_between(years_plot, pred_p25, pred_p75,
                                 color=color, alpha=0.15)

        ax_line.plot(years_plot, pred_med, '-', color=color, lw=2.0,
                     label=f'{"Baseline" if is_baseline else "Pred"} — {label}')

        if not is_baseline and lag == 0 and not obs_plotted:
            ax_line.scatter(years_plot, obs_med, color='black', s=18, zorder=5,
                            alpha=0.85, label='Observed')
            obs_plotted = True

    # Phase shading
    if row_meta and line_data:
        _, _, t_yr, v_yr, te_yr, _ = row_meta[0]
        all_years = sorted({y for d in line_data.values() for y in d})
        if all_years:
            y0_cal        = all_years[0]  + DATA_START_YEAR
            y1_cal        = all_years[-1] + DATA_START_YEAR
            train_end_cal = t_yr          - 0.5 + DATA_START_YEAR
            val_end_cal   = t_yr + v_yr   - 0.5 + DATA_START_YEAR
            xform = ax_line.get_xaxis_transform()
            ax_line.axvspan(y0_cal,        train_end_cal, color='seagreen',   alpha=0.07)
            ax_line.axvspan(train_end_cal, val_end_cal,   color='darkorange', alpha=0.07)
            ax_line.axvspan(val_end_cal,   y1_cal,        color='crimson',    alpha=0.07)
            ax_line.axvline(train_end_cal, color='grey', lw=1.0, ls=':')
            ax_line.axvline(val_end_cal,   color='grey', lw=1.0, ls=':')
            ax_line.text((y0_cal + train_end_cal) / 2,      0.93, 'TRAIN',
                         transform=xform, ha='center', fontsize=FS_PHASE_LABEL,
                         color='seagreen', fontweight='bold')
            ax_line.text((train_end_cal + val_end_cal) / 2,  0.93, 'VALIDATION',
                         transform=xform, ha='center', fontsize=FS_PHASE_LABEL,
                         color='darkorange', fontweight='bold')
            ax_line.text((val_end_cal + y1_cal) / 2,         0.93, 'TEST',
                         transform=xform, ha='center', fontsize=FS_PHASE_LABEL,
                         color='crimson', fontweight='bold')

    # Legend — solid colour patches only, no dashed line entry
    # Strip newlines from labels so they appear on a single line in the legend
    legend_handles = [
        mpatches.Patch(color=color, label=label.replace('\n', ' ').strip()[:70])
        for label, color, *_ in row_meta
    ]
    if obs_plotted:
        legend_handles.append(
            plt.Line2D([0], [0], color='black', marker='o', ms=4,
                       lw=0, label='Observed')
        )
    ax_line.legend(handles=legend_handles, fontsize=FS_LEGEND,
                   loc='upper center', bbox_to_anchor=(0.5, -0.18),
                   framealpha=0.85, ncol=2, borderaxespad=0.)

    ax_line.set_xlabel('Year', fontsize=FS_AXIS_LABEL)
    ax_line.set_ylabel('Total recruit abundance (density)', fontsize=FS_AXIS_LABEL)
    ax_line.set_title(
        'Median Across Bootstrap Aggregate Yearly Recruits',
        fontsize=FS_AXIS_TITLE, fontweight='bold',
    )
    ax_line.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=FS_SUPTITLE, fontweight='bold', y=0.995)

    out = save_path or os.path.join(SAVE_DIR, 'top_model_figure.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved -> {out}')
    return out


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == '__main__':
    make_figure(display_year=2023)