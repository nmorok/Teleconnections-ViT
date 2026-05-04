"""
top_model_figure.py
====================
Creates a 7-column spatial + line-plot summary figure for top CrabTransformer runs.

Column layout (left → right)
------------------------------
  0  Spawner history   — stacked cards: t-1 (front) … t-5 (back)
  1  Spawner current   — t, or blank if config excludes it
  2  Temp history      — stacked cards: t-1 … t-5
  3  Temp current      — t, or blank if config excludes it
  4  Recruit history   — stacked cards: t-1 … t-5
  5  True recruit      — target for the displayed test sample
  6  Predicted recruit — model output for the same sample

Rows: one per entry in TARGET_RUNS, each with a unique border colour that
also determines the line colour in the aggregate-recruit plot below.

Usage
-----
  Adjust REPO_DIR / DRIVE_BASE / SAVE_DIR / TARGET_RUNS at the top, then:

      python models/top_model_figure.py

  Or call make_figure() after importing in a Colab cell.
  GPU is used if available; works on CPU too.
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

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR   = '/content/Teleconnections-ViT'
DRIVE_BASE = '/content/drive/MyDrive/Teleconnection_ViT/model_outputs'
SAVE_DIR   = '/content/drive/MyDrive/Teleconnection_ViT/analysis'

MEMORY_YEARS    = 5
BATCH_SIZE      = 8
DATA_START_YEAR = 1988   # year_idx 0 = 1988  (derived: mask[32]=2020 → 2020-32=1988)

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
# (model_size, level, channel_cfg, pred_mode, criterion)
TARGET_RUNS = [
    ('normal', 'real', 'all', 'normal', 'MSE'),
    ('small', 'real', 'temp_only', 'normal', 'Tweedie'),
    ('small', 'real', 'rec_temp', 'one_year_ahead', 'MSE'),
    ('normal', 'real', 'sp_temp', 'lag5', 'MSE')
]

# ── Run display names ─────────────────────────────────────────────────────────
# Maps (model_size, level, channel_cfg, pred_mode, criterion) → display label.
# Edit entries here to control how each run appears in the row labels and legend.
# If a run spec is not listed, the label is auto-generated from the run tuple.
RUN_DISPLAY_NAMES = {
    ('normal', 'real', 'all',        'normal',        'MSE'):     'All channels | Now-cast | MSE | Base-size',
    ('small',  'real', 'temp_only',  'normal',        'Tweedie'): 'Bottom Temp only | Now-cast | Tweedie | Reduced-size',
    ('small',  'real', 'rec_temp',   'one_year_ahead','MSE'):     'Recruits + Bottom Temp | 1-yr ahead | MSE | Reduced-size',
    ('normal', 'real', 'sp_temp',    'lag5',          'MSE'):     'Spawners + Bottom Temp | Lag-5 | MSE | Base-size',
}

COL_TITLES = [
    'Spawner history\n(t-1 … t-5)',
    'Spawner\ncurrent (t)',
    'Bottom Temperature\nhistory (t-1 … t-5)',
    'Bottom Temperature\ncurrent (t)',
    'Recruit history\n(t-1 … t-5)',
    'Observed recruit',   # col 5 — observed target
    'Predicted recruit', 
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
    """Load checkpoint + metadata; build model and data loaders."""
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
    """
    Split a [C, 50, 50] input array into named channel groups.

    images[0] in each list = t-1 (most recent); images[-1] = t-5 (oldest).

    Returns a dict with keys:
      sp_hist, sp_curr, temp_hist, temp_curr, rec_hist
    Each is a list of arrays or a single array or None.
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


def get_test_sample(test_loader, model, bias, valid_mask, target_year_idx=None):
    """
    Return (inp_np, target_log, pred_display) for a test sample.

    If target_year_idx is given, returns that specific year (or None if absent).
    Otherwise returns the sample with the *highest* valid year_idx (most recent),
    avoiding the edge case where the earliest test year falls in the masked 2020 gap
    and its t-1 history channel (also 2020) appears blank.
    """
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
                    # Exact match requested — return immediately
                    return inputs[i].numpy(), target_log, pred_display

                if yi > best_yr:
                    best_yr   = yi
                    best_data = (inputs[i].numpy(), target_log, pred_display)

    return best_data


def collect_yearly_aggregates(loaders, model, bias, valid_mask):
    """Total recruit abundance per (year, bootstrap) across all splits."""
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
#  DRAWING  —  figure-level axes via fig.add_axes()
# ============================================================
#
# All spatial axes are created with fig.add_axes([x0, y0, w, h]) in
# figure coordinates (0→1).  This avoids the ax.inset_axes() / gridspec
# re-use issues that caused blank panels in previous versions.
# Positions are obtained from temporary placeholder axes, which are
# then removed before content is drawn.

def _cell_pos(fig, gs, row, col):
    """
    Return the figure-level Bbox for gridspec cell (row, col).
    A temporary axes is added, its position is read, then it is removed.
    """
    ax  = fig.add_subplot(gs[row, col])
    pos = ax.get_position()     # Bbox in figure coords
    fig.delaxes(ax)
    return pos


def _blank_ax(fig, bbox, msg='N/A'):
    """Add a grey placeholder axes at the given Bbox."""
    ax = fig.add_axes([bbox.x0, bbox.y0, bbox.width, bbox.height])
    ax.set_facecolor('#e8e8e8')
    ax.text(0.5, 0.5, msg, ha='center', va='center',
            transform=ax.transAxes, fontsize=11, color="#000000", style='italic')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def _make_cmap(name):
    """Return a copy of the named colormap with bad (masked) values set to black."""
    import copy
    cmap = copy.copy(plt.get_cmap(name))
    cmap.set_bad('black')
    return cmap


def _masked(image, valid_mask):
    """Return a masked array with land cells (outside valid_mask) hidden."""
    if valid_mask is None:
        return image
    return np.ma.array(image, mask=~valid_mask)


def draw_single_ax(fig, bbox, image, cmap, vmin, vmax, border_color,
                   valid_mask=None):
    """Draw one image in a new axes at the given Bbox with a black background."""
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
    """
    Draw a deck-of-cards stack of images within the given Bbox.

    images[0] = t-1  (most recent  →  front card, top-right, drawn last/on top)
    images[-1] = t-5 (oldest       →  back card,  bottom-left, drawn first)

    Older cards peek out from the bottom-left corner.
    All cards have a black background (masked land cells appear black).
    """
    valid = [(k, img) for k, img in enumerate(images) if img is not None]
    if not valid:
        _blank_ax(fig, bbox)
        return

    n      = len(valid)
    step_x = bbox.width  * 0.06      # 6 % of cell width per layer
    step_y = bbox.height * 0.06
    card_w = bbox.width  - (n - 1) * step_x
    card_h = bbox.height - (n - 1) * step_y

    cmap_obj = _make_cmap(cmap)

    # Draw back → front so the front card (t-1) lands on top.
    #   rank 0   = t-1  (front): offset = (n-1)*step  → top-right
    #   rank n-1 = t-5  (back):  offset = 0           → bottom-left
    for rank in range(n - 1, -1, -1):   # n-1 first (back), 0 last (front)
        _, img = valid[rank]
        offset = (n - 1 - rank)          # 0 for back card, n-1 for front card
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


# ============================================================
#  MAIN FIGURE
# ============================================================

def make_figure(runs=None, save_path=None, display_year=None,
                title='CrabTransformer — Top Model Summary'):
    """
    runs         : list of (model_size, level, channel_cfg, pred_mode, criterion).
                   Defaults to TARGET_RUNS at the top of this file.
    display_year : int | None  — calendar year to display in the spatial panels
                   (e.g. 2023).  Converted to year_idx = display_year - DATA_START_YEAR.
                   If None, the first valid test sample is used.
    """
    if runs is None:
        runs = TARGET_RUNS

    target_year_idx = None
    if display_year is not None:
        target_year_idx = display_year - DATA_START_YEAR
        title = f'{title} ({display_year})'

    n_rows = len(runs)
    n_cols = 7

    # ── Font sizes  ──────────────────────────────────────────────────────────
    # Adjust these values to resize text throughout the entire figure.
    FS_SUPTITLE    = 18   # main figure title at the very top
    FS_SECTION_HDR = 16   # "Inputs" / "Output" / "Target" section banners
    FS_COL_HDR     = 13   # column header labels (e.g. "Spawner history\n(t-1…t-5)")
    FS_ROW_LABEL   = 14   # run identifier text on the left margin
    FS_PHASE_LABEL = 13   # TRAIN / VAL / TEST labels in the line plot
    FS_AXIS_LABEL  = 14   # x-axis and y-axis labels in the line plot
    FS_AXIS_TITLE  = 13   # title of the line plot
    FS_LEGEND      = 12   # legend entries

    # ── Figure size ──────────────────────────────────────────────────────────
    cell_w    = 2.4
    cell_h    = 2.6
    label_w   = 1.6    # extra left margin for row labels
    line_h    = 5.2
    top_pad   = 0.55   # space for column headers (section headers go above grid_top)
    spacer_w  = 0.30   # width of each spacer column (fraction of cell_w); two spacers total
    fig_w     = label_w + cell_w * n_cols + cell_w * spacer_w * 2
    fig_h     = top_pad + cell_h * n_rows + line_h + 0.4

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=130)

    # ── Layout ───────────────────────────────────────────────────────────────
    # Convert label_w to a fraction of fig_w for the left margin.
    left_margin = label_w / fig_w

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[top_pad + cell_h * n_rows, line_h],
        hspace=0.08,
        top=0.87, bottom=0.10,
        left=left_margin, right=0.99,
    )
    # 9 grid columns: data cols 0-4, spacer, data col 5, spacer, data col 6
    # _gcol maps a 0-6 data column index to its gridspec column index
    def _gcol(c):
        if c < 5:  return c        # inputs: no offset
        if c == 5: return c + 1    # predicted: skip first spacer
        return c + 2               # true recruit: skip both spacers

    grid_top = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols + 2,   # +2 for the two gap spacers
        subplot_spec=outer[0],
        hspace=0.10, wspace=0.06,
        width_ratios=[1, 1, 1, 1, 1, spacer_w, 1, spacer_w, 1],
    )
    ax_line = fig.add_subplot(outer[1])

    # ── Spatial mask ─────────────────────────────────────────────────────────
    mask_path  = os.path.join(REPO_DIR, 'data/real/output/spatial_mask.npy')
    valid_mask = (np.load(mask_path) > 0) if os.path.exists(mask_path) \
                 else np.ones((50, 50), dtype=bool)

    # ── Pre-compute all cell positions BEFORE drawing anything ───────────────
    # This avoids the problem of fig.add_subplot() later replacing axes
    # that already have content.
    cell_bboxes = {}
    for r in range(n_rows):
        for c in range(n_cols):
            cell_bboxes[(r, c)] = _cell_pos(fig, grid_top, r, _gcol(c))

    # ── Header geometry ──────────────────────────────────────────────────────
    # top=0.87 puts the grid ceiling at ~0.87; the band 0.87→0.96 is header space.
    # Compute all positions here (needs cell_bboxes); draw section items LATER
    # (after all image axes) so they land on top in the rendering stack.
    outer_top    = outer[0].get_position(fig).y1   # ≈ 0.87
    top_row_top  = cell_bboxes[(0, 0)].y1           # ≈ 0.87
    fig_hdr_top  = 0.96                             # below suptitle
    hdr_space    = fig_hdr_top - top_row_top        # ≈ 0.09

    col_hdr_y = top_row_top + hdr_space * 0.28     # column header row
    sec_hdr_y = top_row_top + hdr_space * 0.78     # section header row (upper)
    rule_y    = top_row_top + hdr_space * 0.53     # horizontal rule between them

    inputs_x0 = cell_bboxes[(0, 0)].x0
    inputs_x1 = cell_bboxes[(0, 4)].x1
    output_x0 = cell_bboxes[(0, 5)].x0   # "Output" = predicted (col 5 only)
    output_x1 = cell_bboxes[(0, 5)].x1
    target_x0 = cell_bboxes[(0, 6)].x0   # "Target"  = true recruit (col 6 only)
    target_x1 = cell_bboxes[(0, 6)].x1
    inputs_cx = (inputs_x0 + inputs_x1) / 2
    output_cx = (output_x0 + output_x1) / 2
    target_cx = (target_x0 + target_x1) / 2
    div_x     = (cell_bboxes[(0, 4)].x1 + cell_bboxes[(0, 5)].x0) / 2  # inputs|output
    div2_x    = (cell_bboxes[(0, 5)].x1 + cell_bboxes[(0, 6)].x0) / 2  # output|target
    grid_bot  = cell_bboxes[(n_rows - 1, 0)].y0

    # Column headers — drawn now (before images); fine since they're in the
    # clear header band above top_row_top
    for col, ttl in enumerate(COL_TITLES):
        bbox = cell_bboxes[(0, col)]
        fig.text(
            bbox.x0 + bbox.width / 2, col_hdr_y, ttl,
            ha='center', va='center',
            fontsize=FS_COL_HDR, fontweight='bold',
            transform=fig.transFigure,
        )

    # ── Per-run processing ───────────────────────────────────────────────────
    line_data = {}    # row_idx → yearly-aggregate dict
    row_meta  = []    # (label, color, t_yr, v_yr, te_yr)

    for row_idx, run_spec in enumerate(runs):
        model_size, level, channel_cfg, pred_mode, criterion = run_spec
        color = ROW_COLORS[row_idx % len(ROW_COLORS)]
        label = RUN_DISPLAY_NAMES.get(
            run_spec,
            f"{channel_cfg} | {pred_mode} | {criterion} ({model_size})",
        )
        print(f'\n[{row_idx+1}/{n_rows}]  {label}')

        # Row label (figure text to the left of the row)
        bbox0  = cell_bboxes[(row_idx, 0)]
        row_cy = bbox0.y0 + bbox0.height / 2
        fig.text(
            left_margin - 0.01, row_cy, label,
            ha='right', va='center', fontsize=FS_ROW_LABEL,
            transform=fig.transFigure, color=color, fontweight='bold',
            rotation=0, wrap=True,
        )

        # ── Load run ────────────────────────────────────────────────────────
        try:
            (model, meta, bias, t_yr, v_yr, te_yr,
             tr_ld, va_ld, te_ld) = load_run(*run_spec)
        except Exception as exc:
            print(f'  ❌  Load failed: {exc}')
            for c in range(n_cols):
                _blank_ax(fig, cell_bboxes[(row_idx, c)], 'LOAD ERR')
            row_meta.append((label, color, 24, 8, 4, 0))
            continue

        data_type = 'real' if level == 'real' else 'dummy'
        row_meta.append((label, color, t_yr, v_yr, te_yr, meta.get('lag', 0)))

        # ── Spatial sample ──────────────────────────────────────────────────
        # For lag>0 models the dataset year_idx is the *input* year, which is
        # lag years earlier than the displayed calendar year.
        run_lag = meta.get('lag', 0)
        eff_year_idx = (target_year_idx - run_lag
                        if target_year_idx is not None else None)
        print('  Finding representative test sample …')
        inp_np, tgt_log, pred_log = get_test_sample(
            te_ld, model, bias, valid_mask,
            target_year_idx=eff_year_idx,
        )

        if inp_np is None:
            print('  ⚠  No valid test sample found; drawing blanks.')
            for c in range(n_cols):
                _blank_ax(fig, cell_bboxes[(row_idx, c)], 'NO DATA')
        else:
            ch = extract_channels(inp_np, meta)

            # ── Colour scales ────────────────────────────────────────────────
            def arr_of(lst, single=None):
                items = list(lst or []) + ([single] if single is not None else [])
                items = [x for x in items if x is not None]
                return np.stack(items) if items else None

            sp_arr  = arr_of(ch['sp_hist'],   ch['sp_curr'])
            rh_arr  = arr_of(ch['rec_hist'])
            tmp_arr = arr_of(ch['temp_hist'],  ch['temp_curr'])
            rec_arr = arr_of([tgt_log, pred_log])

            def safe_range(arr, symmetric=False):
                if arr is None:
                    return (0.0, 8.0)
                lo, hi = float(arr.min()), float(arr.max())
                if symmetric:
                    ab = max(abs(lo), abs(hi)) or 1.0
                    return (-ab, ab)
                return (lo, hi)

            sp_vmin,  sp_vmax  = safe_range(sp_arr)
            rh_vmin,  rh_vmax  = safe_range(rh_arr)
            tmp_vmin, tmp_vmax = safe_range(tmp_arr, symmetric=True)
            rec_vmin, rec_vmax = safe_range(rec_arr)

            # ── Draw 7 columns ───────────────────────────────────────────────

            # Col 0 — spawner history (stack)
            if ch['sp_hist']:
                draw_stack_ax(fig, cell_bboxes[(row_idx, 0)],
                              ch['sp_hist'], CMAPS['spawner'],
                              sp_vmin, sp_vmax, color, valid_mask)
            else:
                _blank_ax(fig, cell_bboxes[(row_idx, 0)], 'not used in this model')

            # Col 1 — spawner current
            if ch['sp_curr'] is not None:
                draw_single_ax(fig, cell_bboxes[(row_idx, 1)],
                               ch['sp_curr'], CMAPS['spawner'],
                               sp_vmin, sp_vmax, color, valid_mask)
            elif pred_mode in ('normal',):
                # Now-cast models: current spawner absent means not used
                _blank_ax(fig, cell_bboxes[(row_idx, 1)], 'not used in this model')
            else:
                # Forecast modes: current spawner excluded by prediction mode
                _blank_ax(fig, cell_bboxes[(row_idx, 1)], 'not included in\nthis model mode')

            # Col 2 — temp history (stack)
            if ch['temp_hist']:
                draw_stack_ax(fig, cell_bboxes[(row_idx, 2)],
                              ch['temp_hist'], CMAPS['temp'],
                              tmp_vmin, tmp_vmax, color, valid_mask)
            else:
                _blank_ax(fig, cell_bboxes[(row_idx, 2)], 'not used in this model')

            # Col 3 — temp current
            if ch['temp_curr'] is not None:
                draw_single_ax(fig, cell_bboxes[(row_idx, 3)],
                               ch['temp_curr'], CMAPS['temp'],
                               tmp_vmin, tmp_vmax, color, valid_mask)
            elif pred_mode in ('one_year_ahead',):
                _blank_ax(fig, cell_bboxes[(row_idx, 3)], 'not included in\nthis model mode')
            else:
                _blank_ax(fig, cell_bboxes[(row_idx, 3)], 'not used in this model')

            # Col 4 — recruit history (stack)
            if ch['rec_hist']:
                draw_stack_ax(fig, cell_bboxes[(row_idx, 4)],
                              ch['rec_hist'], CMAPS['recruit'],
                              rh_vmin, rh_vmax, color, valid_mask)
            else:
                _blank_ax(fig, cell_bboxes[(row_idx, 4)], 'not used in this model')

            # Col 5 — true recruit (observed target)
            draw_single_ax(fig, cell_bboxes[(row_idx, 5)],
                           tgt_log, CMAPS['recruit'],
                           rec_vmin, rec_vmax, color, valid_mask)

            # Col 6 — predicted recruit (model output)
            draw_single_ax(fig, cell_bboxes[(row_idx, 6)],
                           pred_log, CMAPS['recruit'],
                           rec_vmin, rec_vmax, color, valid_mask)

        # ── Yearly aggregates ────────────────────────────────────────────────
        print('  Collecting yearly aggregates …')
        agg = collect_yearly_aggregates(
            [tr_ld, va_ld, te_ld], model, bias, valid_mask
        )
        line_data[row_idx] = agg

    # ── Section headers + dividers — drawn AFTER all image axes so they sit
    #    on top in the render stack.  White bbox keeps text readable.
    _wbg = dict(facecolor='white', edgecolor='none', alpha=0.90, pad=2)

    # Section header text — font size controlled by FS_SECTION_HDR above
    for cx, label in [(inputs_cx, 'Inputs'),
                      (output_cx, 'Target'),
                      (target_cx, 'Output')]:
        fig.text(cx, sec_hdr_y, label,
                 ha='center', va='center', fontsize=FS_SECTION_HDR,
                 fontweight='bold', transform=fig.transFigure,
                 color='#111111', bbox=_wbg)

    # Horizontal rules under each section header
    for x0, x1 in [(inputs_x0, inputs_x1),
                   (output_x0, output_x1),
                   (target_x0, target_x1)]:
        fig.add_artist(Line2D([x0, x1], [rule_y, rule_y],
                              transform=fig.transFigure,
                              color='#aaaaaa', lw=1.0))

    # Vertical dividers: inputs|output and output|target
    for dx in [div_x, div2_x]:
        fig.add_artist(Line2D([dx, dx], [grid_bot, fig_hdr_top],
                              transform=fig.transFigure,
                              color='#666666', lw=1.5, ls='--'))

    # ── Line plot ─────────────────────────────────────────────────────────────
    print('\nBuilding line plot …')
    obs_plotted = False   # draw observed (black dashed) only once
    for row_idx, (label, color, t_yr, v_yr, te_yr, lag) in enumerate(row_meta):
        if row_idx not in line_data:
            continue
        agg          = line_data[row_idx]
        years_sorted = sorted(agg.keys())
        # Shift by lag so lag>0 models align with their *predicted* year,
        # then add DATA_START_YEAR to convert year_idx → calendar year.
        years_plot   = [y + lag + DATA_START_YEAR for y in years_sorted]
        pred_med = [np.median(agg[y]['pred']) for y in years_sorted]
        obs_med  = [np.median(agg[y]['obs'])  for y in years_sorted]
        pred_p25 = [np.percentile(agg[y]['pred'], 25) for y in years_sorted]
        pred_p75 = [np.percentile(agg[y]['pred'], 75) for y in years_sorted]

        ax_line.fill_between(years_plot, pred_p25, pred_p75,
                             color=color, alpha=0.15)
        ax_line.plot(years_plot, pred_med, '-',  color=color, lw=2.0,
                     label=f'Pred — {label}')
        # Observed: single black dashed line drawn once (first lag-0 run)
        if lag == 0 and not obs_plotted:
            ax_line.scatter(years_plot, obs_med, color='black', s=18, zorder=5,
                            alpha=0.85, label='Observed')
            obs_plotted = True

    # Phase shading — use first run's splits (lag0 reference)
    # all_years holds raw year_idx values; convert to calendar years for plotting.
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
            ax_line.text((y0_cal + train_end_cal) / 2,    0.93, 'TRAIN', transform=xform,
                         ha='center', fontsize=FS_PHASE_LABEL, color='seagreen',   fontweight='bold')
            ax_line.text((train_end_cal + val_end_cal) / 2, 0.93, 'VALIDATION', transform=xform,
                         ha='center', fontsize=FS_PHASE_LABEL, color='darkorange', fontweight='bold')
            ax_line.text((val_end_cal + y1_cal) / 2,      0.93, 'TEST',  transform=xform,
                         ha='center', fontsize=FS_PHASE_LABEL, color='crimson',    fontweight='bold')

    legend_handles = [
        mpatches.Patch(color=color, label=label[:55])
        for label, color, *_ in row_meta
    ] + [
        plt.Line2D([0], [0], color='k', lw=2.0, ls='-', label='Predicted (solid)'),
    ]
    if obs_plotted:
        legend_handles.append(
            plt.Line2D([0], [0], color='black', lw=1.5, ls='--', label='Observed (dashed)')
        )
    ax_line.legend(handles=legend_handles, fontsize=FS_LEGEND,
                   loc='upper center', bbox_to_anchor=(0.5, -0.18),
                   framealpha=0.85, ncol=2, borderaxespad=0.)
    ax_line.set_xlabel('Year', fontsize=FS_AXIS_LABEL)
    ax_line.set_ylabel('Total recruit abundance (density)', fontsize=FS_AXIS_LABEL)
    ax_line.set_title(
        'Aggregate yearly recruits',
        fontsize=FS_AXIS_TITLE,
        fontweight='bold',
    )
    ax_line.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=FS_SUPTITLE, fontweight='bold', y=0.995)

    out = save_path or os.path.join(SAVE_DIR, 'top_model_figure.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n✅  Saved → {out}')
    return out


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == '__main__':
    make_figure(display_year=2023)
