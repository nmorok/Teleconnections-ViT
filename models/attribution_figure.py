"""
attribution_figure.py
=====================
Integrated Gradients attribution analysis for top CrabTransformer runs.

Creates three outputs per run:
  - Per-year spatial panels  (input channels + attributions side-by-side)
  - Grand-average spatial panel
  - Channel-attribution bar chart by year

Usage
-----
    python models/attribution_figure.py

Or import and call run_targeted_ig() from a Colab cell.
"""

import os, sys, json, math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_DIR    = '/content/Teleconnections-ViT'
DRIVE_DIR   = '/content/drive/MyDrive/Teleconnection_ViT'
OUTPUTS_DIR = os.path.join(DRIVE_DIR, 'model_outputs')
SAVE_DIR    = os.path.join(DRIVE_DIR, 'analysis/attribution')

DATA_START_YEAR = 1988   # year_idx 0 = calendar year 1988

sys.path.insert(0, REPO_DIR)
from models.model     import CrabTransformer
from data.data_helper import get_dataloaders

os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Font sizes ─────────────────────────────────────────────────────────────────
# Adjust these to resize all text in every figure produced by this script.
FS_TITLE      = 22   # suptitle / main figure title
FS_PANEL      = 16   # per-panel subplot titles  (channel labels in spatial maps)
FS_AXIS_LABEL = 18   # x / y axis labels in bar chart
FS_TICK       = 15   # tick labels in bar chart
FS_LEGEND     = 15   # legend entries

# ── Run display names ──────────────────────────────────────────────────────────
# Maps (model_size, level, channel_cfg, pred_mode, criterion) → display label.
# Edit entries here to control how each model is named in figure titles.
# Runs not listed here fall back to an auto-generated label.
RUN_DISPLAY_NAMES = {
    ('normal', 'real', 'all',        'normal',        'MSE'):     'All channels | Now-cast | MSE | Base-size',
    ('small',  'real', 'temp_only',  'normal',        'Tweedie'): 'Bottom Temp only | Now-cast | Tweedie | Reduced-size',
    ('small',  'real', 'rec_temp',   'one_year_ahead','MSE'):     'Recruits + Bottom Temp | 1-yr ahead | MSE | Reduced-size',
    ('normal', 'real', 'sp_temp',    'lag5',          'MSE'):     'Spawners + Bottom Temp | Lag-5 | MSE | Base-size',
}
# ── Target runs ────────────────────────────────────────────────────────────────
# (model_size, level, channel_cfg, pred_mode, criterion)
TARGET_MODELS = [
    ('normal', 'real', 'all',       'normal',        'MSE'),
    ('small',  'real', 'temp_only', 'normal',        'Tweedie'),
    ('small',  'real', 'rec_temp',  'one_year_ahead','MSE'),
    ('normal', 'real', 'sp_temp',   'lag5',          'MSE'),
]


# ── Channel name helper ────────────────────────────────────────────────────────

def get_active_channel_names(meta):
    """Return channel display names in the same order as the model's input channels."""
    names = []
    incl  = meta['incl_curr']
    if meta['use_spawners']:
        if incl:
            names.append('Spawner (t)')
        names.extend(['Spawner (t-1)', 'Spawner (t-2)', 'Spawner (t-3)',
                       'Spawner (t-4)', 'Spawner (t-5)'])
    if meta['use_recruits']:
        names.extend(['Recruit (t-1)', 'Recruit (t-2)', 'Recruit (t-3)',
                       'Recruit (t-4)', 'Recruit (t-5)'])
    if meta['use_temp']:
        if incl:
            names.append('Bot. Temp (t)')
        names.extend(['Bot. Temp (t-1)', 'Bot. Temp (t-2)', 'Bot. Temp (t-3)',
                       'Bot. Temp (t-4)', 'Bot. Temp (t-5)'])
    return names


# ── Step 1: Compute baseline ───────────────────────────────────────────────────

def compute_baseline(train_loader):
    print('  Computing baseline (mean training field)...')
    all_inputs = []
    for inputs, _, _, _, _, _ in train_loader:
        all_inputs.append(inputs)
    stacked  = torch.cat(all_inputs, dim=0)
    baseline = stacked.mean(dim=0, keepdim=True)
    print(f'    Baseline from {stacked.shape[0]} samples  '
          f'range [{baseline.min():.3f}, {baseline.max():.3f}]')
    return baseline


# ── Step 2: Organize samples by year ──────────────────────────────────────────

def collect_samples_by_year(loader):
    by_year = {}
    for inputs, targets, mask, year_idx, spat_mask, val_year in loader:
        for i in range(inputs.shape[0]):
            if val_year[i] == 0:
                continue
            yr = int(year_idx[i].item())
            by_year.setdefault(yr, []).append((
                inputs[i:i+1], targets[i:i+1], mask[i:i+1], year_idx[i:i+1],
            ))
    for yr in sorted(by_year.keys()):
        print(f'    Year idx {yr} ({yr + DATA_START_YEAR}): {len(by_year[yr])} bootstrap samples')
    return by_year


# ── Step 3: Integrated Gradients ──────────────────────────────────────────────

def integrated_gradients(model, actual_input, baseline, year_idx,
                         temporal_mask, n_steps=50, output_fn=None):
    if output_fn is None:
        output_fn = lambda out: out.mean()
    delta             = actual_input - baseline
    accumulated_grads = torch.zeros_like(actual_input)
    for step in range(n_steps):
        alpha        = step / n_steps
        interpolated = (baseline + alpha * delta).detach().clone().requires_grad_(True)
        output_fn(model(interpolated, year_idx, temporal_mask)).backward()
        accumulated_grads += interpolated.grad.detach()
        model.zero_grad()
    return ((accumulated_grads / n_steps) * delta).squeeze(0).cpu().numpy()


# ── Step 4: Per-year IG, averaged across bootstraps ───────────────────────────

def compute_yearly_attributions(model, by_year, baseline, n_steps=50, output_fn=None):
    baseline = baseline.to(DEVICE)
    yearly_attr, yearly_inputs = {}, {}
    for yr in sorted(by_year.keys()):
        samples = by_year[yr]
        print(f'\n  Year {yr} ({yr + DATA_START_YEAR}): running IG on {len(samples)} bootstraps...')
        attr_list, input_list = [], []
        for idx, (inp, _, msk, yr_idx) in enumerate(samples):
            attr = integrated_gradients(
                model, inp.to(DEVICE), baseline,
                yr_idx.to(DEVICE), msk.to(DEVICE),
                n_steps=n_steps, output_fn=output_fn,
            )
            attr_list.append(attr)
            input_list.append(inp.squeeze(0).numpy())
            if (idx + 1) % 25 == 0:
                print(f'    {idx + 1}/{len(samples)} done')
        yearly_attr[yr]   = np.stack(attr_list).mean(axis=0)
        yearly_inputs[yr] = np.stack(input_list).mean(axis=0)
        print(f'    Done. Attr range [{yearly_attr[yr].min():.6f}, {yearly_attr[yr].max():.6f}]')
    return yearly_attr, yearly_inputs


# ── Step 5: Visualization ─────────────────────────────────────────────────────

def plot_year_panel(yr, mean_input, mean_attr, meta, out_name, display_name):
    """
    Spatial panel: top half = mean input channels, bottom half = attributions.
    Grid width and height scale automatically to the number of channels.
    """
    mask_path  = os.path.join(REPO_DIR, 'data/real/output/spatial_mask.npy')
    valid_mask = (np.load(mask_path) > 0) if os.path.exists(mask_path) \
                 else np.ones((50, 50), dtype=bool)

    def _masked(image):
        return np.ma.array(image, mask=~valid_mask)

    import copy
    def _cmap(name):
        c = copy.copy(plt.get_cmap(name))
        c.set_bad('white')
        return c

    n_channels   = meta['in_channels']
    ch_names     = get_active_channel_names(meta)
    n_items      = n_channels + 1          # channels + aggregate panel
    n_cols       = min(n_channels, 6)      # at most 6 columns
    rows_per_half = math.ceil(n_items / n_cols)
    total_rows   = rows_per_half * 2

    cell_size = 4.0
    fig, axes = plt.subplots(
        total_rows, n_cols,
        figsize=(cell_size * n_cols, cell_size * total_rows),
        squeeze=False,
    )

    # Hide all panels; we'll re-enable only the ones that have content.
    for ax in axes.flat:
        ax.axis('off')

    abs_attr  = np.abs(mean_attr)
    vmax_attr = np.percentile(abs_attr, 99)

    # Top half — mean input channels
    for c in range(n_channels):
        row, col = divmod(c, n_cols)
        ax = axes[row, col]
        is_temp = 'Temp' in ch_names[c]
        cmap    = 'RdBu_r' if is_temp else 'viridis'
        vmax    = None if is_temp else 8.0
        ax.imshow(_masked(mean_input[c]), cmap=_cmap(cmap), vmin=None if is_temp else 0, vmax=vmax)        
        ax.set_title(f'Input: {ch_names[c]}', fontsize=FS_PANEL, fontweight='bold')
        ax.axis('off')

    agg_row, agg_col = divmod(n_channels, n_cols)
    axes[agg_row, agg_col].imshow(_masked(np.abs(mean_input).sum(axis=0)), cmap=_cmap('viridis'))
    axes[agg_row, agg_col].set_title('Input: Aggregate', fontsize=FS_PANEL, fontweight='bold')
    axes[agg_row, agg_col].axis('off')

    # Bottom half — attribution channels
    for c in range(n_channels):
        row, col = divmod(c, n_cols)
        ax = axes[rows_per_half + row, col]
        ax.imshow(_masked(mean_attr[c]), cmap=_cmap('RdBu_r'), vmin=-vmax_attr, vmax=vmax_attr)
        ax.set_title(f'Attr: {ch_names[c]}', fontsize=FS_PANEL, fontweight='bold')
        ax.axis('off')

    axes[rows_per_half + agg_row, agg_col].imshow(_masked(abs_attr.sum(axis=0)), cmap=_cmap('hot'))
    axes[rows_per_half + agg_row, agg_col].set_title(
        'Attr: Aggregate |IG|', fontsize=FS_PANEL, fontweight='bold')
    axes[rows_per_half + agg_row, agg_col].axis('off')

    cal_yr = f'{yr + DATA_START_YEAR}' if isinstance(yr, int) else 'Average'
    fig.suptitle(
        'Spatial Attribution' +
        f'{display_name} — {cal_yr}\n'
        'Top: Mean input across bootstraps  |  Bottom: Attribution  (red = +recruit, blue = −recruit)',
        fontsize=FS_TITLE, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(SAVE_DIR, f'ig_year{yr}_{out_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close()


def plot_grand_average(yearly_attr, yearly_inputs, meta, out_name, display_name):
    years       = sorted(yearly_attr.keys())
    grand_attr  = np.stack([yearly_attr[yr]   for yr in years]).mean(axis=0)
    grand_input = np.stack([yearly_inputs[yr] for yr in years]).mean(axis=0)
    plot_year_panel('AVG', grand_input, grand_attr, meta, out_name, display_name)


def plot_channel_bar_yearly(yearly_attr, meta, out_name, display_name):
    """
    Bar chart: share of total |attribution| per channel, one bar group per year.
    Year indices are converted to calendar years in the legend.
    """
    years    = sorted(yearly_attr.keys())
    ch_names = get_active_channel_names(meta)
    n_ch     = meta['in_channels']

    # Build data: calendar-year string → % attribution array
    data = {}
    for yr in years:
        totals      = np.abs(yearly_attr[yr]).sum(axis=(1, 2))
        data[str(yr + DATA_START_YEAR)] = totals / totals.sum() * 100

    grand        = np.stack([yearly_attr[yr] for yr in years]).mean(axis=0)
    grand_totals = np.abs(grand).sum(axis=(1, 2))
    data['Average'] = grand_totals / grand_totals.sum() * 100

    x        = np.arange(n_ch)
    n_groups = len(data)
    width    = 0.8 / n_groups
    colors   = plt.cm.tab10(np.linspace(0, 1, n_groups))

    fig_w = max(16, 1.8 * n_ch)
    fig, ax = plt.subplots(figsize=(fig_w, 8))

    for i, (label, pcts) in enumerate(data.items()):
        offset = (i - n_groups / 2 + 0.5) * width
        ax.bar(x + offset, pcts, width, label=label,
               color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ch_names, rotation=45, ha='right', fontsize=FS_TICK)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_ylabel('Share of Total Attribution (%)', fontsize=FS_AXIS_LABEL)
    ax.set_title(
        f'Channel Attribution by Year\n{display_name}',
        fontsize=FS_TITLE, fontweight='bold',
    )
    ax.legend(loc='upper right', fontsize=FS_LEGEND)
    ax.grid(axis='y', alpha=0.3, ls='--')
    plt.tight_layout()

    path = os.path.join(SAVE_DIR, f'ig_bar_yearly_{out_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close()


# ── Cache loader ──────────────────────────────────────────────────────────────

def load_saved_attributions(out_name):
    """
    Look for previously saved .npy attribution arrays in SAVE_DIR.

    Returns (yearly_attr, yearly_inputs) if all matching pairs are found,
    or (None, None) if no attr files exist for this run.
    """
    import glob
    prefix = 'ig_attr_yr'
    suffix = f'_{out_name}.npy'
    attr_paths = sorted(glob.glob(os.path.join(SAVE_DIR, f'{prefix}*{suffix}')))

    if not attr_paths:
        return None, None

    yearly_attr, yearly_inputs = {}, {}
    for attr_path in attr_paths:
        fname  = os.path.basename(attr_path)
        yr_str = fname[len(prefix):-len(suffix)]
        yr     = int(yr_str)

        input_path = os.path.join(SAVE_DIR, f'ig_input_yr{yr}{suffix}')
        if not os.path.exists(input_path):
            print(f'  Warning: attr file exists for year {yr} but input file is missing — will recompute.')
            return None, None

        yearly_attr[yr]   = np.load(attr_path)
        yearly_inputs[yr] = np.load(input_path)
        print(f'  Loaded year {yr} ({yr + DATA_START_YEAR})')

    return yearly_attr, yearly_inputs


# ── Main entry point ──────────────────────────────────────────────────────────

def run_targeted_ig(model_size, level, channel_cfg, pred_mode, criterion,
                    phase='TEST', n_steps=50, force_recompute=False):
    """
    Run (or reload) Integrated Gradients for one model configuration.

    Parameters
    ----------
    force_recompute : bool
        If False (default), saved .npy arrays in SAVE_DIR are reused and the
        expensive IG computation is skipped.  Set to True to redo everything
        from scratch (e.g. after changing n_steps or the model checkpoint).
    """
    run_spec     = (model_size, level, channel_cfg, pred_mode, criterion)
    out_name     = f'{model_size}_{level}_{channel_cfg}_{pred_mode}_{criterion}'
    display_name = RUN_DISPLAY_NAMES.get(
        run_spec,
        f'{channel_cfg} | {pred_mode} | {criterion} ({model_size})',
    )

    print(f'\n{"="*70}')
    print(f' IG attribution: {display_name}')
    print(f' Phase: {phase} | Steps: {n_steps} | force_recompute: {force_recompute}')
    print(f'{"="*70}')

    ckpt_dir     = os.path.join(OUTPUTS_DIR, model_size, level, channel_cfg, pred_mode, criterion)
    history_path = os.path.join(ckpt_dir, 'training_history.json')

    if not os.path.exists(history_path):
        print(f'  Missing training_history.json in {ckpt_dir} — skipping.')
        return

    # Meta is always loaded from JSON (fast — no model or data needed).
    with open(history_path) as f:
        meta = json.load(f)['channel_cfg_meta']
    print(f"  Channels: {meta['in_channels']}  Lag: {meta['lag']}  "
          f"Temp: {meta['use_temp']}  Recruits: {meta['use_recruits']}")

    # ── Try to use saved arrays ───────────────────────────────────────────────
    yearly_attr, yearly_inputs = None, None
    if not force_recompute:
        print('  Checking for saved attribution arrays...')
        yearly_attr, yearly_inputs = load_saved_attributions(out_name)
        if yearly_attr:
            print(f'  Found {len(yearly_attr)} saved years — skipping IG computation.')

    # ── Compute IG if no cached arrays ───────────────────────────────────────
    if yearly_attr is None:
        ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
        if not os.path.exists(ckpt_path):
            print(f'  Missing best_model.pt in {ckpt_dir} — skipping.')
            return

        model = CrabTransformer(
            grid_size=50, patch_size=5,
            in_channels=meta['in_channels'],
            embed_dim=meta['embed_dim'],
            num_heads=meta['num_heads'],
            num_layers=meta['num_layers'],
            d_ff=meta['d_ff'],
            dropout=0,
            channel_mask_indices=meta['channel_mask_indices'],
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()

        t_yr, v_yr, te_yr = (24, 8, 4) if meta['lag'] == 0 else (21, 6, 4)
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=8, memory_years=5,
            train_years=t_yr, val_years=v_yr, test_years=te_yr,
            data_type='real', level=level,
            include_current_spawner=meta['incl_curr'],
            lag=meta['lag'],
            use_temp=meta['use_temp'],
            use_spawners=meta['use_spawners'],
            use_recruits=meta['use_recruits'],
        )

        baseline      = compute_baseline(train_loader)
        target_loader = {'TEST': test_loader, 'VAL': val_loader, 'TRAIN': train_loader}[phase]

        print(f'\n  Collecting {phase} samples by year...')
        by_year = collect_samples_by_year(target_loader)

        yearly_attr, yearly_inputs = compute_yearly_attributions(
            model, by_year, baseline, n_steps=n_steps,
        )

        for yr in sorted(yearly_attr.keys()):
            np.save(os.path.join(SAVE_DIR, f'ig_attr_yr{yr}_{out_name}.npy'),   yearly_attr[yr])
            np.save(os.path.join(SAVE_DIR, f'ig_input_yr{yr}_{out_name}.npy'), yearly_inputs[yr])
        print(f'  Arrays saved to {SAVE_DIR}')

    # ── Regenerate figures (always runs) ─────────────────────────────────────
    print('\n  Generating visualizations...')
    for yr in sorted(yearly_attr.keys()):
        plot_year_panel(yr, yearly_inputs[yr], yearly_attr[yr], meta, out_name, display_name)

    plot_grand_average(yearly_attr, yearly_inputs, meta, out_name, display_name)
    plot_channel_bar_yearly(yearly_attr, meta, out_name, display_name)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Set force_recompute=True to redo the IG computation from scratch.
    for run in TARGET_MODELS:
        run_targeted_ig(*run, phase='TEST', n_steps=50, force_recompute=False)
