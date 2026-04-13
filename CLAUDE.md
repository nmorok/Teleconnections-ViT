# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

CrabTransformer: a Vision Transformer (ViT) that predicts snow crab recruit density from multi-channel spatial grids. The model takes gridded spawner biomass, historical recruit counts, and bottom temperature as inputs and outputs a 50×50 predicted recruit density map.

## Common Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
# PyTorch must be installed separately — not listed in requirements.txt
```

**Create data splits (run from repo root):**
```bash
python data/create_splits.py
# Outputs to data/dummy/splits/{easy,medium,hard}/ and data/real/splits/{nolag,lag5}/real/
```

**Process real data (requires R):**
```bash
Rscript data/real/process_data.R
Rscript data/real/process_temperature_data.R
```

**Training:** `models/train.ipynb` is run on Google Colab Pro (GPU). Data lives on Google Drive at `/content/drive/MyDrive/Teleconnection_ViT/`. The notebook contains four main scripts as cells:

- **Cell 5** — `validate_configs.py`: sanity-checks every channel-config × model-size combination with a single CPU forward pass; generates `validation_results.csv` and channel-grid PNGs.
- **Cell 7** — `train.py`: master training loop; iterates over the full run matrix.
- **Cell 9** — `run_batch_evaluation.py`: auto-discovers all `best_model.pt` checkpoints under `DRIVE_BASE`, runs inference on all splits, writes `full_results.csv` / `summary_results.csv`, and saves per-run plots.
- **Cell 12** — Integrated Gradients attribution analysis (post-training).

**Validate data pipeline:**
```bash
python data/validate_data_helper.py
```

## Architecture

### Model (`models/`)

`CrabTransformer` (`model.py`) is the top-level module. It wires together:

1. **`PatchEmbedding`** (`components.py`): Divides the 50×50 grid into 5×5 patches (100 patches total). Each input channel has its own learnable linear projection. Channels are zeroed out using a **6-slot temporal mask** (`[current_year, t-1, …, t-5]`) via `channel_mask_indices`, which maps each channel to the correct mask slot.
2. **`PositionalEncoding2D`**: Learned 2D position embeddings (one per patch).
3. **`TemporalEncoding`**: Fixed sinusoidal encoding added across all patches, keyed by `year_index`.
4. **`TransformerBlock`** × N: Pre-norm self-attention (MHSA) + GELU feed-forward. Returns attention weights when `return_attention=True`.
5. **`SpatialDecoder`**: Three bilinear-upsample + conv stages with GroupNorm: `[B, D, 10, 10] → [B, 1, 50, 50]`. Softplus at the end enforces non-negativity. Multiplied by the spatial (land/ocean) mask.

**Critical wiring:** `dataset.in_channels` and `dataset.channel_mask_indices` must be passed to `CrabTransformer`. Do not hardcode channel counts — they vary by channel-group configuration.

### Data (`data/`)

`CrabDataset` / `get_dataloaders` (`data_helper.py`):

- **Channel groups** (any non-empty subset):
  - Spawners: current-year + 5-year history → 6 ch (or 5 ch in one-year-ahead mode)
  - Recruits: 5-year history only → always 5 ch (no current channel by design)
  - Temperature: current-year + 5-year history → 6 ch (or 5 ch in one-year-ahead mode)
- **`get_channel_info(use_spawners, use_recruits, use_temp, include_current)`** computes `(in_channels, channel_mask_indices)` without needing to instantiate a dataset.
- Spawner and recruit arrays are **log1p-transformed** at load time. Temperature is not (has negative values).
- **2020 is masked** (`year_mask[relative_year] = 0`) in real data due to COVID survey gap.
- Data is structured as `n_bootstraps × n_years` (100 bootstraps default). Historical context from the previous split is passed explicitly to bridge the train→val and val→test boundaries.
- `get_dataloaders` returns `(train_loader, val_loader, test_loader)` and handles the historical carryover automatically.

**Data types:**
- `dummy`: GMRF-generated synthetic data at three difficulties (`easy`/`medium`/`hard`), stored in `data/dummy/splits/{level}/`. No temperature.
- `real`: actual snow crab survey data from `data/real/splits/{nolag,lag5}/real/`. Requires the R preprocessing scripts.

### Losses (`models/losses.py`)

- `TweedieLoss(power=1.5)`: main loss for non-negative data with excess zeros. Both losses accept `mask` (spatial land mask) and `sample_mask` (per-sample validity flag to exclude gap years from the gradient).
- `MSELoss_cm`: spatially masked MSE, used for comparison.
- **Critical difference**: Tweedie loss receives **back-transformed** predictions (`torch.expm1(outputs).clamp(min=1e-6)`); MSE loss receives **log-space** predictions directly.

## Training Details

### Run Matrix

The full run matrix (built by `build_run_matrix()` in Cell 7) crosses:
- **Model sizes**: `normal` (128d, 8h, 6L, d_ff=512, dropout=0.1) and `small` (128d, 4h, 3L, d_ff=512, dropout=0.2)
- **7 channel configs**: `all`, `sp_rec`, `sp_temp`, `rec_temp`, `spawners_only`, `recruits_only`, `temp_only`
- **3 prediction modes**: `normal` (incl_curr=True, lag=0), `one_year_ahead` (incl_curr=False, lag=0), `lag5` (incl_curr=True, lag=5)
- **2 criteria**: `MSE`, `Tweedie`
- **Data**: dummy (easy/medium/hard, no temp) and real

Rules: dummy data skips temp-requiring configs and `lag5`; `one_year_ahead` is skipped for `recruits_only` (no current-year channel to drop).

### Hyperparameters

```
NUM_EPOCHS=20, BATCH_SIZE=8, MEMORY_YEARS=5
TWEEDIE_POWER=1.2, MAX_LR=3e-4, BASE_LR=1e-4
WEIGHT_DECAY=1e-4, GRAD_CLIP=1.0, PATIENCE=20
Optimizer: AdamW + OneCycleLR (pct_start=0.3, cos annealing)
```

### Checkpoint Structure

```
model_outputs/{model_size}/{level}/{channel_cfg}/{pred_mode}/{criterion}/
    best_model.pt
    training_history.json   ← includes channel_cfg_meta for eval recovery
    training_curves.png
```

`training_history.json` stores `channel_cfg_meta` (in_channels, channel_mask_indices, all flags) so `run_batch_evaluation.py` can reconstruct the model without re-running `get_dataloaders`.

### Decoder Bias Warm-Start

Before training, `decoder.conv_out.bias` is set to `mean(log1p(training targets))` over valid ocean cells only. This prevents severe underestimation at epoch 0.

### Post-Training Bias Correction (MSE only)

After loading the best checkpoint, residuals `(target - pred)` are computed over the training set. The correction factor `exp(μ + σ²/2)` is stored in `training_history.json` and applied at evaluation time: `pred_final = exp(pred_log) × bias_correction - 1`.

### Evaluation Metrics

`run_batch_evaluation.py` computes metrics **twice** per sample:
- `raw_*`: back-transformed predictions with no additional zeroing
- `thresh_*`: predictions and targets zeroed below `ZERO_THRESHOLD = 14.23` first

Metrics include: Spearman/Pearson spatial correlation, MAE, RMSE, bias, unbiased RMSE, abundance capture ratio, top-10% Jaccard, zero-cell precision/recall/F1, and MSE/MAE skill scores vs. training climatology.

### Year Splits

| Data type | Train | Val | Test |
|-----------|-------|-----|------|
| dummy (all levels) | 18 | 9 | 3 |
| real, lag=0 | 24 | 8 | 4 |
| real, lag=5 | 21 | 6 | 4 |

## Key Conventions

- All data arrays are flattened to `[n_bootstraps × n_years, 50, 50]` before being saved to `.npy` files.
- `year_offset` in `CrabDataset` ensures that year indices are contiguous across train/val/test for `TemporalEncoding`.
- `SKIP_IF_EXISTS=True` in `train.py` lets interrupted runs resume safely by skipping any directory that already has `best_model.pt`.
- `diagnose_model.py` provides gradient explosion checks — run it when adding new model variants.
