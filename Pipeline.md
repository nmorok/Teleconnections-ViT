# Transformer Transformer Project - Complete Pipeline Documentation

**Snow Crab Recruitment Prediction using Vision Transformers**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Specifications](#data-specifications)
3. [Model Specifications](#model-specifications)
4. [Training Specifications](#training-specifications)
5. [Pipeline Skeleton](#pipeline-skeleton)
6. [Detailed Function Documentation](#detailed-function-documentation)
7. [Workflow Summary](#workflow-summary)
8. [Completion Checklist](#completion-checklist)

---

## Project Overview

### Research Question
Can a Vision Transformer predict teleconnections, with a case study on the snow crab spawner/recruitment relationship.

### Approach
- **Input**: Spawner density fields ([PLACEHOLDER]×[PLACEHOLDER] km grid, eastern Bering Sea)
- **Output**: Recruitment density fields (same grid, 5 years later)
- **Architecture**: Vision Transformer adapted for spatial regression
- **Baseline**: Compare against EOF-GLLVM, traditional EOF+GAM, spectral clustering

### Data Sources
- **Real data**: NOAA bottom trawl survey (349 stations, 1988-2023)
- **Dummy data**: GMRF-generated synthetic fields (for pipeline testing)

---

## 📊 Data Specifications

### Real Data Dimensions
```
Spawners:  [2800, [PLACEHOLDER], [PLACEHOLDER]]  = 30 years × 100 bootstraps × [PLACEHOLDER]×[PLACEHOLDER] grid
Recruits:  [2800, [PLACEHOLDER], [PLACEHOLDER]]  = 30 years × 100 bootstraps × [PLACEHOLDER]×[PLACEHOLDER] grid
```

**Why different lengths?**
- Spawner years: 1988-2018 (30 years, excluding 2015 survey gap)
- Recruitment years: 1993-2023 (30 years, 5-year lag, excluding 2020)
- Each bootstrap = one SPDE interpolation realization

### Dummy Data Dimensions
```
Spawners:  [3000, 10, 10] or [3000, 50, 50]
Recruits:  [3000, 10, 10] or [3000, 50, 50]
```
- **10×10**: Fast testing, prototype development
- **50×50**: Production scale, final training

### Spatial Properties
- **Grid resolution**: [PLACEHOLDER] km per cell
- **Coverage**: Eastern Bering Sea survey domain
- **CRS**: [PLACEHOLDER]
- **Spatial correlation**: ~60 km range (κ = 0.3)

### Temporal Properties
- **Temporal autocorrelation**: ρ = 0.7 (strong year-to-year persistence)
- **Spawner-recruitment lag**: none, since dummy data
- **Spawner-recruitment correlation**: r ≈ 0.3 (moderate relationship)

### Data Organization
```
data/
├── raw/                          # Original R data
│   ├── survDAT.RData            # NOAA survey data
│   └── EBS.rds                  # Survey domain polygon
├── processed/                    # R-extracted CSVs
│   ├── spawner_station_data.csv
│   └── recruit_station_data.csv
    └── splits/                       # Train/val/test
        ├── train_spawners.npy
        ├── train_recruits.npy
        ├── val_spawners.npy
        ├── val_recruits.npy
        ├── test_spawners.npy
        └── test_recruits.npy
├── dummy/   
    └── generate_dummy_data.py         # GMRF synthetic data
│   └── output/
        ├── gmrf_spawners_10x10.npy
│       ├── gmrf_recruits_10x10.npy
│       ├── gmrf_spawners_50x50.npy
│       ├── gmrf_recruits_50x50.npy
│       └── gmrf_params.json
    └── splits/                       # Train/val/test
        ├── train_spawners.npy
        ├── train_recruits.npy
        ├── val_spawners.npy
        ├── val_recruits.npy
        ├── test_spawners.npy
        └── test_recruits.npy
├── spde_output/                  # Real SPDE bootstrap
│   ├── spawners_50x50.npy       # [3000, 50, 50]
│   └── recruits_50x50.npy       # [3000, 50, 50]

```

---

## Model Specifications

### Architecture: Vision Transformer for Spatial Regression

#### Input Processing
- **Input shape**: `[batch, 1, [PLACEHOLDER], [PLACEHOLDER]]` (single-channel density map)
- **Patch size**: [PLACEHOLDER]×[PLACEHOLDER] pixels
- **Number of patches**: [PLACEHOLDER] ([PLACEHOLDER]×[PLACEHOLDER] grid of patches)
- **Patch embedding**: Linear projection to 128-dim vectors

#### Transformer Encoder
- **Embedding dimension**: 128
- **Number of layers**: [PLACEHOLDER]
- **Attention heads**: [PLACEHOLDER] ([PLACEHOLDER] dim per head)
- **MLP hidden dim**: [PLACEHOLDER] (4× embedding dim)
- **Dropout**: [PLACEHOLDER]
- **Layer norm**: Pre-norm architecture

#### Positional Encoding
- **Type**: Learnable 2D spatial positions
- **Shape**: `[100, 128]` (one vector per patch)
- **Why learnable?** Grid has irregular survey boundaries; learned positions capture domain shape

#### Decoder
- **Architecture**: Reshape → Conv transpose layers
- **Output shape**: `[batch, 1, [PLACEHOLDER], [PLACEHOLDER]]`
- **Activation**: None (predicting densities directly)

#### Model Size
- **Parameters**: ~[PLACEHOLDER] million
- **Memory**: ~[PLACEHOLDER] MB (weights only)
- **Training memory**: ~[PLACEHOLDER] GB (batch_size=[PLACEHOLDER], grad checkpointing off)

### Loss Function
**Weighted MSE with bootstrap uncertainty**:
```
Loss = Σ [w_i × (pred_i - target_i)²]
where w_i = 1 / (uncertainty_i² + ε)
```

**NOTE: or do we use Tweedie to handle the inflated zeros?**

**Why weighted?**
- High weight where bootstrap uncertainty is low (confident predictions)
- Low weight where bootstrap uncertainty is high (noisy regions)
- Focuses learning on reliable spatial patterns

### Optimization
- **Optimizer**: AdamW
- **Learning rate**: 1e-4
- **Weight decay**: 1e-5
- **Scheduler**: ReduceLROnPlateau (patience=10, factor=0.5)
- **Gradient clipping**: max_norm=1.0

---

## Training Specifications

### Data Splits
**Temporal splits** (respecting time series structure):
```
Train:      Years 1-20  (1993-2014)  →  2000 samples (22 years × 100 bootstraps)
Validation: Years 21-24 (2015-2019)  →   400 samples (5 years × 100 bootstraps)
Test:       Years 25-28 (2021-2023)  →   400 samples (3 years × 100 bootstraps)
```

**Why temporal splits?**
- Prevents data leakage (same year shouldn't appear in train and test)
- Mimics real forecasting scenario
- Tests generalization to future years

### Training Hyperparameters
```python
batch_size = 32 [PLACEHOLDER]
max_epochs = 200 [PLACEHOLDER]
early_stopping_patience = 20 [PLACEHOLDER]
learning_rate = 1e-4 [PLACEHOLDER]
weight_decay = 1e-5 [PLACEHOLDER]
```

### Compute Requirements

#### 10×10 Grid (Prototyping)
- **Training time**: ~5 minutes/epoch (CPU)
- **Total training**: ~30 minutes (with early stopping)
- **Memory**: < 1 GB
- **Hardware**: Any laptop

#### 50×50 Grid (Production)
- **Training time**: ~2 hours/epoch (GPU) or ~8 hours/epoch (CPU)
- **Total training**: ~40 hours (GPU) or ~160 hours (CPU)
- **Memory**: ~4 GB GPU / ~8 GB RAM
- **Hardware**: Recommended GPU (CUDA 11.8+)

### Early Stopping
- **Metric**: Validation loss
- **Patience**: 20 epochs
- **Mode**: Minimize
- **Action**: Stop training, restore best weights

**Why early stopping?**
- Prevents overfitting to training set
- Saves computation time
- Objective stopping criterion (no manual intervention)

### Checkpointing
- **Save frequency**: Every epoch when validation loss improves
- **Save location**: `experiments/runs/{run_name}/checkpoints/`
- **Files saved**: 
  - `best_model.pth` (model weights)
  - `checkpoint_epoch_{n}.pth` (periodic backups)
  - `training_history.json` (loss curves)

---

## Pipeline Skeleton

### Phase 1: Data Generation (Week 1)

#### File: `generate_gmrf_dummy_data.py`

**Purpose**: Create synthetic spatiotemporal data with realistic correlation structure for rapid pipeline development without waiting for slow SPDE bootstrap.

**Functions**:

##### `create_spatial_precision_matrix(grid_size, kappa)`
- **Purpose**: Build spatial correlation structure (precision matrix Q)
- **What it does**: Creates sparse matrix encoding 4-neighbor relationships on 2D grid
- **Key parameter**: `kappa` controls correlation range (smaller = smoother fields)
- **Output**: Sparse matrix (grid_size² × grid_size²)
- **Why needed**: Makes neighboring grid cells correlated (realistic spatial patches)

##### `create_temporal_precision_matrix(n_years, rho, sigma)`
- **Purpose**: Build temporal correlation structure (AR(1) process)
- **What it does**: Encodes year-to-year persistence in crab populations
- **Key parameter**: `rho` = temporal correlation (0.7 = strong persistence)
- **Output**: Sparse matrix (n_years × n_years)
- **Why needed**: Populations don't reset each year; gradual evolution

##### `sample_gmrf(Q, n_samples)`
- **Purpose**: Draw random samples from GMRF by solving sparse linear system
- **What it does**: Converts precision matrix Q into actual spatial fields
- **Method**: Solves Q × X = Z where Z ~ N(0, I)
- **Output**: Array of correlated random fields
- **Why needed**: Q is recipe; sampling creates actual realizations (bootstrap variability)

##### `create_spatiotemporal_gmrf_data(grid_size, n_years, n_bootstraps, spatial_kappa, temporal_rho, mean_density, seed)`
- **Purpose**: Generate complete dataset with spatial + temporal correlation
- **What it does**: 
  - Combines spatial and temporal correlation
  - Generates multiple bootstrap replicates
  - Transforms to positive density scale
- **Key parameters**:
  - `spatial_kappa = 0.3`: ~60km correlation range
  - `temporal_rho = 0.7`: Strong persistence
  - `n_bootstraps = 100`: Uncertainty quantification
- **Output**: Array (n_years × n_bootstraps, grid_size, grid_size)
- **Why needed**: Single function creates entire training dataset

##### `create_spawner_recruit_pairs(grid_size, n_spawner_years, n_recruit_years, n_bootstraps, recruitment_correlation, ...)`
- **Purpose**: Generate correlated spawner and recruitment fields
- **What it does**: 
  - Creates spawner fields with spatiotemporal structure
  - Creates recruitment fields partially correlated with spawners
  - Uses formula: R = α×S + √(1-α²)×ε (ensures Corr(R,S) = α)
- **Key parameter**: `recruitment_correlation = 0.3` (moderate relationship)
- **Output**: Two arrays (spawners, recruits) + metadata
- **Why needed**: Tests if transformer can learn spawner→recruitment relationship

##### `visualize_gmrf_properties(spawners, recruits, params, output_dir)`
- **Purpose**: Create verification plots to check data quality
- **What it does**:
  - Spatial pattern maps (verify smoothness)
  - Temporal evolution plots (verify persistence)
  - Spawner-recruitment scatterplot (verify correlation)
- **Output**: Three PNG figures
- **Why needed**: Catch bugs before training; visual quality check

---

#### File: `verify_dummy_data.py`

**Purpose**: Automated verification that generated data has correct properties.

**Functions**:

##### `verify_shapes()`
- **Checks**: Array dimensions match specifications
- **Why**: Catch dimension mismatches before training

##### `verify_spatial_correlation()`
- **Checks**: Neighboring pixels are correlated (smooth fields)
- **Why**: Ensure spatial structure present

##### `verify_temporal_correlation()`
- **Checks**: Consecutive years are correlated (persistence)
- **Why**: Ensure temporal structure present

##### `verify_spawner_recruit_relationship()`
- **Checks**: Spawners and recruits are moderately correlated
- **Why**: Ensure learning signal exists

##### `verify_bootstrap_variability()`
- **Checks**: Bootstrap replicates differ from each other
- **Why**: Ensure uncertainty quantification works

**Output**: Console report + verification plots

---

#### File: `create_train_test_splits.py`

**Purpose**: Split data temporally to prevent leakage and enable proper evaluation.

**Functions**:

##### `create_temporal_splits(spawners, recruits, train_years, val_years)`
- **Purpose**: Split by year while keeping bootstraps together
- **What it does**:
  - First 22 years → training
  - Next 5 years → validation
  - Last 3 years → test
  - All 100 bootstraps stay in same split
- **Why temporal splits**: 
  - No leakage (same year not in train AND test)
  - Mimics real forecasting (predict future from past)
  - More realistic evaluation
- **Output**: Dictionary with 6 arrays (train/val/test × spawners/recruits)

##### `save_splits(splits, output_dir)`
- **Purpose**: Save splits as separate numpy files
- **Output**: 6 .npy files in splits directory
- **Why needed**: Training scripts load from disk (reproducible)

##### `load_splits(splits_dir)`
- **Purpose**: Convenience loader for all splits at once
- **Output**: Dictionary with all 6 arrays
- **Why needed**: One-line loading in training scripts

---

### Phase 2: Model Development (Week 1-2)

#### File: `src/models/transformer.py`

**Purpose**: Define Vision Transformer adapted for spatial regression.

**Classes & Functions**:

##### `PatchEmbedding`
- **Purpose**: Convert 50×50 grid into sequence of patch embeddings
- **What it does**: 
  - Divides grid into 5×5 patches (100 total)
  - Projects each patch to 128-dim vector
- **Input**: [B, 1, 50, 50]
- **Output**: [B, 100, 128]
- **Why needed**: Transformers operate on sequences, not 2D grids

##### `PositionalEncoding2D`
- **Purpose**: Add spatial position information to patches
- **Type**: Learnable parameters (not sinusoidal)
- **Shape**: [100, 128]
- **Why learnable**: Grid has irregular survey boundaries; learned positions adapt to domain shape

** QUESTION ** 
The grid is going to be masked so that there are nans outside of the survey area. Do I still need to have the positional encodings be learned and not sinusoidal?

##### `TransformerEncoderLayer`
- **Purpose**: Single transformer block (attention + MLP)
- **Components**:
  - Multi-head self-attention (8 heads)
  - Feed-forward MLP (128 → 512 → 128)
  - Layer normalization (pre-norm)
  - Residual connections
  - Dropout (0.1)
- **Why needed**: Core computation unit of transformer

##### `TransformerEncoder`
- **Purpose**: Stack multiple encoder layers
- **Layers**: 6
- **Total params**: ~1.2M
- **Why 6 layers**: Balance between capacity and training time

##### `SpatialDecoder`
- **Purpose**: Convert patch embeddings back to 50×50 grid
- **Architecture**:
  - Reshape [B, 100, 128] → [B, 128, 10, 10]
  - Conv transpose: upsample to [B, 64, 25, 25]
  - Conv transpose: upsample to [B, 1, 50, 50]
- **Activation**: None (direct density prediction)
- **Why needed**: Transform sequence back to spatial grid

##### `CrabTransformer` (Main Model)
- **Purpose**: Complete end-to-end architecture
- **Pipeline**: Input → Patches → Encoder → Decoder → Output
- **Methods**:
  - `forward(x)`: Full forward pass
  - `get_attention_maps(x)`: Extract attention for visualization
  - `count_parameters()`: Report model size
- **Why needed**: Combines all components into trainable model

---

#### File: `src/data/dataset.py`

**Purpose**: PyTorch dataset for efficient data loading during training.

**Classes & Functions**:

##### `CrabDataset`
- **Purpose**: Wrap numpy arrays in PyTorch Dataset interface
- **Features**:
  - Loads spawner/recruit pairs
  - Optional: bootstrap uncertainty weights
  - Optional: data augmentation (flips/rotations)
  - Normalizes to [0, 1] range -- Do we need to do this? might not work great with the ranges of the data. 
- **Methods**:
  - `__len__()`: Return dataset size
  - `__getitem__(idx)`: Return (spawner, recruit, uncertainty) tuple
  - `get_normalization_stats()`: Return mean/std for denormalization
- **Why needed**: PyTorch requires Dataset interface for DataLoader

##### `create_dataloaders(batch_size, num_workers, ...)`
- **Purpose**: Create train/val/test DataLoaders with proper settings
- **Settings**:
  - Train: shuffle=True
  - Val/Test: shuffle=False
  - Pin memory for GPU transfer
  - Multiple workers for parallel loading
- **Output**: Dictionary of DataLoaders
- **Why needed**: Efficient batching and GPU transfer

---

### Phase 3: Training (Week 2-3)

#### File: `src/training/train.py`

**Purpose**: Main training script with early stopping, checkpointing, and logging.

**Functions**:

##### `train_epoch(model, dataloader, optimizer, loss_fn, device)`
- **Purpose**: Execute one training epoch
- **Process**:
  1. Set model to train mode (dropout active)
  2. Loop over batches
  3. Forward pass
  4. Compute loss (weighted MSE)
  5. Backward pass (compute gradients)
  6. Gradient clipping (prevent explosion)
  7. Optimizer step (update weights)
- **Output**: Average epoch loss
- **Why needed**: Core training iteration

##### `validate_epoch(model, dataloader, loss_fn, device)`
- **Purpose**: Evaluate on validation set (no gradient computation)
- **Process**:
  1. Set model to eval mode (dropout off)
  2. Loop over validation batches
  3. Forward pass only
  4. Compute loss
  5. Accumulate metrics
- **Output**: Average validation loss + metrics
- **Why needed**: Monitor overfitting, trigger early stopping

##### `compute_metrics(predictions, targets)`
- **Purpose**: Calculate evaluation metrics
- **Metrics**:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² (coefficient of determination)
  - Spatial correlation
- **Output**: Dictionary of metrics
- **Why needed**: Quantify prediction quality

##### `save_checkpoint(model, optimizer, epoch, loss, path)`
- **Purpose**: Save model state for resumption or evaluation
- **Saves**:
  - Model weights (state_dict)
  - Optimizer state
  - Learning rate scheduler state
  - Training history (loss curves)
  - Epoch number
- **Why needed**: Resume training, prevent loss of progress, evaluation

##### `load_checkpoint(path, model, optimizer)`
- **Purpose**: Resume training from saved checkpoint
- **Restores**: All states from checkpoint file
- **Why needed**: Continue interrupted training

##### `EarlyStopping` (Class)
- **Purpose**: Monitor validation loss and stop when no improvement
- **Attributes**:
  - `patience = 20`: Epochs to wait without improvement
  - `best_loss`: Best validation loss seen
  - `counter`: Epochs since last improvement
- **Methods**:
  - `__call__(val_loss)`: Check current validation loss
  - `should_stop()`: Return boolean stop decision
- **Why needed**: Prevent overfitting, save computation time

##### `train_model(config)` (Main Function)
- **Purpose**: Orchestrate full training pipeline
- **Process**:
  1. Load train/val data
  2. Create model, optimizer, scheduler
  3. Initialize early stopping
  4. Training loop:
     - Train one epoch
     - Validate
     - Update learning rate (if plateau)
     - Save checkpoint if improved
     - Check early stopping
  5. Load best model (lowest validation loss)
  6. Return trained model + history
- **Output**: Trained model, training history
- **Why needed**: Single function runs entire training

---

#### File: `src/training/losses.py`

**Purpose**: Custom loss functions for spatial prediction tasks.

**Functions**:

##### `weighted_mse_loss(predictions, targets, uncertainty)`
- **Purpose**: MSE weighted by bootstrap uncertainty
- **Formula**: `Σ [w_i × (pred_i - target_i)²]` where `w_i = 1/(σ_i² + ε)`
- **Effect**: Penalizes errors more where uncertainty is low
- **Why needed**: Focus learning on reliable predictions

##### `spatial_correlation_loss(predictions, targets)`
- **Purpose**: Penalize lack of spatial structure
- **Method**: Compute correlation between predicted and true spatial patterns
- **Formula**: `loss = 1 - correlation`
- **Why needed**: Encourage spatially coherent predictions

##### `combined_loss(predictions, targets, weights)`
- **Purpose**: Weighted combination of multiple losses
- **Formula**: `α × MSE + β × spatial_loss`
- **Why needed**: Balance multiple objectives

---

### Phase 4: Evaluation (Week 3-4)

#### File: `src/training/evaluate.py`

**Purpose**: Comprehensive evaluation on test set with visualizations.

**Functions**:

##### `evaluate_model(model, test_loader, device)`
- **Purpose**: Run model on test set and compute all metrics
- **Process**:
  1. Load trained model
  2. Load test data
  3. Generate predictions (no gradients)
  4. Compute metrics
  5. Create visualizations
- **Output**: Dictionary of metrics + predictions array
- **Why needed**: Final unbiased evaluation

##### `compute_spatial_metrics(predictions, targets)`
- **Purpose**: Metrics specific to spatial predictions
- **Metrics**:
  - Spatial RMSE (per-pixel error map)
  - Spatial correlation (agreement in patterns)
  - Hotspot detection (precision/recall for high-density areas)
  - Spatial autocorrelation of errors (Moran's I)
- **Why needed**: Understand where/how model fails spatially

##### `compute_temporal_metrics(predictions, targets, years)`
- **Purpose**: Metrics for temporal prediction quality
- **Metrics**:
  - Time series correlation (per location)
  - Trend prediction accuracy
  - Interannual variability captured
- **Why needed**: Understand temporal generalization

##### `plot_predictions(predictions, targets, years, output_dir)`
- **Purpose**: Visualize predictions vs. ground truth
- **Creates**:
  - Side-by-side maps (predicted vs. actual)
  - Difference maps (prediction errors)
  - Scatterplots (predicted vs. actual densities)
  - Time series at key locations
- **Output**: Multiple PNG files
- **Why needed**: Visual interpretation of model performance

##### `plot_attention_maps(model, sample_input, output_dir)`
- **Purpose**: Visualize what transformer "looks at"
- **Creates**:
  - Attention weight heatmaps
  - Attention flow diagrams
  - Head-specific attention patterns
- **Why needed**: Interpret what model learned

##### `create_error_analysis(predictions, targets, output_dir)`
- **Purpose**: Deep dive into model mistakes
- **Analyzes**:
  - Where errors are largest (spatial patterns)
  - When errors are largest (temporal patterns)
  - Types of errors (over vs. underestimation)
- **Output**: Error analysis report + visualizations
- **Why needed**: Identify failure modes for improvement

---

#### File: `src/baselines/compare_methods.py`

**Purpose**: Compare transformer against EOF-GLLVM and other baselines.

**Functions**:

##### `load_baseline_predictions(baseline_dir)`
- **Purpose**: Load predictions from R-based baseline methods
- **Baselines**:
  - EOF-GLLVM (EFA)
  - EOF-GLLVM (CFA)
  - Traditional EOF + GAM
  - Spectral clustering + GAM
- **Output**: Dictionary of baseline predictions
- **Why needed**: Fair comparison on same test set

##### `compare_all_methods(transformer_preds, baseline_preds, targets)`
- **Purpose**: Compute metrics for all methods on same test set
- **Creates comparison table**:
  ```
  Method              | RMSE  | R²   | Spatial Corr
  --------------------|-------|------|-------------
  Transformer         | 12.3  | 0.72 | 0.81
  EOF-GLLVM (EFA)     | 15.7  | 0.64 | 0.76
  EOF-GLLVM (CFA)     | 14.2  | 0.68 | 0.78
  Traditional EOF+GAM | 18.1  | 0.52 | 0.69
  Spectral+GAM        | 16.9  | 0.58 | 0.71
  ```
- **Output**: Pandas DataFrame
- **Why needed**: Quantify transformer improvement

##### `plot_method_comparison(comparison_df, output_dir)`
- **Purpose**: Visualize performance differences
- **Creates**:
  - Bar charts of metrics
  - Prediction quality maps for each method
  - Error distribution comparisons
- **Why needed**: Visual communication of results

##### `statistical_significance_tests(transformer_errors, baseline_errors)`
- **Purpose**: Test if transformer improvement is statistically significant
- **Tests**:
  - Paired t-test on spatial errors
  - Wilcoxon signed-rank test
  - Diebold-Mariano test (forecast accuracy)
- **Output**: P-values and significance indicators
- **Why needed**: Support claim of improvement in thesis

---

### Phase 5: Real Data Pipeline (Week 4+)

#### File: `src/data_preparation/01_extract_station_data.R`

**Purpose**: Extract station data from R's survDAT and prepare for SPDE.

**Functions**:

##### `load_survey_data()`
- **Purpose**: Load NOAA bottom trawl survey data
- **Source**: `survDAT.RData`
- **Filters**: Opilio crab, years > 1987, standard hauls
- **Output**: Filtered survey dataframe
- **Why needed**: Starting point for all analyses

##### `calculate_spawner_density()`
- **Purpose**: Compute spawning stock biomass density at each station
- **Definition**: Female crabs with eggs (CLUTCH_SIZE > 0)
- **Formula**: `density = sum(SAMPLING_FACTOR) / (AREA_SWEPT × 3.4299)`
- **Output**: Station × Year matrix of spawner densities
- **Why needed**: Input to prediction models

##### `calculate_recruit_density()`
- **Purpose**: Compute recruitment density at each station
- **Definition**: Small crabs (44-56mm width, assumed age-5)
- **Formula**: Same as spawner
- **Output**: Station × Year matrix of recruit densities
- **Why needed**: Target for prediction models

##### `extract_station_locations()`
- **Purpose**: Get mean lat/lon for each station
- **Method**: Average across all years
- **Output**: Station × (lat, lon) dataframe
- **Why needed**: Spatial coordinates for SPDE mesh

##### `save_for_python()`
- **Purpose**: Export to CSV for Python pipeline
- **Saves**:
  - `spawner_station_data.csv`
  - `recruit_station_data.csv`
  - `station_locations.csv`
- **Why needed**: Bridge R and Python workflows

---

#### File: `src/data_preparation/03_run_spde_bootstrap.R`

**Purpose**: Run 100 bootstrap iterations of SPDE interpolation.

**⚠️ WARNING**: This is SLOW (2-5 days of computation)

**Functions**:

##### `create_mesh(station_locations, survey_domain)`
- **Purpose**: Build INLA triangular mesh over survey domain
- **Uses**: `fmesher::fm_mesh_2d()`
- **Parameters**: cutoff=40km, refine=TRUE
- **Output**: Mesh object with ~200 vertices
- **Why needed**: SPDE requires mesh for continuous field representation

##### `create_spde_model(mesh)`
- **Purpose**: Define SPDE model (Matérn covariance)
- **Uses**: `INLA::inla.spde2.matern()`
- **Parameters**: Matérn smoothness, range prior
- **Output**: SPDE model object
- **Why needed**: Statistical model for spatial correlation

##### `create_projection_matrices(mesh, stations, grid)`
- **Purpose**: Link mesh vertices to data stations and prediction grid
- **Creates**:
  - `A_is`: Observation projection (stations → mesh)
  - `A_gs`: Prediction projection (grid → mesh)
- **Why needed**: Map between discrete stations and continuous field

##### `run_single_bootstrap(data, mesh, spde, A_is, A_gs, seed)`
- **Purpose**: One bootstrap iteration
- **Process**:
  1. Sample stations with replacement
  2. Fit SPDE model to sampled data
  3. Predict on 50×50 grid using A_gs
  4. Return interpolated field
- **Output**: 50×50 numpy array
- **Time**: ~2-3 minutes per iteration
- **Why needed**: Each bootstrap = one uncertainty realization

##### `run_full_bootstrap(data, n_bootstraps=100, n_cores=16)`
- **Purpose**: Execute all 100 bootstrap iterations (parallelized)
- **Process**:
  1. Set up parallel cluster
  2. Loop over bootstrap iterations
  3. Each iteration runs SPDE
  4. Stack results into single array
  5. Save to disk
- **Output**: [n_years × n_bootstraps, 50, 50] array
- **Time**: 2-5 days on 16-core machine
- **Why needed**: Uncertainty quantification for predictions

##### `save_spde_output(spawners, recruits, output_dir)`
- **Purpose**: Save bootstrap results as numpy arrays
- **Saves**:
  - `data/spde_output/spawners_50x50.npy`
  - `data/spde_output/recruits_50x50.npy`
- **Why needed**: Python pipeline loads these files

---

#### File: `src/data_preparation/04_use_real_data.py`

**Purpose**: Replace dummy data with real SPDE output in training pipeline.

**Functions**:

##### `load_real_data(spde_dir)`
- **Purpose**: Load SPDE bootstrap results
- **Source**: `data/spde_output/*.npy`
- **Output**: Spawner and recruit arrays
- **Why needed**: Interface to real data

##### `validate_real_data(spawners, recruits)`
- **Purpose**: Check data quality before training
- **Checks**:
  - No NaN values (interpolation succeeded)
  - Positive densities only (no negative crabs)
  - Reasonable ranges (not extreme outliers)
  - Correct dimensions (matches specifications)
- **Output**: Boolean + error report
- **Why needed**: Catch data issues early

##### `prepare_for_training(spawners, recruits, output_dir)`
- **Purpose**: Format real data identically to dummy data
- **Process**:
  1. Load SPDE output
  2. Normalize to [0, 1] or standardize
  3. Create train/val/test splits (temporal)
  4. Save in standard format
- **Output**: Same structure as dummy data (seamless swap)
- **Why needed**: Training code works identically for dummy and real data

---

## 🎯 Workflow Summary

### Week 1: Dummy Data Pipeline
```bash
# 1. Generate dummy data (10×10 for fast testing)
python generate_gmrf_dummy_data.py --grid_size 10

# 2. Verify quality
python verify_gmrf_data.py

# 3. Create splits
python create_train_test_splits.py

# 4. Build transformer model (start simple)
# Implement src/models/transformer.py

# 5. Test on single batch (sanity check)
python test_single_batch.py
```

### Week 2: Training Pipeline
```bash
# 1. Train on dummy data (10×10 - fast iterations)
python src/training/train.py --grid_size 10 --max_epochs 50

# 2. Evaluate
python src/training/evaluate.py --checkpoint experiments/runs/run_001/best_model.pth

# 3. Generate 50×50 dummy data
python generate_gmrf_dummy_data.py --grid_size 50

# 4. Train on 50×50 (production scale)
python src/training/train.py --grid_size 50 --max_epochs 200

# 5. Benchmark training speed (decide CPU vs GPU)
python scripts/benchmark_training_speed.py
```

### Week 3: Real Data
```bash
# 1. Extract from R (run once)
Rscript src/data_preparation/01_extract_station_data.R

# 2. Run SPDE bootstrap (SLOW - 2-5 days, consider cluster)
Rscript src/data_preparation/03_run_spde_bootstrap.R

# 3. Prepare for training
python src/data_preparation/04_use_real_data.py

# 4. Train on real data
python src/training/train.py --use_real_data --max_epochs 200

# 5. Evaluate on real data
python src/training/evaluate.py --use_real_data
```

### Week 4: Comparison & Analysis
```bash
# 1. Run all baseline methods (in R)
Rscript src/baselines/run_all_baselines.R

# 2. Compare methods (generate thesis table)
python src/baselines/compare_methods.py

# 3. Statistical significance tests
python src/baselines/statistical_tests.py

# 4. Create all thesis figures
python src/visualization/create_thesis_figures.py

# 5. Generate LaTeX tables
python src/visualization/create_latex_tables.py
```

---

## Completion Checklist

### Data Pipeline
- [ ] GMRF dummy data (10×10) generates correctly -- check
- [ ] GMRF dummy data (50×50) generates correctly -- check
- [ ] Data verification passes all checks
- [ ] Train/val/test splits created -- check
- [ ] (Later) R data extraction completes
- [ ] (Later) SPDE bootstrap completes (2-5 days)
- [ ] (Later) Real data loads successfully

### Model Development
- [ ] `PatchEmbedding` implemented
- [ ] `PositionalEncoding2D` implemented
- [ ] `TransformerEncoderLayer` implemented
- [ ] `TransformerEncoder` implemented
- [ ] `SpatialDecoder` implemented
- [ ] `CrabTransformer` (full model) implemented
- [ ] Forward pass works (no errors)
- [ ] Attention extraction works
- [ ] Model fits on single batch (sanity check)
- [ ] Parameter count matches specification (~1.5M)

### Training Pipeline
- [ ] `train_epoch()` implemented
- [ ] `validate_epoch()` implemented
- [ ] `compute_metrics()` implemented
- [ ] `save_checkpoint()` implemented
- [ ] `load_checkpoint()` implemented
- [ ] `EarlyStopping` class implemented
- [ ] `train_model()` main function implemented
- [ ] Training loop runs without errors
- [ ] Early stopping triggers correctly
- [ ] Checkpoints save/load properly
- [ ] Loss decreases over epochs
- [ ] Validation loss computed correctly
- [ ] Training on 10×10 completes (~30 min)
- [ ] Training on 50×50 completes (~40 hours GPU)

### Evaluation
- [ ] `evaluate_model()` implemented
- [ ] `compute_spatial_metrics()` implemented
- [ ] `compute_temporal_metrics()` implemented
- [ ] `plot_predictions()` implemented
- [ ] `plot_attention_maps()` implemented
- [ ] `create_error_analysis()` implemented
- [ ] Metrics computed correctly
- [ ] Predictions visualized
- [ ] Attention maps generated
- [ ] Error analysis completed

### Baseline Comparison
- [ ] All baseline methods run in R
- [ ] `load_baseline_predictions()` implemented
- [ ] `compare_all_methods()` implemented
- [ ] `plot_method_comparison()` implemented
- [ ] `statistical_significance_tests()` implemented
- [ ] Comparison table generated
- [ ] Significance tests completed
- [ ] All comparison figures created

### Real Data Integration
- [ ] `01_extract_station_data.R` completes
- [ ] `03_run_spde_bootstrap.R` completes (SLOW)
- [ ] `04_use_real_data.py` works
- [ ] Real data validation passes
- [ ] Training on real data completes
- [ ] Evaluation on real data completes

### Thesis Deliverables
- [ ] All figures created (high-resolution)
- [ ] All tables created (LaTeX format)
- [ ] Comparison with all baselines
- [ ] Statistical significance documented
- [ ] Attention map interpretations
- [ ] Error analysis writeup
- [ ] Results reproducible (scripts + documentation)
- [ ] Code repository organized and documented

---

## 📝 Development Notes

### Key Design Decisions

**1. Why Vision Transformer?**
- No predetermined correlation structure (unlike EOF)
- Learns spatial relationships from data
- Self-attention can capture long-range dependencies
- State-of-art in computer vision, worth testing for ecology

**2. Why temporal splits instead of random?**
- Prevents leakage (same year in train and test)
- Mimics real forecasting scenario
- More conservative evaluation
- Better test of generalization

**3. Why weighted loss by uncertainty?**
- Bootstrap gives uncertainty estimate at each location
- Confident predictions should be more heavily penalized for errors
- Focuses learning on reliable patterns
- Prevents overfitting to noisy regions

**4. Why start with dummy data?**
- SPDE takes 2-5 days to run
- Dummy data generates in minutes
- Can test entire pipeline before committing to SPDE
- Faster iteration during development
- Separates concerns (transformer bugs vs SPDE issues)

**5. Why 10×10 before 50×50?**
- 10×10 trains in minutes, 50×50 takes hours/days
- Fast debugging and prototyping
- Same code works for both (just change grid_size)
- Verify pipeline works before scaling up

### Common Pitfalls to Avoid

1. **Don't use random splits** - breaks temporal structure
2. **Don't forget to normalize** - transformer sensitive to input scale
3. **Don't skip sanity checks** - overfit single batch first
4. **Don't ignore early stopping** - will overfit without it
5. **Don't train too long** - diminishing returns after convergence
6. **Don't compare on different test sets** - biases comparison
7. **Don't forget to set seeds** - needed for reproducibility

### Debugging Tips

**Model won't train (loss not decreasing)**:
- Check learning rate (try 1e-3, 1e-4, 1e-5)
- Verify data is normalized
- Check for NaN in data
- Overfit single batch first (should reach near-zero loss)
- Check gradient flow (use gradient clipping)

**Out of memory errors**:
- Reduce batch_size (try 16 or 8)
- Use smaller grid_size (10 instead of 50)
- Enable gradient checkpointing
- Use mixed precision training (fp16)

**Training too slow**:
- Use GPU if available
- Increase num_workers in DataLoader
- Reduce grid_size for prototyping
- Profile code to find bottlenecks

**Overfitting (train loss << val loss)**:
- Increase dropout (try 0.2 or 0.3)
- Add weight decay (try 1e-4)
- Reduce model size (fewer layers/dims)
- Get more training data (more years/bootstraps)
- Use data augmentation

---

## References

### Papers
- Vaswani et al. (2017) - Attention Is All You Need
- Dosovitskiy et al. (2021) - An Image is Worth 16×16 Words (Vision Transformer)
- Thorson et al. (2015) - Geostatistical delta-GLMM (VAST package)
- Lindgren et al. (2011) - SPDE approach for spatial modeling



---

## 📄 License

Academic research code. Please cite if you use this work.

---

**Last updated**: 2026-01-08

**Status**: Development phase - dummy data pipeline complete, data validation in progress

**Next steps**: Finish data validation, Build transformer model, test on 10×10 dummy data, then scale to 50×50