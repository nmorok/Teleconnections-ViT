# Methods

---

## Opening Paragraph

We developed a spatiotemporal deep learning pipeline to predict the spatial distribution of snow crab (*Chionoecetes opilio*) recruitment across the Eastern Bering Sea (EBS) from historical observations of spawner density, recruit density, and bottom temperature. The central prediction goal is to map the 50 × 50 grid of juvenile (recruit) density at year *t* from multi-channel spatial inputs encoding the recent history of spawner abundance, recruit abundance, and environmental conditions at every location. We evaluated the pipeline on both real survey data, which was augmented via spatial bootstrap resampling to address data scarcity, and three synthetic benchmark datasets generated under a range of spawner–recruit covariance structures, allowing controlled evaluation in settings where the ground-truth relationship is known. The core model is a Vision Transformer (ViT) architecture (Dosovitskiy et al., 2021), which we call the **[CrabTransformer]**, and is adapted for ecological spatiotemporal prediction. We chose to use a ViT architecture because its attention mechanism allows it to learn non-local dependencies between spatially distant regions of the EBS, making it suited to investigating potential teleconnections in the spawner–recruit system. We explored multiple forecasting scenarios to assess performance across conditions by varying the temporal information available to the model at prediction time, ranging from same-year nowcasting to multi-year-ahead forecasting.

---

## 3.1 Study Region and Survey Data

Snow crab density data were obtained from the NOAA Eastern Bering Sea (EBS) bottom trawl survey, which has been conducted annually since the mid-1970s at approximately 349 fixed stations arranged on a regular grid spanning the EBS shelf (Markowitz et al., 2024). Survey years with incomplete or absent coverage were excluded from the analysis; specifically, the 2020 survey year was omitted due to the suspension of field operations during the COVID-19 pandemic. The survey dataset covers the years 1988–2023 and contains catch information (including sex, size, and clutch data), as well as station-level location, bottom temperature, depth, and other abiotic factors. One 30-minute bottom trawl is conducted at each station during daylight hours on a consistent 20 × 20 nm systematic grid.

The dataset was divided into two biological groups used as prediction targets. Spawners were defined as mature female crabs, identified by the presence of eggs (clutch size > 0), which serves as a proxy for maturity in the absence of a direct maturity indicator. Recruits were defined as crabs with carapace width between 45–55 mm, assumed to represent a cohort approximately five years old [refs].

Bottom temperature data were obtained from the coldpool R package [refs], which provides interpolated rasters of EBS bottom temperature from 1982–2025 (excluding 2020). Temperature surfaces are produced by ordinary kriging of bottom temperatures recorded at hauls with confirmed good performance, using Stein's parameterization of the Matérn semivariogram. These rasters were projected onto the same 50 × 50 prediction grid described below.

Raw station-level density observations were projected onto a regular spatial grid of 50 × 50 cells, with each cell covering approximately 23 × 23 km (cell size = 25 km; grid extent: 50 columns × 35 rows over the EBS region, with the remaining rows padded to produce a square grid). Projection was carried out using a Stochastic Partial Differential Equation (SPDE) approach implemented in R-INLA (Lindgren et al., 2011), which approximates a Gaussian Markov Random Field (GMRF) and allows geostatistical interpolation of irregularly spaced point observations onto a regular lattice. A binary spatial mask, a matrix of zeros and ones identifying cells that fall within the EBS survey footprint, was constructed to exclude out-of-region cells from all model inputs and loss calculations; such cells are set to zero throughout.

---

## 3.2 Bootstrap Data Augmentation

A fundamental challenge in applying deep learning methods to fisheries survey data is data scarcity: even a long time series of annual surveys yields only on the order of 30–40 independent spatial snapshots, far fewer than typically required to train a neural network. To address this, we augmented the dataset using a spatial bootstrap resampling approach. For each of 100 bootstrap replicates, we drew a random subsample of 300 of the 349 survey stations (without replacement), fit an independent SPDE model to each year's data at those 300 stations, and projected predictions to the 50 × 50 grid. This procedure introduces realistic variability in spatial interpolation arising from measurement uncertainty and sampling variability, effectively producing 100 plausible realizations of the spatial density field for each survey year.

We note that while this augmentation increases the number of training samples from roughly 30 to 3,000, the 100 bootstrap replicates for any given year are highly correlated (Pearson *r* > 0.96), meaning the effective number of statistically independent training patterns remains approximately equal to the number of survey years (~24 after accounting for the train/validation/test split and the excluded 2020 observation). The implications of this fundamental data constraint for model performance are discussed in Section X.

---

## 3.3 Synthetic Benchmark Data

To validate the model pipeline and provide a controlled environment where the ground-truth spawner–recruit relationship is known, we also generated three simulated datasets from a Gaussian Markov Random Field (GMRF) model, extended to include temporal autocorrelation to emulate variations in biomass that correlate across space and time, and standardized to have a proportion of zero values that mirrored the observed data (~30%). Full GMRF simulation details are provided in Supplemental Section SX. Temperature channels were not simulated for the synthetic datasets; those input channels were set to zero and masked out during training and evaluation on synthetic data.

The three datasets were characterized by the magnitude and complexity of the covariance structure between spawner and recruit biomass. For the easy level, we set the spawner–recruit correlation to 0.9 with no temporal autocorrelation and no lag, representing a strong, contemporaneous relationship. For the medium level, we kept the spawner–recruit correlation at 0.9 but introduced a temporal autocorrelation of 0.7 and a three-year lag, meaning that recruit density in a given year was correlated with spawner density from three years prior. For the hard level, we decreased the spawner–recruit correlation to 0.8, retained temporal autocorrelation at 0.7, and increased the lag to five years. The key parameters for each level are summarized in Table S1.

> **[FLAG — Do I need to redo the synthetic data to include temperature?]:** 

---

## 3.4 Data Preprocessing and Temporal Splits

All density values (both real and synthetic) were log-transformed prior to input to the model using the transformation $x' = \log(1 + x)$, which compresses the heavy right tail of crab density distributions, reduces the influence of extreme values on gradient updates, and converts the multiplicative structure of count-like data to an approximately additive one. The spatial mask was applied prior to transformation, setting land and out-of-region cells to zero. Temperature values were not log-transformed because they can take negative values.

Data were split into training, validation, and test sets by temporal ordering. For the real EBS data, the 35 survey years (excluding 2020) were partitioned into 24 training years, 8 validation years, and 3 test years, preserving chronological order to prevent data leakage from future years into model training. The same chronological split was applied consistently across all 100 bootstrap replicates, so that each split contains the same calendar years across all replicates.

The 2020 survey gap was handled through a per-sample validity flag, which takes the value 0 for observations corresponding to 2020 and 1 otherwise. This flag is propagated into the loss calculation, ensuring that predictions for 2020 contribute zero weight to the training and validation objectives without requiring truncation of the time series.

---

## 3.5 CrabTransformer Architecture
 
The CrabTransformer is a Vision Transformer (ViT) that takes the 17-channel spatial input described in Section 3.6 and produces a 50 × 50 grid of predicted recruit density. Rather than processing the grid cell by cell, the model first divides the 50 × 50 input into 100 non-overlapping 5 × 5 spatial blocks (patches), each summarized by a learned vector representation that combines information from all 17 input channels. This step is analogous to aggregating point observations into spatial units before analysis: it makes the problem computationally tractable while preserving local spatial structure. Each channel:current spawner density, spawner history, recruit history, and bottom temperature history, is projected through its own independent learned transformation before the channel representations are combined, allowing the model to weight each type of input differently rather than treating all predictors identically. To ensure the model retains spatial and temporal awareness, each patch is injected with a learned spatial coordinate vector and a fixed mathematical fingerprint representing the specific prediction year before entering the attention layers.
 
The 100 patch representations are then passed through a series of self-attention layers, which form the core of the transformer architecture. Self-attention can be understood as a data-adaptive spatial weighting scheme: at each layer, every patch computes a weighted average of information from all other patches, where the weights are learned from the data and reflect how useful each spatial location is for predicting conditions at another. Unlike regression models in which spatial relationships must be specified in advance (e.g., through a covariance function or neighbourhood matrix), the transformer learns these relationships directly from the training data — including potentially long-range dependencies between distant regions of the EBS. After the attention layers, a convolutional decoder progressively upsamples the 10 × 10 patch grid back to the full 50 × 50 spatial resolution, producing the final recruitment prediction map. Full architectural details, including patch embedding, positional and temporal encoding, multi-head attention, the feed-forward sublayers, and the spatial decoder, are provided in Supplemental Section SX.



---

## 3.6 Input Representation and the Temporal Memory Bank

A central design question for predicting snow crab recruitment is how much temporal context to provide to the model. Spawner–recruit dynamics in EBS snow crab operate over multi-year lags reflecting maturation time [refs], and recruitment success may depend not only on current spawner abundance but on the history of spawning events and environmental conditions over preceding years. We therefore constructed a 17-channel input tensor for each prediction year *t*:

| Channels | Content |
|---|---|
| Channel 0 | Spawner density at year *t* |
| Channels 1–5 | Spawner density at years *t*−1 through *t*−5 |
| Channels 6–10 | Recruit density at years *t*−1 through *t*−5 |
| Channel 11 | Bottom temperature at year *t* |
| Channels 12–16 | Bottom temperature at years *t*−1 through *t*−5 |

This 17-channel representation provides the model with a memory bank of the recent history of both spawner abundance and environmental conditions at every location simultaneously. All channels are spatial grids of dimension 50 × 50.

For years early in the time series where fewer than five years of historical data exist, channels corresponding to unavailable years are filled with zeros and flagged as missing using a temporal mask vector of length 6 (one entry for the current year, five entries for the historical years), with entries set to 0 for missing years and 1 for valid years. This temporal mask is passed into the model and used to zero out contributions from missing channels before they can influence predictions, treating the zero-padded values as missing data rather than genuine zero-density or zero-temperature observations. A similar masking mechanism handles the 2020 gap when it falls within the historical window. Historical data from the final five years of each split were carried forward to provide context for the first years of the subsequent split.

### 3.6.1 Prediction Modes

We explored three prediction modes that differ in what information is available at prediction time. The modes are summarized in Table X; the key distinction is whether the current-year spawner density and temperature (Channel 0 and Channel 11) are available to the model.

**Table X. Input channels available under each prediction mode.** A checkmark indicates the channel is provided; a dash indicates it is replaced with zeros and masked out. The recruit history channels contain different years depending on the lag applied.

| Channel | Content | Same-year | One-year-ahead | Lagged (*L* years) |
|---|---|---|---|---|
| 0 | Spawner (*t*) | ✓ | — | ✓ |
| 1–5 | Spawner (*t*−1 to *t*−5) | ✓ | ✓ | ✓ |
| 6–10 | Recruit history | *t*−1 to *t*−5 | *t*−1 to *t*−5 | *t*−1+L to *t*−5+L |
| 11 | Temperature (*t*) | ✓ | — | ✓ |
| 12–16 | Temperature (*t*−1 to *t*−5) | ✓ | ✓ | ✓ |
| **Target** | | Recruit (*t*) | Recruit (*t*) | Recruit (*t*+L) |

In **same-year mode** (the default), the model receives all 17 channels including the current-year spawner density and temperature (Channels 0 and 11), and predicts recruit density at year *t*.

In **one-year-ahead mode**, Channels 0 and 11 are replaced with zero grids and their temporal mask entries are set to 0, so the model must predict recruitment at year *t* using only spawner, recruit, and temperature data from years *t*−1 through *t*−5. This mode corresponds to a genuine forecasting scenario in which the survey for year *t* has not yet occurred, closely mirroring operational contexts where managers need forward-looking recruitment predictions.

In **lagged mode**, recruit observations are temporally offset by *L* years so that the model is trained to predict recruitment *L* years in advance. The spawner and temperature channels remain anchored at year *t*, while the recruit history channels contain densities from years *t*−1+*L* through *t*−5+*L* — that is, recruit observations that will have been collected by the time the prediction is needed. For example, with *L* = 5 and a target year of 2005: Channel 0 contains spawner density in 2000, Channels 1–5 contain spawners from 1999–1995, Channels 6–10 contain recruits from 2004–2000 (already observed by 2005), and the target is recruit density in 2005. This corresponds to the hypothesis that the crabs we defined as recruits are considered 5 years old when they show up in the survey. Applying a lag of L years therefore corrects the temporal mismatch between the spawning event and the observation of the resulting cohort in the survey.
---


## 3.7 Loss Functions

We trained models using two loss functions to assess sensitivity to this choice.

### 3.7.1 MSE Loss

The primary loss function was Mean Squared Error (MSE), operating directly on the log-transformed predictions and targets. Because the mean of log(*x*) is not the same as the log of the mean of *x* (Jensen's inequality), predictions were corrected for lognormal bias at evaluation time using the factor $\exp(\hat{\mu} + \hat{\sigma}^2 / 2) - 1$, where $\hat{\mu}$ and $\hat{\sigma}^2$ are the mean and variance of the training residuals in log space, so that the back-transformed prediction approximates the mean density on the original scale rather than the median. The per-pixel losses are averaged over valid (in-region) pixels within each sample, producing a per-sample scalar loss, which is then averaged over valid samples (excluding 2020) within each batch.

### 3.7.2 Tweedie Loss

We also trained models using the Tweedie deviance, which is specifically suited to non-negative continuous data with a high proportion of zeros. The Tweedie distribution with power parameter 1 < *p* < 2 corresponds to a compound Poisson-Gamma distribution, placing a point mass at zero combined with a continuous distribution for positive densities. We used a power parameter of 1.2. Because the model predicts in log space, predictions are back-transformed via $\hat{y}' = e^{\hat{y}} - 1$ before computing the Tweedie loss, so the loss operates on the original density scale.

---

## 3.8 Model Training

### 3.8.1 Weight Initialization and Bias Warm-Start

All linear layers were initialized using Xavier uniform initialization, which scales initial weights to maintain approximately constant gradient variance through the forward and backward passes. Convolutional layers in the decoder were initialized with Kaiming (He) normal initialization, which compensates for the approximate halving of signal variance that occurs under ReLU-family activations (GELU in our case). Normalization layer scale and shift parameters were initialized to 1 and 0, corresponding to an identity transformation at the start of training.

To prevent the model from predicting near-zero outputs regardless of input early in training, the bias of the final decoder output convolution was initialized to the mean log-transformed recruit density over all valid (non-masked, non-2020) training pixels.

### 3.8.2 Optimizer and Learning Rate Schedule

Model parameters were updated using AdamW (Loshchilov & Hutter, 2019), which incorporates decoupled weight decay (λ = 10⁻⁴) as a true L2 penalty on parameter magnitudes. The base learning rate was set to 10⁻⁴. We employed the OneCycleLR scheduler, which ramps the learning rate from an initial value of ~1.2 × 10⁻⁵ to a maximum of 3 × 10⁻⁴ over the first 30% of training steps, then anneals to a final rate of 3 × 10⁻⁷ over the remaining 70%. This warm-up and decay schedule helps the model establish productive gradient directions before taking large update steps, then fine-tune near a minimum. Gradient clipping (max norm = 1.0) was applied after each backward pass.

### 3.8.3 Model Selection

Training ran for 20 epochs. Validation loss was evaluated after each epoch and the checkpoint with the lowest validation loss was retained for evaluation on the held-out test set.

---

## 3.9 Model Configurations

Because the effective number of independent training patterns is small (~24 spatial configurations), we trained two model configurations as a sensitivity analysis:

| Configuration | embed_dim | Heads | Layers | d_ff | Dropout | Approx. Parameters |
|---|---|---|---|---|---|---|
| Full | 128 | 8 | 6 | 512 | 0.1 | ~1.3M |
| Reduced | 128 | 4 | 3 | 512 | 0.2 | ~500K |

Both configurations were trained on the same data splits and evaluated using the same metrics. Given the fundamental data constraint described in Section 3.2, we expect both configurations to show some training/validation loss divergence, and we report results from both to assess overfitting behaviour across model sizes.

---

## 3.10 Model Evaluation and Interpretability

Model predictions were evaluated against held-out observations using a comprehensive set of per-sample metrics, where each sample corresponds to a full 50 × 50 spatial recruitment field for a single bootstrap replicate and year. All metrics were computed over valid (in-region, non-masked) cells only.

Given that models struggle to predict true zero outputs, we set all of the values below the minimum observed crab density to zero. 

Pointwise accuracy was assessed using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE), both in log space. To decompose sources of prediction error, RMSE was partitioned into a systematic bias component and a spatial pattern component (unbiased RMSE), following Murphy (1988). Spatial pattern accuracy was assessed using Pearson and Spearman rank correlations between predicted and observed fields; Spearman correlation is distribution-free and robust to scale differences, making it appropriate when predictions may be biased in magnitude but correct in rank order. Overall abundance accuracy was quantified as the percentage error in total predicted density summed over valid cells: (Σŷ − Σy) / Σy × 100.

Two additional metrics assessed spatial structure specifically. Top-10% spatial accuracy measured the Jaccard overlap between the top decile of observed and predicted densities, capturing whether the model correctly identifies high-recruitment hotspots regardless of absolute scale. Zero-cell classification performance was assessed using precision, recall, and F1 score for identifying cells below a density threshold of 14.23 (the 10th percentile of non-zero observed training densities).

Finally, overall predictive skill was assessed relative to a climatological mean baseline — a model that always predicts the per-cell mean training density regardless of input. Positive skill scores (both MSE-based and MAE-based) indicate that the transformer predictions contain useful signal beyond simply predicting average conditions.

### 3.11 Post-hoc Attribution via Integrated Gradients

To identify which spatial regions and input channels most strongly drive recruitment predictions, we applied Integrated Gradients (IG; Sundararajan et al., 2017), a post-hoc attribution method satisfying the completeness axiom: the sum of all input attributions equals exactly the difference between the model output for the actual input and a chosen baseline. The reference input was set to the mean of all valid (non-2020) training samples, representing an average year across all bootstraps.

For each year, IG was computed independently for each of the 100 bootstrap replicates and then averaged. For a single sample, a path of 50 interpolated inputs was constructed between the baseline and the actual input, gradients of the total predicted density (summed over valid cells) were computed via backpropagation at each interpolated input, and the average gradient was multiplied by the input difference:

$$\text{IG}(x) = (x - x_\text{base}) \cdot \frac{1}{m}\sum_{\alpha=1}^{m} \nabla_x f\!\left(x_\text{base} + \frac{\alpha}{m}(x - x_\text{base})\right)$$

Attribution maps were visualized alongside the corresponding mean input fields for all 17 channels. An aggregated panel summing absolute attribution across all channels was also produced to highlight spatially influential regions regardless of which channel drove the effect, allowing inspection of whether the model relies on biologically plausible spatial structure.


---
 
## Supplemental
 
### SX.1 GMRF Simulation Details
 
To generate synthetic benchmark data, we used a Gaussian Markov Random Field (GMRF) model, a spatially correlated random field in which the precision matrix (the inverse of the covariance matrix) is sparse, encoding the property that each cell is conditionally independent of all other cells given only its immediate neighbours. A sparse precision matrix **Q** was constructed on the 50 × 50 grid with diagonal entries equal to κ² × (number of neighbours) and off-diagonal entries of −κ² for neighbouring cells, where κ controls the spatial range of correlation. A spatial field was sampled by solving **Qx** = **z** for a standard normal random vector **z**. Temporal autocorrelation was introduced by blending the field at each time step with the previous year's field (coefficient ρ), preserving variance. Synthetic density values were obtained by standardizing and exponentially transforming the GMRF samples, scaling to target mean densities, and applying a threshold to produce approximately 30% zeros. Spawner–recruit correlations were introduced by blending the recruit field with an optionally lagged copy of the spawner field.
 
**Table S1: GMRF simulation parameters for each difficulty level.**
 
| Parameter | Easy | Medium | Hard |
|---|---|---|---|
| grid_size | 50 | 50 | 50 |
| n_years | 30 | 30 | 30 |
| n_bootstraps | 100 | 100 | 100 |
| spatial_kappa (κ) | 0.3 | 0.3 | 0.3 |
| temporal_rho (ρ) | 0.0 | 0.7 | 0.7 |
| recruitment_correlation | 0.9 | 0.9 | 0.8 |
| lag (years) | 0 | 3 | 5 |
| mean_spawner_density | 3728 | 3728 | 3728 |
| mean_recruit_density | 4514 | 4514 | 4514 |
| seed | 2026 | 2026 | 2026 |
 
---
 
### SX.2 CrabTransformer Architecture Details
 
#### SX.2.1 Patch Embedding
 
Rather than treating each of the 2,500 grid cells individually (which would create an intractably large sequence for attention computation), we divide the 50 × 50 grid into non-overlapping 5 × 5 patches, yielding 100 patches arranged in a 10 × 10 grid. This is analogous to grouping nearby grid cells into spatial blocks before analysis, similar to spatial aggregation in geostatistical or species distribution modelling.
 
Each patch contains 5 × 5 = 25 values per channel, giving a raw vector of 25 values per channel per patch. Each of the 17 input channels is processed by its own independent learned linear projection, mapping the 25-element raw patch vector for that channel to a per-channel embedding vector. Using separate projections for each channel allows the model to learn a different representation for current spawners, spawner history, recruit history, and temperature, as these channels have fundamentally different ecological meanings. The 17 per-channel embeddings are then concatenated and passed through a final linear projection to produce a 128-element patch embedding. This two-stage process (project-then-combine) is analogous to constructing separate regression models for each predictor type and then combining their outputs in a meta-model.
 
Temporal masking is applied at this stage: for any channel flagged as missing (mask value = 0), the corresponding channel embedding is multiplied by zero before concatenation, ensuring that missing channels contribute nothing to the combined patch representation. After per-channel projection, each embedding is normalized using Layer Normalization, and dropout regularization (p = 0.1 in the full model; p = 0.2 in the reduced model) is applied to reduce overfitting.
 
#### SX.2.2 Positional and Temporal Encoding
 
After patch embedding, two types of contextual information are added to each patch's embedding vector.
 
**Spatial (positional) encoding.** Each of the 100 patch positions is assigned a learned position vector of length 128, initialized from a zero-mean normal distribution scaled by 0.1. These position vectors are added to the patch embeddings, allowing the model to distinguish patches from different grid locations, analogous to including spatial coordinates as covariates in a regression.
 
**Temporal encoding.** Each prediction year is assigned a fixed sinusoidal encoding based on the year index:
 
$$\text{PE}(t, 2i) = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \quad \text{PE}(t, 2i+1) = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$
 
where *t* is the year index, *d* = 128 is the embedding dimension, and *i* indexes pairs of embedding dimensions. This encoding provides a unique temporal fingerprint for each year that encodes the relative distance between years without requiring the model to learn year-specific patterns, and allows the model to generalize to years not seen during training.
 
#### SX.2.3 Multi-Head Self-Attention
Self-attention can be understood as a data-adaptive weighting scheme: for each patch, it computes a weighted average of all other patches' representations, where the weights reflect how 'relevant' each other patch is to the current one. High attention weights between two patches indicate that information at one location is useful for predicting the representation at the other. Formally, each patch embedding is projected through three learned matrices to produce a query (Q), key (K), and value (V) vector. The attention weight between patch *i* and patch *j* is the scaled dot product between their query and key vectors, passed through a softmax:
 
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_{\text{head}}}}\right) V$$
 
The scaling by $\sqrt{d_{\text{head}}}$ stabilizes the dot products. This process is conducted in multiple heads simultaneously (8 in the full model; 4 in the reduced model), where each head learns its own Q, K, V projections and can attend to different aspects of the spatial pattern independently. Outputs of all heads are concatenated and passed through a final linear projection.
 
#### SX.2.4 Feed-Forward Network and Transformer Blocks
 
Following each self-attention operation, the updated patch embeddings are passed through a position-wise feed-forward network: a two-layer perceptron with GELU activation that expands the embedding from 128 to 512 dimensions and contracts it back to 128. Self-attention and the feed-forward network are combined into a Transformer Block using residual (skip) connections and Layer Normalization (pre-LN variant):
 
$$x \leftarrow x + \text{Attention}(\text{LayerNorm}(x))$$
$$x \leftarrow x + \text{FFN}(\text{LayerNorm}(x))$$
 
Residual connections ensure information can flow directly through the network, stabilizing training in deep networks. The CrabTransformer stacks this block *L* times (L = 6 in the full model; L = 3 in the reduced model), with each layer refining the spatial representations produced by the previous one.
 
#### SX.2.5 Spatial Decoder
 
After the transformer blocks, the 100 patch embeddings are reshaped into a 10 × 10 × 128 feature map and passed through a convolutional decoder that progressively upsamples to the full 50 × 50 output resolution. The decoder proceeds through three upsampling stages: bilinear interpolation from 10 × 10 to 20 × 20, then 40 × 40, and finally to exactly 50 × 50. Each stage is followed by a 3 × 3 convolutional layer (with reflection padding to avoid edge artefacts), Group Normalization (which normalizes within groups of channels, providing stable normalization with small batch sizes), and GELU activation. Channel depth is reduced from 128 to 64 to 32 to 16 across these stages, and a final 3 × 3 convolution produces the single-channel 50 × 50 output.
 
#### SX.2.6 Output Activation and Spatial Masking
 
Crab densities are non-negative by definition. To enforce this constraint, the decoder output is passed through a softplus activation function, $f(x) = \log(1 + e^x)$, which is a smooth approximation to ReLU that maps the real line to strictly positive values without an upper bound. The spatial mask is applied as a final step, setting all out-of-region cells to exactly zero.

