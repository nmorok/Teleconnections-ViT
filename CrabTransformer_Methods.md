# Methods

## 3.1 Study Region and Survey Data

Snow crab (*Chionoecetes opilio*) density data were obtained from the NOAA Eastern Bering Sea (EBS) bottom trawl survey, which has been conducted annually since the mid-1970s at approximately 349 fixed stations arranged on a regular grid spanning the EBS shelf. Survey years with incomplete or absent coverage were excluded from the analysis; specifically, the 2020 survey year was omitted due to the suspension of field operations during the COVID-19 pandemic. For each survey year, we work with two separate spatial fields: adult female (spawner) density and juvenile (recruit) density, treated as distinct input channels to the model.

Raw station-level density observations were projected onto a regular spatial grid of 50 × 50 cells, with each cell covering approximately 23 × 23 km (cell size = 25 km, grid extent: 50 columns × 35 rows over the EBS region, with the remaining rows padded to produce a square grid). Prediction to the grid was carried out using a Stochastic Partial Differential Equation (SPDE) approach implemented in R-INLA (Lindgren et al., 2011), which approximates a Gaussian Markov Random Field (GMRF) and allows geostatistical interpolation of irregularly spaced point observations onto a regular lattice. A binary spatial mask was constructed to identify cells falling within the EBS survey region; cells outside this region are set to zero throughout and excluded from all loss calculations.

## 3.2 Bootstrap Data Augmentation

A fundamental challenge in applying deep learning methods to fisheries survey data is data scarcity: even a long time series of annual surveys yields only on the order of 30-40 independent spatial snapshots, far fewer than typically required to train a neural network. To address this, we augmented the dataset using a spatial bootstrap resampling approach. For each of 100 bootstrap replicates, a random subsample of 300 of the 349 survey stations was drawn (without replacement), and an SPDE model was independently fitted to this subsample and projected to the 50 × 50 prediction grid. This procedure introduces realistic variability in spatial interpolation arising from measurement uncertainty and sampling variability, effectively producing 100 plausible realizations of the spatial density field for each survey year. We note that while this augmentation increases the number of training samples from roughly 30 to 3,000, the 100 bootstrap replicates for any given year are highly correlated (Pearson *r* > 0.96), meaning the effective number of statistically independent training patterns remains approximately equal to the number of survey years (~24 after accounting for the train/validation/test split and the excluded 2020 observation). The implications of this fundamental data constraint for model performance are discussed in Section X.

## 3.3 Synthetic Benchmark Data

To validate the model pipeline and provide a controlled environment where the ground-truth spawner–recruit relationship is known, we also generated synthetic data from a Gaussian Markov Random Field (GMRF) model. A GMRF is a spatially correlated random field in which the precision matrix (the inverse of the covariance matrix) is sparse, encoding the property that each cell is conditionally independent of all other cells given only its immediate neighbours. This produces smooth, spatially autocorrelated fields similar in character to observed crab density patterns. A sparse precision matrix **Q** was constructed on the 50 × 50 grid with diagonal entries equal to κ² × (number of neighbours) and off-diagonal entries of −κ² for neighbouring cells, where κ controls the spatial range of correlation. A spatial field was then sampled by solving **Qx** = **z** for a standard normal random vector **z**. Optional temporal autocorrelation was introduced by blending the field at each time step with the previous year's field (with coefficient ρ), while preserving variance.

Synthetic density values were obtained by standardizing the GMRF samples and applying an exponential transformation, then scaling to target mean densities and applying a threshold to produce a realistic proportion of zero values that mirrors the observed data (~30%). Spawner–recruit correlations were introduced by blending the recruit field with an optionally lagged copy of the spawner field, with a blending coefficient equal to the desired recruitment correlation. Three difficulty levels were simulated, spanning conditions from a strong, zero-lag spawner–recruit relationship (easy) to a weak, five-year-lagged relationship with substantial temporal autocorrelation (hard). Full GMRF parameters for each difficulty level are provided in Table S1.

## 3.4 Data Preprocessing and Temporal Splits

All density values (both real and synthetic) were log-transformed prior to input to the model using the transformation $x' = \log(1 + x)$, which compresses the heavy right tail of crab density distributions (many near-zero values, occasional very high densities), reduces the influence of extreme values on gradient updates during training, and converts the multiplicative structure of count-like data to an approximately additive one. The spatial mask was applied prior to transformation, setting land and out-of-region cells to zero.

Data were split into training, validation, and test sets by temporal ordering. For the real EBS data, the 35 survey years (excluding 2020) were partitioned into 24 training years, 8 validation years, and 3 test years, preserving chronological order to prevent data leakage from future years into model training. This is important because any shuffling of the time series would allow the model to implicitly learn future conditions during training, producing optimistically biased validation performance. The same chronological split was applied consistently across all 100 bootstrap replicates, so that each split contains the same calendar years across all replicates.

The 2020 survey gap was handled through a per-sample validity flag (`valid_year`), which takes the value 0 for observations corresponding to 2020 and 1 otherwise. This flag is propagated into the loss calculation, ensuring that predictions for 2020 contribute zero weight to the training and validation objectives, rather than being treated as valid observations or excluded by truncating the time series (which would break the temporal continuity of the historical memory bank described below).

## 3.5 Input Representation and the Temporal Memory Bank

A central design question for predicting snow crab recruitment from spawner density is how much temporal context to provide to the model. Spawner-recruit dynamics in EBS snow crab operate over multi-year lags reflecting maturation time, and recruitment success may depend not only on the current spawner abundance but on the history of spawning events over the preceding several years. We therefore constructed an 11-channel input tensor for each prediction year *t*, organized as follows:

| Channels | Content |
|---|---|
| Channel 0 | Spawner density at year *t* |
| Channels 1–5 | Spawner density at years *t*−1 through *t*−5 |
| Channels 6–10 | Recruit density at years *t*−1 through *t*−5 |

This 11-channel representation can be thought of as providing the model with a "memory bank" of the recent history of both spawners and recruits at every location simultaneously. All channels are spatial grids of dimension 50 × 50.

For years early in the time series (where fewer than five years of historical data exist), channels corresponding to unavailable years are filled with zeros and flagged as missing using a temporal mask vector of length 6 (one entry for the current-year spawner, five entries for the historical years), with entries set to 0 for missing years and 1 for valid years. This mask is passed into the model and used to zero out the contributions of missing channels before they can influence predictions, rather than treating the zero-padded values as genuine zero-density observations. A similar masking mechanism handles the 2020 gap when it falls within the historical window.

To maintain the integrity of the historical memory bank across the train/validation/test boundary, the last five years of the training set are stored and passed as historical context to the validation dataset, and similarly the last five years of the validation set are passed to the test dataset. This ensures that predictions in the first years of each split have access to real historical data from the preceding split rather than zero padding.

### 3.5.1 One-Year-Ahead Prediction Mode

The pipeline supports three prediction modes that differ in what information is available at prediction time. In the **same-year mode** (the default), the model receives all 11 channels including the current-year spawner density (Channel 0). In the **one-year-ahead mode**, Channel 0 is replaced with a zero grid and its temporal mask entry is set to 0, so the model must predict recruitment at year *t* using only spawner and recruit data from years *t*−1 through *t*−5. This mode corresponds to a genuine forecasting scenario in which the survey for year *t* has not yet occurred, and closely mirrors the operational context for fisheries stock assessment where managers need forward-looking recruitment predictions. A second **one-year-ahead-lagged mode** was added to compare model performance. In this method, we lag the recruit data with the spawner data, and keep all of the channels like in the **same-year mode**. Channel 0 is the current year spawner, channels 1-5 are the *t*−1 through *t*−5 spanwers, and channels 6-10 are the *t*−1+lag through *t*−5+lag. So if we we thought recruitment takes 5 years to show up in the survey, then we would have lag of 5 years. Our channel set up would look like: for prediction in 2000. Channel 0: spawners in 2000, Channels 1-5: spawners in 1999-1995, Channels 6-10: recruits in 2004-2000, and our target would be recruits in 2005. 

In one-year-ahead mode, a sample is treated as a valid training observation provided at least one year of historical context is available (i.e., the temporal mask sum is greater than zero). Samples from the very first year of the training set, which have no historical data whatsoever (temporal mask all zeros), are excluded from the loss. Samples from year two onward—even if only one or two years of history are present—are included, allowing the model to implicitly learn to make predictions under varying degrees of historical context.

<span style="color:red"> Look into adding a temperature channel. The one-year-ahead-lagged mode can handle making predictions with it. <span>

## 3.6 CrabTransformer Architecture

We developed a Vision Transformer (ViT) model, which we call the **CrabTransformer**, to predict the 50 × 50 grid of recruit density from the 11-channel input tensor described above. The transformer architecture was originally developed for natural language processing (Vaswani et al., 2017) and subsequently adapted for image analysis (Dosovitskiy et al., 2021). Its central mechanism, self-attention, allows every location in a spatial domain to directly influence, and be influenced by, every other location simultaneously, without the locality constraints of traditional convolutional neural networks. This property is particularly well-suited to spatiotemporal teleconnection problems, where recruitment at one region of the EBS may be predicted by spawner densities at spatially distant locations. Below we describe each architectural component in terms accessible to ecologists familiar with regression and multivariate statistics.

### 3.6.1 Patch Embedding

Rather than treating each of the 2,500 grid cells individually (which would create an intractably large sequence for attention computation), we divide the 50 × 50 grid into non-overlapping 5 × 5 patches, yielding 100 patches arranged in a 10 × 10 grid. This is analogous to grouping nearby grid cells into spatial blocks before analysis, similar to the spatial aggregation often used in geostatistical or species distribution modelling to manage computation.

Each patch contains 5 × 5 = 25 pixel values per channel, giving a raw vector of 25 values per channel per patch. Critically, each of the 11 input channels is processed by its own independent learned linear projection (a matrix multiplication), mapping the 25-element raw patch vector for that channel to a 12-element embedding vector. Using separate projections for each channel rather than forcing a single projection to handle all channel types allows the model to learn a different representation for current spawners, for spawner history, and for recruit history, as these channels have fundamentally different ecological meanings. The 11 per-channel embeddings are then concatenated into a 132-element vector and passed through a final linear projection to produce a 128-element patch embedding. This two-stage process (project-then-combine) can be compared to constructing separate regression models for each predictor type and then combining their outputs in a meta-model.

Temporal masking is applied at this stage: for any channel flagged as missing (mask value = 0), the corresponding channel embedding vector is multiplied by zero before concatenation, ensuring that missing channels contribute nothing to the combined patch representation.

After per-channel projection, the embedding for each channel is normalized using Layer Normalization, and a small amount of dropout regularization (p = 0.1 in the full model; p = 0.2 in the reduced model) is applied to reduce overfitting.

### 3.6.2 Positional and Temporal Encoding

After patch embedding, two types of contextual information are added to each patch's embedding vector.

**Spatial (positional) encoding.** Each of the 100 patch positions is assigned a learned position vector of length 128, initialized from a zero-mean normal distribution scaled by 0.1. These position vectors are added to the patch embeddings, allowing the model to distinguish patches from different locations on the grid, analogous to including spatial coordinates as covariates in a regression. Unlike fixed sinusoidal positional encodings common in natural language applications, learned positional encodings allow the model to discover whatever spatial structure is most useful for the prediction task.

**Temporal encoding.** Each prediction year is assigned a fixed (non-learned) sinusoidal encoding based on the year index, computed as:

$$\text{PE}(t, 2i) = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \quad \text{PE}(t, 2i+1) = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

where *t* is the year index, *d* = 128 is the embedding dimension, and *i* indexes pairs of embedding dimensions. This encoding provides a unique temporal "fingerprint" for each year that encodes the relative distance between years mathematically (Year 5 is closer to Year 6 than to Year 20 in this representation) without requiring the model to explicitly learn year-specific patterns. Using fixed rather than learned temporal encodings also allows the model to generalize to years not seen during training, which is important for forecasting beyond the observed record.

### 3.6.3 Multi-Head Self-Attention

The core of the transformer is the self-attention mechanism, applied over the sequence of 100 patch embeddings. Self-attention can be understood as a data-adaptive weighting scheme: for each patch, it computes a weighted average of all other patches' representations, where the weights are determined by how "relevant" each other patch is to the current one. High attention weights between two patches indicate that information in one location is useful for predicting the representation at the other.

Formally, each patch embedding is projected through three learned matrices to produce a **query** (Q), a **key** (K), and a **value** (V) vector. The attention weight between patch *i* and patch *j* is computed as the scaled dot product between their query and key vectors, passed through a softmax function:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_{\text{head}}}}\right) V$$

The scaling by $\sqrt{d_{\text{head}}}$ prevents the dot products from becoming very large (which would push softmax weights toward zero or one), and corresponds to a variance-stabilization step. Patches whose temporal mask indicates missing data are masked out before the softmax (their scores are set to −10⁹) so they cannot receive attention weight.

This process is conducted in **multiple heads** simultaneously (8 heads in the full model; 4 in the reduced model), where each head learns its own Q, K, V projection matrices and can therefore attend to different aspects of the spatial pattern independently. The outputs of all heads are concatenated and passed through a final linear projection. 

### 3.6.4 Feed-Forward Network and Transformer Blocks

Following each self-attention operation, the updated patch embeddings are passed through a position-wise feed-forward network (FFN): a two-layer perceptron with a GELU activation that expands the embedding from 128 to 512 dimensions and then contracts it back to 128. The expansion and contraction allow the network to apply non-linear transformations independently to each patch's representation after information has been mixed across patches by attention.

Self-attention and the FFN are combined into a **Transformer Block** using residual (skip) connections and Layer Normalization (pre-LN variant):

$$x \leftarrow x + \text{Attention}(\text{LayerNorm}(x))$$
$$x \leftarrow x + \text{FFN}(\text{LayerNorm}(x))$$

The residual connections ensure that information can flow directly through the network without passing through every transformation, which stabilizes training in deep networks and ensures that early layers do not lose information before later layers can use it. The CrabTransformer stacks this block *L* times (L = 6 in the full model; L = 3 in the reduced model), with each layer refining the spatial representations produced by the previous layer.

### 3.6.5 Spatial Decoder

After the transformer blocks, each of the 100 patch embeddings has been transformed into a 128-dimensional representation encoding learned spatial relationships across the entire grid. To convert this back into a 50 × 50 spatial prediction map, we apply a convolutional decoder that progressively upsamples the 10 × 10 x 128 patch grid to the full 50 × 50 x 1 output resolution.

The decoder proceeds through three upsampling stages: bilinear interpolation from 10 × 10 to 20 × 20, then 20 × 20 to 40 × 40, and finally a size-preserving upsample to exactly 50 × 50. Each upsampling step is followed by a 3 × 3 convolutional layer (with reflection padding to avoid edge artefacts), Group Normalization (which normalizes within groups of channels rather than across the batch, providing stable normalization with small batch sizes), and GELU activation. The channel depth is reduced from 128 to 64 to 32 to 16 across these stages, and a final 3 × 3 convolution produces the single-channel 50 × 50 output.

### 3.6.6 Output Activation and Spatial Masking

Crab densities are non-negative by definition. To enforce this constraint, the decoder output is passed through a **softplus** activation function, $f(x) = \log(1 + e^x)$, which is a smooth, differentiable approximation to ReLU that maps the real line to strictly positive values without an upper bound. This choice is preferable to sigmoid (which caps predictions at 1) or direct exponentiation (which can produce explosive gradients). The spatial mask is applied as a final step, setting all out-of-region cells to exactly zero.

## 3.7 Loss Functions

### 3.7.1 MSE Loss

The primary loss function used for model training was Mean Squared Error (MSE), operating directly on the log-transformed predictions and targets. Because predicting the mean of log(*x*) is not the same as predicting the mean of *x* (Jensen's inequality), predictions evaluated under MSE were corrected by a lognormal bias correction factor of $\exp(\hat{\sigma}^2 / 2)$, where $\hat{\sigma}^2$ is the variance of the training residuals in log space. This correction converts the median prediction back to an approximately unbiased estimate of the mean density. <span style="color:red"> If the average density values (not variance) for the training, validation, and test are  different from eachother, train: 1612, val: 1801, test: 406. How should I handle the bias? </span>

The per-pixel losses are averaged over valid (in-region) pixels within each sample, producing a per-sample scalar loss, which is then averaged over valid samples (excluding 2020) within each batch.

### 3.7.2 Tweedie Loss

Another loss function used was the **Tweedie deviance**, which belongs to the generalized linear model family and is specifically suited to non-negative continuous data with a high proportion of zeros. The Tweedie distribution with power parameter 1 < *p* < 2 corresponds to a compound Poisson-Gamma distribution: it places a point mass at zero (modelling the probability of observing no crabs in a cell) combined with a continuous distribution for positive densities. The Tweedie log-likelihood (as a function of the predicted value $\hat{y}$ and observed value $y$) gives rise to the following per-pixel loss:

$$\ell(\hat{y}, y) = -\frac{y \, \hat{y}^{1-p}}{1-p} + \frac{\hat{y}^{2-p}}{2-p}$$

with power parameter *p* = 1.5. Because the model predicts in log space (after the $\log(1 + x)$ input transform), predictions are back-transformed via $\hat{y}' = e^{\hat{y}} - 1$ before computing the Tweedie loss, so the loss operates on the original density scale rather than the log-transformed scale.


## 3.8 Model Training

### 3.8.1 Weight Initialization and Bias Warm-Start

All linear layers in the transformer (attention projections, patch embeddings, feed-forward layers) were initialized using **Xavier uniform initialization**, which scales initial weights by $\sqrt{6 / (d_{\text{in}} + d_{\text{out}})}$ to maintain approximately constant gradient variance through the forward and backward passes. Convolutional layers in the decoder were initialized with **Kaiming (He) normal initialization**, which is designed for layers followed by ReLU-family activations (GELU in our case) and compensates for the fact that approximately half of the pre-activation values are non-positive. Layer and Group Normalization scale and shift parameters were initialized to 1 and 0, respectively, corresponding to an identity transformation at the start of training.

To prevent the model from getting stuck in the early training phase predicting near-zero outputs regardless of input (the so-called "identity trap"), the bias of the final decoder output convolution was initialized to the mean log-transformed recruit density over all valid (non-masked, non-2020) training pixels. This warm-starts the output layer at a reasonable baseline prediction without touching any of the learned representations.

### 3.8.2 Optimizer

Model parameters were updated using **AdamW** (Loshchilov & Hutter, 2019), a variant of the widely-used Adam optimizer that incorporates weight decay as a true L2 penalty on parameter magnitudes (rather than as a modification to the gradient updates as in standard Adam). AdamW maintains per-parameter adaptive learning rates based on estimates of the first and second moments of the gradients, allowing different components of the model (e.g., attention layers versus convolutional decoder) to learn at appropriate speeds. Weight decay (λ = 10⁻⁴) provides regularization analogous to ridge regression. The base learning rate was set to 10⁻⁴.

### 3.8.3 Learning Rate Schedule

We employed the **OneCycleLR** scheduler, which varies the learning rate across training according to a three-phase curve. Starting from an initial rate of max_lr / div_factor (3 × 10⁻⁴ / 25 ≈ 1.2 × 10⁻⁵), the rate warms up via cosine annealing to a maximum of max_lr = 3 × 10⁻⁴ over the first 30% of total training steps (pct_start = 0.3). It then anneals down to a final rate of max_lr / final_div_factor = 3 × 10⁻⁷ over the remaining 70% of steps. The learning rate is updated after every batch (not every epoch). The warm-up phase helps the model establish productive gradient directions before taking large steps, while the subsequent decay phase fine-tunes parameters near a local minimum. The peak learning rate is designed to be large enough to escape the identity trap (predicting mean density everywhere) while remaining stable.

### 3.8.4 Model Selection during Training

Training was run for up to 15 epochs. Validation loss was evaluated after each epoch, and the model checkpoint with the lowest validation loss was saved. The best checkpoint was loaded at the end of training for evaluation on the held-out test set.

Gradient clipping (max norm = 1.0) was applied after the backward pass to prevent large gradient updates from destabilizing training, particularly in the early epochs before the scheduler has warmed up.

## 3.9 Model Configurations

Because the effective number of independent training patterns is small (~24 spatial configurations), the appropriate model complexity is not obvious *a priori*. We therefore trained two configurations as a sensitivity analysis:

| Configuration | embed_dim | Heads | Layers | d_ff | Dropout | Approx. Parameters |
|---|---|---|---|---|---|---|
| Full | 128 | 8 | 6 | 512 | 0.1 | ~1.3M |
| Reduced | 64 | 4 | 3 | 256 | 0.2 | ~130K |

Both configurations were trained on the same data splits and evaluated using the same metrics, allowing comparison of the overfitting behaviour across model sizes. Results from both configurations are reported; we expect that both exhibit training/validation loss divergence given the fundamental data constraint (Section 3.2), and discuss the implications of this for transformer-based ecological modelling.

## 3.10 Model Evaluation and Interpretability

Model predictions were evaluated against held-out observations using a 
comprehensive set of per-sample metrics, where each sample corresponds to 
a full 50×50 spatial recruitment field for a single bootstrap replicate 
and year. All metrics were computed over valid (in-region, non-masked) 
cells only.

Pointwise accuracy was assessed using Root Mean Squared Error (RMSE) and 
Mean Absolute Error (MAE), both computed in log space to match the 
training objective. To decompose the sources of prediction error, RMSE 
was partitioned into a systematic bias component and a spatial pattern 
component (unbiased RMSE), following the decomposition of Murphy (1988). 
The fraction of total squared error attributable to mean bias versus 
spatial misalignment was reported separately, allowing us to distinguish 
models that predict the correct spatial pattern but at the wrong scale 
from those that fail to capture spatial structure entirely.

Spatial pattern accuracy was assessed using Pearson and Spearman rank 
correlations between predicted and observed fields. Spearman correlation 
is distribution-free and robust to scale differences between predictions 
and observations, making it appropriate when predictions may be 
systematically biased in magnitude but correct in rank order. Overall 
abundance accuracy was quantified as total predicted density versus total 
observed density (summed over valid cells), with percentage abundance 
error reported as (Σŷ − Σy) / Σy × 100.

To evaluate performance on the spatial structure of recruitment 
specifically, two additional metrics were computed. Top-10% spatial 
accuracy measured the Jaccard overlap between the set of cells in the 
top decile of observed density and the corresponding set of cells in the 
top decile of predicted density, capturing whether the model correctly 
identifies high-recruitment hotspots regardless of absolute scale. 
Zero-cell classification performance was assessed using precision, recall, 
and F1 score for identifying cells below a density threshold of 14.23 
(corresponding to the 10th percentile of non-zero observed densities in 
the training set), reflecting the model's ability to correctly place 
spatial zeros.

Finally, overall predictive skill was assessed relative to a 
climatological mean baseline — a model that always predicts the 
per-cell mean training density regardless of input. Positive skill 
scores (both MSE-based and MAE-based) indicate that the transformer 
predictions contain useful signal beyond simply predicting average 
conditions.

### 3.12 Post-hoc Attribution via Integrated Gradients

To identify which spatial regions and input channels most strongly 
drive recruitment predictions, we applied Integrated Gradients (IG; 
Sundararajan et al., 2017), a post-hoc attribution method that satisfies 
the completeness axiom: the sum of all input attributions equals exactly 
the difference between the model output for the actual input and a 
chosen baseline input. This property makes IG more interpretable than 
raw gradient saliency, which can highlight inputs the model is sensitive 
to locally but that contribute little to the actual prediction.

**Baseline.** The reference input was set to the mean of all valid 
(non-2020) training samples, producing a single [11, 50, 50] tensor 
representing an "average year" across all bootstraps. Positive 
attributions at a given pixel and channel indicate that the input 
exceeded the baseline at that location in a way that increased the 
predicted total recruitment; negative attributions indicate the 
opposite.

**Attribution computation.** For each year in the dataset, IG was 
computed independently for each of the 100 bootstrap replicates and 
then averaged, yielding a single per-year attribution map. For a single 
sample, the IG computation proceeds as follows. A straight-line path 
of 50 interpolated inputs was constructed between the baseline and 
the actual input: $x_\alpha = x_\text{base} + \alpha \cdot 
(x - x_\text{base})$ for $\alpha \in \{1/50, \ldots, 50/50\}$. 
At each interpolated input, the model was run in forward mode and 
gradients of the scalar output (total predicted density summed over 
valid cells) with respect to the input were computed via backpropagation. 
The average gradient across all 50 steps was then multiplied by the 
input delta to produce the final [11, 50, 50] attribution tensor:

$$\text{IG}(x) = (x - x_\text{base}) \cdot 
\frac{1}{m}\sum_{\alpha=1}^{m} \nabla_x f\!\left(x_\text{base} + 
\frac{\alpha}{m}(x - x_\text{base})\right)$$

**Visualization.** For each year, attribution maps were visualized 
alongside the corresponding mean input fields for all 11 channels. 
An aggregated attribution panel — summing absolute attribution across 
all channels — was also produced to highlight spatially influential 
regions regardless of which input channel drove the effect. These 
panels allow inspection of whether the model relies on biologically 
plausible spatial structure (e.g., attributing high recruitment 
predictions to high spawner densities in known breeding grounds) or 
on spurious patterns in the data.

---
*Table S1: GMRF synthetic data parameters for easy, medium, and hard difficulty levels — see Supplementary Material.*
