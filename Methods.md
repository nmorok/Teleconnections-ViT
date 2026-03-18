# Overview of the pipeline

### generate_dummy_data.py
**Idea** 
The idea for this file was to generate spatiotemporal data from a known distribution, so that we could have a dataset to test the model pipeline. 

**Steps**
1. Create a spatial precision matrix where the diagonal $[i,i]$ is $\kappa^2$ $\times$ number of neighbors, and the off-diagonal $[i,j]$ is $-\kappa^2$ if $i$ and $j$ are neighbors and $0$ otherwise
   1. initializes a sparse matrix
   2. loop through all grid cells, counting how many neighbors each cell has (2-4)
   3. for each neighbor, its value is set to $-\kappa^2$
   4. set each diagonal to number of neighbors $\times \kappa^2$
2. Sample from the precision matrix  
   1. take the precision matrix we just created from the GMRF
   2. generate a standard normal random vector ($z$) using np.random.randn
   3. solve the sparse linear sustem $Q \times x = z$ 
      1.  can potentially switch to cholesky decomposition if desired in the future for less smooth
   4. return $x$ as our sampled spatial correlation matrix
3. Add the temporal autocorrelation, $\rho$ (if desired)
   1. for years after the first, we'll take the previous year's gmrf and multiply it by $\rho$.
   2. add the new field that we sample multiplied by sqrt(1-rho^2) to keep the variance constant over time.
   3. new gmrf = past gmrf $\times \rho + $current gmrf $\times \sqrt(1-\rho)$
4. Transform the data to a realistic scale
   1. data is currently around $N(0,1)$ from the GMRF sampling. We want positive, realistic crab densities that follow a tweedie.
   2. standardize the data via mean and std.
   3. transform the data via $exp(datastandardized \times factor)$ to create a right-skewed distribution. factor was chosen arbitrarily to be 1.5.
   4. multiply by the mean density that we set (currently spawners set to 50, and recruitment to 20)
   5. apply a threshold to force zeros for the data. Set the threshold to 15 which yielded ~30% zeros. $(data - 15) < 15 = 0$
5. Repeat for spawners and recruits using different seeds for the GMRF sampling
6. Add spawner and recruitment correlation
   1. for each bootstrap and year, we get the corresponding fields for both the spawners and the recruits
   2. if we have a lag, this is where it comes into play by changing the year of the spawner field. 
   3. for the years in which there are no spanwer fields for the recruits $(year-lag < 0)$ we only use the recruit field with no alteration.
   4. where there is a spawner field, we do: recruitment_correlation $\times$ spawners + $sqrt(1-recruitmentcorrelation^2) \times $ recruits

**Parameters**

Easy:
- grid size: 50
- years: 30
- bootstraps: 100
- spatial kappa: 0.3
- temporal rho: 0
- recruitment_correlation: 1.0
- mean spawners: 50
- mean recruits: 30
- temporal lag: 0
- recruits seed: 2026
- spawners seed: 3026

Medium: 
- grid size: 50
- years: 30
- bootstraps: 100
- spatial kappa: 0.3
- temporal rho: 0.7
- recruitment_correlation: 0.9
- mean spawners: 50
- mean recruits: 30
- temporal lag: 3
- recruits seed: 2026
- spawners seed: 3026
- 
Hard:
- grid size: 50
- years: 30
- bootstraps: 100
- spatial kappa: 0.3
- temporal rho: 0.7
- recruitment_correlation: 0.6
- mean spawners: 50
- mean recruits: 30
- temporal lag: 5
- recruits seed: 2026
- spawners seed: 3026

### process_data.R
**Idea**
Sample the EBS bottom trawl data to augment the amount of data we have using SPDE method.

**Steps**
1. Build a 50 x 35 prediction grid over the EBS region. (values picked arbitrarily) (the remaining 15 rows get padded)
2. Construct an SPDE mesh from the station locations
3. Subsample 300 of the 349 stations for each bootstrap
4. Fit an INLA SPDE model per year on the subsampled stations
5. Project predictions to the grid

**Parameters**
- Cellsize: 25 (roughly 23 x 23 km cells)
- Grid_nx: 50 (number of columns) 
- Grid_ny: 35 (number of rows)
- Pad_ny: 50 (extra rows to make square grid)
- N_bootstraps: 100
- N_subsample: 300 (number of stations to subsample)
- cutoff: 30 (for INLA methods)

### create_splits.py
**Idea**
Split the 100 bootstraps of 30 year data into training, validation, and testing.

**Steps**
1. split the data by the year ranges given
2. data is formatted by B0Y0, B0Y1, B0Y2 ... 

**Parameters**

Dummy data:
- training years: 18 (or 22 in some runs — check gmrf params file for the run)
- validation years: 9 (or 5)
- testing years: 3

Real data (current active pipeline):
- training years: 24
- validation years: 8
- testing years: 4
- lag: 0 (lag is pre-aligned in create_splits.py; the memory bank handles lookback)

### data_helper.py
**Idea**
Give the training/model the correct batches of data. 
With the historical data, bridge the data between training, validation, and testing sets. For the early years, add a temporal mask.
Also, transform the data if need be for better model performance.

**Steps**
1. load the training data
   1. transform the data if desired
      1. currently support log transform
      2. if transformation = log, take $log(1+density)$
2. load the historical data if available
   1. get the data from the previous years (all bootstraps) and add it to the channel data
   2. for the training data where the years are less than the lag, only give as much data is available, and then mask out the missing years using a tenor vector that is 0 for missing years and 1 for valid years, including the current year. The historical year channels that are not valid are set to 0 everywhere and still passed, since the model needs full channel data. The mask will handle telling the model the data is empty/padding instead of all 0s.
   3. to ensure that the historical data is available across validation, and testing sets, there is a bridge function to get the historical data from either the training or validation set. 
3. Calls the torch dataloader which:
   1. creates a random permutation of indices if shuffle = true.
   2. takes the first batchsize indices and calls ```__getitem__``` from my function
   3. appends the result to a samples vector. 
   4. returns that vector (batch size long)
   5. continue until all samples have been used.
4. If we are doing one year ahead prediction (`include_current_spawner=False`), then:
   1. Channel 0 (current spawner) is replaced with a zero grid.
   2. The temporal mask at index 0 is set to 0, so the model knows this channel is invalid.
   3. The model must predict recruitment at year *t* using only spawner and recruit data from years t-1 through t-5.
   4. A sample is marked invalid (`valid_year = 0`) ONLY if the entire temporal mask sums to 0, meaning no current spawner AND no historical data whatsoever. This only occurs for year 0 of the training set when no bridge data is available. All other samples (even those with partial history) are treated as valid training observations.
   5. This mode corresponds to a genuine forecasting scenario where the current-year survey has not yet occurred.
   6. NOTE: The same `valid_year` flag is reused for both the 2020-exclusion logic and the one-year-ahead empty-sample logic. These are distinct reasons for exclusion - <span style="color:red">consider using separate flags in future refactors for clarity</span>.

```
Training dataset example with idx=1432
bootstrap_idx = 65, year_idx = 2

Since this is the TRAINING dataset:
- self.historical_spawners = None (no previous split)
- Only years 0, 1 exist before year 2

for i in range(5):
    lookback = i + 1  # 1, 2, 3, 4, 5
    historical_year = 2 - lookback  # 1, 0, -1, -2, -3
    
    # i=0: historical_year=1 → EXISTS in current split ✓
    if historical_year >= 0:
        historical_idx = 65 * 24 + 1 = 1431
        memory_spawners[0] = self.spawners[1431]
        temporal_mask[1] = 1.0  # ✓
    
    # i=1: historical_year=0 → EXISTS in current split ✓
    elif historical_year >= 0:
        historical_idx = 65 * 24 + 0 = 1430
        memory_spawners[1] = self.spawners[1430]
        temporal_mask[2] = 1.0  # ✓
    
    # i=2: historical_year=-1 → DOES NOT EXIST
    elif self.historical_spawners is not None:  # ✗ FALSE (is None)
        # This branch is SKIPPED
        # memory_spawners[2] stays 0.0
        # temporal_mask[3] stays 0.0
    
    # i=3: historical_year=-2 → DOES NOT EXIST
    # Same - skipped
    
    # i=4: historical_year=-3 → DOES NOT EXIST  
    # Same - skipped
    
    CORRECT Result for training dataset, year 2:
    temporal_mask = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
                    ^cur ^-1  ^-2  ^-3  ^-4  ^-5
                     ✓    ✓    ✓    ✗    ✗    ✗
                     Only 3 positions valid (current year + 2 years history)
```

**Parameters**
- batch_size: 12
- memory_years: 5
- train_years: 22
- val_years: 5
- test_years: 3
- transform: 'log'

### components.py
**Idea**
Contains all of the functions for the transformer model. 

#### PatchEmbedding
**Idea**
Divide the grids into patchs of size patch_size and then embed each channel of patches into a vector of embed_dim long. 
Wanted to process each channel seperately and then combine at the end because the different channels have different meanings, temporal masking needs to be applied selectively, and allows different learned projections for each type of infromation.
**Steps**
1. Create the patches [B, 11, 50, 50] --> [B, 100, 11, 25]
   1. unfold along height dimension [B, 11, 50, 50] -> [B, 11, 10, 50, 5] (10: number of patches along height) (5: patch height)
   2. unfold along width dimension [B, 11, 10, 50, 5] -> [B, 11, 10, 10, 5, 5] (10: number of patches along width) (5: patch width)
   3. Rearrange dimensions [B, 11, 10, 10, 5, 5] -> [B, 10, 10, 11, 5, 5]
   4. Flatten into sequences [B, 10, 10, 11, 5, 5] -> [B, 100, 11, 25]
2. Process each channel separately
   1. apply a learned linear transformation to go from dim 25 to dim 128
      1. Using nn.Linear 
   2. normalize the vector using LayerNorm -- $LN(x) = \gamma * ((x-\mu) / \sqrt(\sigma^2+\epsilon)) + \beta$, where $\gamma$ and $\beta$ are learnable scale and shift parameters, and $\epsilon$ is a small constant for numerical stability.
   3. apply dropout
3. each channel gets processed into a 12 embedding dimensionality vector (math.ceil(128/11) = 12). [B, 100, 25] -> [B, 100, 12]
4. multiply the channel by the temporal mask (1.0 if valid, 0 if invalid)
5. concatenate all of the channel embeddings together so now we have [B, 100, 132]
6. apply one more linear transformation to go from [B, 100, 132] to [B, 100, 128]

**Parameters**
- grid_size: 50
- patch_size: 5
- in_channels: 11
- embed_dim: 128
- dropout: 0.1

#### PositionalEncoding2D
**Idea**
Add learned positional embeddings to patches. Each of the 100 patches gets a unique position vector
**Steps**
1. initialize a learned vector of length embed_dim with initial values taken from normal(0,1) then multiplied by a scaler for stability and to ensure the values don't mask the actual data
   1. Using nn.Parameter()
2. add the vector to the input data
**Parameters**
- n_patches: 100
- embedding_dim: 128
- scale: 0.1

#### TemporalEmbedding
**Idea**
Add a fixed sinusoidal encoding to represent the year. Unlike learned embeddings, this uses math (sine/cosine) to create a unique temporal signature that allows the model to understand the distance between years (e.g., Year 5 is closer to Year 6 than Year 20), even for years it hasn't seen during training.
**Steps**
1. Create the temporal signatures
   1. Generate Time Indices: Create a column vector of years (0, 1, 2...).
   2. Calculate Frequency Scale (div_term): Compute 10000 raised to decreasing powers. This ensures that some dimensions of the embedding track "days" while others track "decades"
   3. Apply Trigonometry: Multiply the year by the div_term and pass it through sin (for even indices) and cos (for odd indices).
   4. Buffer Storage: Use register_buffer to store this as a constant "lookup table" so you don't waste CPU cycles re-calculating the same waves every batch.
   5. Dynamic Fallback: If a year index exceeds the table, use the same formula to compute the vector on the fly, ensuring the model can extrapolate into the future.
2. select the temporal signature relating to the year and add it to the input

**Parameters**
- embed_dim: 128
- max_years: 30
- precompute_years: 50

#### MultiHeadAttention
**Idea**
The guts of the transformer model. Allows different parts of the grid to attend to other parts.

**Steps**
1. Learned linear projection of the Q, K, V weights onto the data
   1. Using nn.Linear
   2. projections are initialized using xavier uniform for the weights, and zeros for the bias
2. reshape the output into multiple heads [B, 100, 128] -> [B, 100, heads (8), dim_head (16)] 
3. Compute attention scores
   1. Q @ K^T
   2. Scale by square root of dim_head for numerical stability
   3. Apply the temporal mask, setting all of the scores from the years without data to -1e9 so softmax transposes it to 0.0. 
   4. Apply softmax to get attention weights
   5. apply dropout
4. Apply attention weights to values via matrix multiplication
5. Combine the heads together
6. Apply an output projection to allow all of the heads to talk to each other

**Parameters**
- dim_model: 128
- num_heads: 8
- dropout: 0.1

#### FeedForward
**Idea**
This allows the model to do expanded thinking and learn from the attention process

**Steps**
1. linear transformation of the data to expand the dimensionality [B, 100, 128] -> [B, 100, 512]
   1. using nn.Linear
2. Apply a non-linear transformation (GELU)
3. Apply dropout
4. Apply a linear transformation to get back into the dimensionality of our data [B, 100, 512] -> [B, 100, 128]
   1. using nn.Linear
5. Apply dropout

**Parameters**
- dim_model: 128
- d_ff: 512
- dropout: 0.1

#### TransformerBlock
**Idea**
Puts together the multihead attention and feedforward process

**Steps**
1. normalize the data
   1. Using LayerNorm
2. apply attention to the normalized data and add it to the un-normalized data.
3. normalize the output and apply the feedforward module
   1. using layerNorm
4. add it to the un-normalized output of the attention module

**Parameters**
- d_model: 128
- num_heads: 8
- d_ff: 512
- dropout: 0.1

#### SpatialDecoder
**Idea**
Transform the embeddings back into our 50 x 50 grid. reverses the encoding process by expanding the spatial dimensions while compressing feature depth. 

**Steps**
1. upsample the data [B, 128, 10, 10] -> [B, 128, 20, 20] because we have 100 vectors each of 128 and we rearranged them to be in this format.
   1. using nn.Upsample with a scaling factor of 2, bilinear mode, and don't align corners
2. Convolution layer to reduce the dimensionality [B, 128, 20, 20] -> [B, 64, 20, 20]
   1. going from 128 -> 64, using nn.Conv2d with a kernel size of 3, padding of 1, and padding mode 'reflection'
   2. 64 different $3 \times 3$ filters scan the new $20 \times 20$ grid. They look at the 128 abstract features and condense them into 64 more "visual" features.
3. Group normalization
   1. used nn.groupnorm instead of batch norm for better stability when batch size is small
   2. Group Normalization divides the 128 channels into 8 groups and normalizes them independently of the batch size.
4. GELU activation
5. Updample the data [B, 64, 20, 20] -> [B, 64, 40, 40]
   1. using nn.Upsample with a scaling factor of 2, bilinear mode, and don't align corners
6. Convolution layer to reduce the dimensionality [B, 64, 40, 40] -> [B, 32, 40, 40]
   1. going from 64 -> 32, using nn.Conv2d with a kernel size of 3, padding of 1, and padding mode 'reflection'
7. Group normalization
   1. used groupnorm instead of batch norm for better stability when batch size is small
8. GELU activation
9. Upsample to size of 50 [B, 32, 40, 40] -> [B, 32, 50, 50]
   1.  using nn.Upsample with size of 50, bilinear, don't align corners
10. Convolution layer to reduce the dimensionality [B, 32, 50, 50] -> [B, 16, 50, 50
    1.  Going from 32 -> 16, using nn.Conv2d with a kernel size of 3, padding of 1, and padding mode 'reflection'
11. GELU activation
12. Convvolution layer to reduce the dimensionality [B, 16, 50, 50] -> [B, 1, 50, 50]
    1.  using nn.Conv2D going from 16 -> 1, with a kernel size of 3, padding of 1, and padding mode 'reflection'

**Parameters**
- embed_dim: 128
- patch_grid_size: 10

### model.py
**Idea**
Put all of the components together to form a whole ViT transformer architecture. 

**Steps**
1. Initialize all of the variables, embedding vectors, transformer blocks, etc
2. Initialize the weights of the parameters
   1. All Linear parameters (attention weights, patchembedding, spatialencoding) get initialized with Xavier uniform
      1. it picks weights from a distribution with a variance of 2/(in+out)
      2. where in is the number of input units connected to a neuron. since this is a linear layer, it it the size of the previous layer. Out is the number of output units a neuron sends its signal to. 
      3. The attention layers require a lot of matrix multtiplication. If the weights start too large, the output values of the dot-product attention become massive, pushing the softmax into the flat regions where the gradient is almost zero. Xavier keeps the signal variance the same from input to output, ensuring the 'flow' of information stays steady.
   2. All of the convolutional layers get initialized with Kaiming (He) Normal
      1. It picks weights from a normal distribution with a variance of 2/in
      2. where in is the number of input units connected to a layer. Since this is a convolution layer it is the kernal size * kernal size * input channels. 
      3. Because of the GELU activations in the decoder, the GELU 'kills' half of the signal (anything negative). So this initialization compenstates for this 'lost' signal by making the starting weights slightly larger. It ensures that even after 3 or 4 layers of convolutions, the data hasn't completely shriveled to 0. 
   3. Batch Norm and LayerNorm modules get initialized with 1 for weights and 0 for bias. 
      1. normalization layers are designed to 'reset' the data to a mean of 0 and a sd of 1. by initializing the weights to 1 and biases to 0, we are telling the layer: start by doing a perfect normalization, and only change the scale/shift if you find a reason to during training.
3. Convert the grid to patch embeddings using the PatchEmbedding script
4. Add the positional encoding using the PositionalEncoding2D script
5. Add the temporal encoding using the TemporalEmbedding script
   - <span style="color:red">KNOWN RISK: Sinusoidal temporal encoding allows the model to use the year index as a direct lookup key rather than learning genuine spawner–recruit relationships. If the model memorizes year-specific patterns during training (effectively learning that "year 3 always has high recruitment"), it will fail to generalize to test years. Monitor for this by checking whether test-year predictions collapse. Removing temporal encoding and replacing with GroupNorm in the decoder is a documented fix if this occurs. If test-year Spearman correlation is high but the model fails on years with indices it hasn't seen (e.g., if you retrain with a gap year held out), that's the smoking gun for year-index memorization.</span>
6. Apply dropout to the embeddings. Randomly zeros individual elements of the 128-dimensional vectors, independently per element per forward pass. Not the entire patch or entire channels, just scalar positions within the embedding vectors. 
7. Run the full transformer block using the TransformerBlock script num_layers times
8. normalize the output of the transformer blocks 
   1. using layernorm
9.  Get output using the SpatialDecoder script
   1.  Before calling the decoder, first need to reshape into the correct order
10. Apply softplus
    1.  To constrain the output to be greater than 0 and uncapped.
    2.  Smooth function
11. Multiply by the spatial mask to ensure that the cells outside of the EBS region are set to 0.
12. return the predicted 50 x 50 grid

**Parameters**

Two configurations are trained as a sensitivity analysis (see paper methods §3.9):

Full model (larger):
- grid_size: 50
- patch_size: 5
- in_channels: 11
- embed_dim: 128
- num_heads: 8
- num_layers: 6
- d_ff: 512
- dropout: 0.1
- approx params: ~1.3M

Reduced model (active in notebook):
- grid_size: 50
- patch_size: 5
- in_channels: 11
- embed_dim: 64
- num_heads: 4
- num_layers: 3
- d_ff: 256
- dropout: 0.2
- approx params: ~130K

### losses.py
**Idea**
Since the observed data has inflated zeros, wanted to use Tweedie distribution for the loss calculation.

**Steps**
1. Force all values to be at least 1e-8 to avoid dividing by zero.
2. Flatten the spatial dimension to compute the loss per pixel.
3. Calculate tweedie loss using formula:
   1. $term1 = -target * prediction ^ {1-p} / (1-p))$
   2. $term2 = prediction^{2-p} / (2-p)$
   3. $loss = term1 + term2$
   4. IMPORTANT: The Tweedie loss operates on the ORIGINAL (non-log) density scale. In the training loop, both `outputs` and `targets` are back-transformed via `torch.expm1()` before being passed to TweedieLoss. The MSE loss, by contrast, operates on the log-transformed scale directly.
4. Calculate MSE loss normally (on log scale)
   1. For MSE: a lognormal bias correction factor of exp(σ²/2) is computed from training residuals and applied at evaluation time to correct for Jensen's inequality.
5. Multiply by the spatial mask to ensure we only count valid errors
6. return the mean of all of the errors

**Parameters**
- power: 1.5

### train.ipynb
**Idea**
Training the model using compute GPUs from google via google colab. 
Want to train the model on the training data, validate the model on the validation data. Take the model with the best validation error, not necessarily the best training error. 

**Steps**
1. Mount google drive
2. Clone github repo into drive
3. Initialize the data via get_dataloaders() from data_helper.py
4. Transfer the model to GPU 
5. Initialize the transformer model via CrabTransformer from model.py
6. calculate the mean recruit density to overwrite the decoder output convolution bias
   1. and then actually overwrite it
   2. this is to set the output to something that is reasonable for the model so it doesn't get lost right away
7. Initialize the tweedie loss
8. Initialize the optimizer (AdamW)
   1. This is what actually tunes the model parameters.
   2. Chose AdamW because of the capabilities of adaptive learning rates. AdamW allows for different layers (embeddings, attention, decoder) to learn at different speeds. AdamW gives each parameter its own 'speedomoter' adjusting the learning rate individually.
   3. The W in AdamW stands for decoupled weight decay, and assists in regularization.
   4. AdamW maintains two running statistics for each parameter: a first moment 
   estimate (the exponential moving average of past gradients, analogous to 
   momentum) and a second moment estimate (the exponential moving average of 
   past squared gradients, analogous to a per-parameter learning rate scale). 
   At each update step, the gradient is divided by the square root of the 
   second moment, which automatically shrinks the effective learning rate for 
   parameters that have historically received large gradients, and enlarges it 
   for parameters that have received small or infrequent gradients. This allows 
   attention layers, which accumulate large gradient signals from many patches, 
   to take smaller cautious steps, while the decoder's final output convolution, which receives sparse gradients early in training, can take larger steps. 
   The "W" distinguishes AdamW from standard Adam: rather than folding weight 
   decay into the gradient update (which distorts the adaptive scaling), AdamW 
   applies weight decay as a direct multiplicative shrinkage to the parameter 
   values after the gradient step, equivalent to true L2 regularization.
1. Initialize the scheduler (OneCycleLR)
   1. It updates the learning rate for every batch, not just once per epoch. This creates a smooth curve for the learning rate, making training much more stable.
   2. Chose OneCycleLR to help the model avoid the identity trap, which is when the model gets stuck predicting the average density. The OnceCycleLR works by starting with a very low learning rate to find the correct direction, then ramps up to a high rate to jump over local minima and finally slows down to fine-tune pixels.
   3. OneCycleLR updates the learning rate every batch (not every epoch). It follows a three-phase curve: it starts at max_lr / div_factor (so 3e-4 / 25 ≈ 1.2e-5), ramps up to max_lr (3e-4) over the first 30% of total steps (pct_start=0.3), then cosine-anneals down to max_lr / final_div_factor (3e-4 / 1000 = 3e-7) over the remaining 70%. The total number of steps is epochs × steps_per_epoch (and yes, steps_per_epoch = len(train_loader) which is number of batches, not samples).
   4. <span style="color:red">KNOWN PROBLEM: OneCycleLR precomputes the total number of steps from (epochs × steps_per_epoch) at initialization. If early stopping fires before all epochs complete, the scheduler has already planned a learning rate trajectory for the full run — it won't gracefully anneal down to a low final rate. This means the saved best model may have been checkpointed at a relatively high mid-cycle learning rate. A ReduceLROnPlateau scheduler would avoid this issue. Consider switching for future runs. </span>
2.  For every input, target pair in the training loader (so this would be for every pair in a batch)
    1.  zero out the gradient
    2.  Forward model pass
    3.  Compute the loss
    4.  Move backwards and compute the gradient
    5.  clip the gradients to avoid exploding gradients
    6.  optimize the weights
    7.  run the scheduler # what does this do?
3.  compute the average training loss across all batches
4.  change the model into evaluation mode (doesn't compute gradients)
5.  for every input, target pair in the validation loader (every batch)
    1.  Forward model pass
    2.  Compute the loss
6.  compute the average validation loss across all batches
7.  calculate the bias of the data using the validation data
8.  save the learning rate, the training loss, the validation loss
9.  save the model if the validation error is less than the current model's validation error
10. increase patience counter if no improvement has occured
11. Once all epochs have run or we stopped from earlystopping, load the best model
12. evaluate the model on the test data. 
13. plot the training curves

**Parameters**

dummy data:
- batch_size: 8
- memory_years: 5
- train_years: 22
- val_years: 5
- test_years: 3

real data:
- batch_size: 8
- memory_years: 5
- train_years: 24
- val_years: 8
- test_years: 4

model:
- grid_size: 50
- patch_size: 5
- in_channels: 11
- embed_dim: 128
- num_heads: 8
- num_layers: 6
- d_ff: 512
- dropout: 0.1

loss:
- power: 1.5

optimizer:
- learning rate: 1e-4
- weight decay: 1e-4

scheduler:
- max learning rate: 3e-4
- epochs: 15
- steps_per_epoch: len(train_loader) # number of batches per epoch, NOT number of samples
- pct_start: 0.3 # 30% warmup (NOT 10% as in some earlier runs)
- anneal_strategy: 'cos'
- div_factor: 25 # initial_lr = max_lr / 25 ≈ 1.2e-5
- final_div_factor: 1000 # final_lr = max_lr / 1000 = 3e-7

epochs:
- epochs: 15
- early_stop_patience: 20

gradient clipping: 
- max: 1.0


### run_batch_evaluation
**Idea** 
After training, run the best saved model checkpoint against all three data splits (train, val, test) and compute a comprehensive suite of spatial prediction metrics for every bootstrap replicate and year. Also visualize channel importance and attention maps.

**Steps**
1. Load the saved checkpoint for a given (level, criterion) combination from Google Drive
2. Load the lognormal bias correction factor from the training history JSON (if MSE loss was used; default 1.0 for Tweedie)
3. Initialize data loaders using `get_dataloaders()` with the same split sizes used during training
4. Plot channel importance
   1. For each of the 11 input channels, extract the corresponding `nn.Linear` layer (index 0 of the `nn.Sequential`) from `model.patch_embed.channel_projections`
   2. Compute the mean absolute weight magnitude: `weight.abs().mean()`
   3. Normalize all 11 values to [0, 1] so channels can be compared on a relative scale
   4. Bar plot of normalized importance, with Channel 0 (current spawner) highlighted in red
   5. Save to `analysis/batch_evaluation/channel_importance.png`
5. Run inference across all splits (train, val, test), collecting per-sample records
   1. For each batch, run a forward pass with `return_attention=True` to simultaneously collect predictions and attention maps
   2. Back-transform predictions and targets from log space to original density scale using `expm1()`; apply bias correction factor to predictions
   3. Skip any sample where `valid_year == 0` (i.e., 2020)
   4. For each valid sample, compute the full metric suite (see Parameters) on the flattened, spatially-masked valid cells
   5. Pre-zero cells below a threshold of 14.23 (the smallest observed non-zero density) before computing zero-classification metrics, matching the treatment used in the R-based baseline comparisons
6. For each phase (TRAIN/VAL/TEST), plot one attention map from the first un-plotted year for visual inspection
7. Aggregate records into a summary DataFrame with columns: year, bootstrap, phase, and all metrics
8. Print and save summary statistics (mean ± SD per phase) and the full per-replicate table

**Metrics computed per sample (spatial field)**
- Spearman rank correlation (distribution-free, robust to scale differences)
- Pearson correlation
- Mean Absolute Error (MAE)
- Total observed vs. predicted abundance (summed over valid cells)
- Percentage abundance error: `(pred_total - obs_total) / obs_total × 100`
- RMSE (total), RMSE unbiased (pattern component only after removing mean bias), bias, and the percentage of total RMSE attributable to systematic bias vs. spatial pattern error
- Zero-cell classification: Precision, Recall, and F1 for correctly identifying zero-density cells (cells below threshold 14.23)
- Top-10% spatial accuracy: Jaccard overlap between the top 10% of cells by observed density vs. predicted density
- Quantile bin error: mean absolute difference in density quantile bin assignment between observed and predicted
- Skill scores: MSE-based and MAE-based skill relative to a climatological mean baseline (positive = better than always predicting the mean)

**Parameters**
- levels: ['easy', 'medium', 'hard'] for dummy; ['real'] for real data
- criteria: ['MSE', 'Tweedie']
- data_type: 'dummy' or 'real'
- zero_threshold: 14.23 (smallest observed non-zero density value)
- top_k_fraction: 0.1 (top 10% of cells for spatial accuracy)
- n_quantile_bins: 10


### data_comparison_plot
**Idea**
Visualize the raw observed spawner and recruit abundance time series across all 100 bootstrap replicates, before any model is involved. This is a data quality and structure check: it confirms the data pipeline is working, shows the bootstrap uncertainty envelope around each year's total abundance, and makes the spawner–recruit relationship (or lack thereof) visually apparent across the time series. Panels are produced for each difficulty level (easy, medium, hard) or for real data.

**Steps**
1. Initialize data loaders for each level at batch size 1 (so every sample is processed individually)
2. Loop through all three loaders (train, val, test) in sequence to collect the complete time series in chronological order
3. For each batch:
   1. Back-transform inputs (spawner channel 0) and targets (recruits) from log space via `expm1()`
   2. Sum over valid spatial cells only (using the spatial mask for real data; all cells for dummy data) to get a scalar total abundance per bootstrap-year sample
4. Stack results into a (n_bootstraps × n_years) matrix
5. Compute the 5th, 50th (median), and 95th percentiles across bootstraps for each year
6. Plot for each level:
   1. Spawner median time series (dashed green line) with 5–95% bootstrap envelope (shaded)
   2. Recruit median time series (black line) with 5–95% bootstrap envelope (shaded)
   3. Vertical dashed lines and shaded background regions marking train/val/test boundaries
7. Save as `analysis/batch_evaluation/data_comparison_panel_{data_type}.png`

**Parameters**
- levels: ['easy', 'medium', 'hard'] or ['real']
- num_reps: 100 (number of bootstrap replicates)
- percentiles: [5, 50, 95]
- data_type: 'dummy' or 'real'


### integrated_gradients
**Idea** 
Use Integrated Gradients (IG) to attribute each model prediction back to specific input pixels and channels. IG is a post-hoc interpretability method that answers: "Which pixels in which input channels, and to what degree, caused the model to predict what it predicted for this year?" It satisfies the *completeness axiom*, meaning the sum of all pixel attributions equals exactly the difference between the model's output for the actual input and a chosen reference (baseline) input. This makes it more interpretable than simpler saliency methods like raw gradients. For a teleconnection analysis, IG maps allow us to identify which historical spawner or recruit fields, and which spatial regions, are driving recruitment predictions.

**Steps**
1. Compute a baseline input
   1. Average the input tensor across all valid (non-2020) training samples to produce a mean [1, 11, 50, 50] field
   2. This baseline represents an "average year" — the reference against which attribution is measured. A pixel with high positive attribution means the input was above the baseline at that location in a way that pushed the prediction up.
2. Organize all samples by year index, grouping all 100 bootstrap replicates under each year key. Skip 2020 samples.
3. For each year, run IG across all 100 bootstrap replicates and average
   1. **Core IG computation** (for a single sample):
      1. Compute the delta: $\delta = x_{\text{actual}} - x_{\text{baseline}}$
      2. Construct $n_{\text{steps}} = 50$ interpolated inputs along the straight-line path from baseline to actual: $x_\alpha = x_{\text{baseline}} + \alpha \cdot \delta$ for $\alpha \in \{0/50, 1/50, \ldots, 49/50\}$
      3. For each interpolated input, run a forward pass and compute the scalar output (sum of predicted density over valid spatial cells), then call `backward()` to get gradients with respect to the input
      4. Accumulate gradients across all steps: $\bar{g} = \frac{1}{n_{\text{steps}}} \sum_\alpha \nabla_x f(x_\alpha)$
      5. Multiply by delta: $\text{IG} = \bar{g} \cdot \delta$, giving a [11, 50, 50] attribution map
   2. Average the attribution maps and mean input fields across all 100 bootstraps to produce per-year summary maps
4. Visualize per-year panels
   1. Top rows: mean input value for each of the 11 channels (in log space) at this year
   2. Bottom rows: IG attribution map for each channel — warm colours indicate pixels whose above-baseline values pushed recruitment predictions up; cool colours indicate the opposite
   3. An additional aggregated attribution panel (sum of absolute attribution across all channels) highlights the most influential spatial regions regardless of which channel they came from
5. Save all per-year panels to `analysis/attribution/`

**Parameters**
- n_steps: 50 (number of interpolation steps along the IG path — more steps = more accurate but slower)
- output_fn: sum of predicted density over valid spatial cells (using spatial mask)
- baseline: mean valid training input [1, 11, 50, 50]
- channel_names: ['Spawner (t)', 'Spawner (t-1)', ..., 'Recruit (t-5)']


### generate_report
**Idea**
Produce a standardized, self-contained multi-page PDF report for each (level, criterion) model run, consolidating all evaluation outputs into a single document for comparison across runs. Each report is written to `reports/` in Google Drive.

**Steps**
1. Load the best checkpoint and training history JSON for the specified (level, criterion) combination
2. Initialize data loaders with the same split parameters used during training
3. Run inference across all splits, collecting per-sample predictions and observations
4. Assemble the following pages into a multi-page PDF using `matplotlib.backends.backend_pdf.PdfPages`:

   **Page 1 — Title and Training Config Summary**
   - Run identifier (level, criterion, timestamp)
   - Key hyperparameters: model size, batch size, optimizer settings, scheduler settings, total trainable parameters
   - Final training, validation, and test loss values

   **Page 2 — Training Curves**
   - Left panel: training loss and validation loss vs. epoch, with vertical line at the best epoch
   - Right panel: learning rate schedule vs. epoch (log scale), showing the OneCycleLR warmup and annealing phases

   **Page 3 — Channel Importance**
   - Bar chart of normalized mean absolute weight magnitudes from each channel's projection layer (same as `run_batch_evaluation`), providing a proxy for which temporal lags the model weighted most heavily

   **Page 4 — Abundance and Recruitment Deviations**
   - Time series of total observed vs. predicted recruitment abundance (median and 5–95% bootstrap envelope) across all years
   - Train/val/test regions shaded for context
   - Percentage error per year overlaid

   **Page 5 — Spatial Trajectory**
   - Grid of 50 × 50 spatial maps across all years for the first bootstrap replicate
   - Three rows per year: observed recruits, predicted recruits, signed residual (predicted − observed)
   - Fixed colour limits in log space (vmax = 8.0) to prevent extreme hotspot cells from collapsing the colormap

   **Page 6 — Attention Maps**
   - One representative attention map per phase (train, val, test), averaged across heads and converted to the 10 × 10 patch grid
   - Shows which spatial patches the model attended most strongly to when making predictions

   **Page 7 — Summary Statistics Table**
   - Mean ± SD for each metric (Spearman, Pearson, MAE, Skill Score, Zero F1, etc.) broken down by phase (TRAIN / VAL / TEST)

   **Page 8 — Detailed Metrics Table**
   - Full per-bootstrap, per-year metric table including RMSE decomposition (bias component vs. pattern component), zero-classification precision/recall/F1, top-k spatial accuracy, and quantile bin error

**Parameters**
- report pages: 8
- spatial color limits: vmin=0, vmax=8.0 (log1p scale)
- zero_threshold: 14.23
- output format: multi-page PDF saved to `reports/CrabTransformer_report_{level}_{criterion}.pdf`



#### to-dos:
-  Add cold pool channel. 
  
1. Spatial Alignment and Grid TranslationThe baseline bottom temperature data, sourced as a 10 km interpolated raster over the Eastern Bering Sea domain, was temporally subset to the target study period. To ensure perfect spatial congruence with the spawner and recruit arrays, the valid $25 \text{ km} \times 25 \text{ km}$ prediction grid cells (originally constructed for the SPDE model) were transformed to match the coordinate reference system of the temperature raster (Alaska Albers, EPSG:3338).
2. 
3. Spatial Gap-Filling and ImputationThe raw temperature raster contained spatial data gaps (NA values) resulting from survey boundaries and missing observations. To prevent the Vision Transformer from misinterpreting these missing regions as artificial $0^\circ\text{C}$ anomalies—which is a highly valid physical temperature within the Bering Sea ecosystem—a two-step spatial gap-filling protocol was applied prior to extraction:Primary Gap-Filling: A $3 \times 3$ focal window operation (terra::focal) was applied to the raster to smoothly interpolate missing values based on the mean of the immediately surrounding valid water temperatures. This maintained local temperature gradients and removed artificial boundaries.Secondary Imputation (Fail-Safe): If any spatial gaps were too large to be fully closed by the focal window, the remaining unmapped pixels were imputed using the domain-wide annual mean temperature for that specific year.
4. 
5. Zonal Extraction and DownscalingTemperatures were spatially aggregated from the gap-filled 10 km raster to the ViT's 25 km grid resolution. For each valid 25 km grid polygon, the mean of all underlying 10 km temperature pixels was calculated (terra::extract). Grid cells falling entirely outside the spatial footprint of the temperature data were initialized to zero.4. Array Construction and Temporal MaskingThe extracted 1D temperature vectors were mapped back to their deterministic (row, column) coordinates to populate a padded $50 \times 50$ spatial matrix for each year. These matrices were stacked to create a final 3D tensor with dimensions [36, 50, 50], representing the continuous temporal sequence from 1988 to 2023. To maintain consistency with the survey data gaps, the entire spatial matrix for the year 2020 (canceled survey) was explicitly masked with zeros. The final array was exported in .npy format for native ingestion as an environmental channel in the PyTorch data pipeline.