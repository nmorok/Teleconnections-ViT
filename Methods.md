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

### create_splits.py
**Idea**
Split the 100 bootstraps of 30 year data into training, validation, and testing.

**Steps**
1. split the data by the year ranges given
2. data is formatted by B0Y0, B0Y1, B0Y2 ... 

**Parameters**
- training years: 22
- validation years: 5
- testing years: 3

### data_helper.py
**Idea**
Give the training/model the correct batches of data. 
With the historical data, bridge the data between training, validation, and testing sets. For the early years, add a temporal mask.
Also, transform the data if need be for better model performance.

**Steps**
1. load the training data
   1. transform the data if desired
      1. currently support log transform, and max standardization
      2. if transformation = log, take $log(1+density)$
      3. if transformation = max, calculate the global max across all bootstraps in the training data only. Save that value and use it to transform the validation and testing data as well. Transform via $density/density_{max}$
2. load the historical data if available
   1. get the data from the previous years (all bootstraps) and add it to the channel data
   2. for the training data where the years are less than the lag, only give as much data is available, and then mask out the missing years using a tenor vector that is 0 for missing years and 1 for valid years, including the current year. The historical year channels that are not valid are set to 0 everywhere and still passed, since the model needs full channel data. The mask will handle telling the model the data is empty/padding instead of all 0s.
   3. to ensure that the historical data is available across validation, and testing sets, there is a bridge function to get the historical data from either the training or validation set. 
3. Calls the torch dataloader which:
   1. creates a random permutation of indices if shuffle = true.
   2. takes the first batchsize indices and calls __getitem__ from my function
   3. appends the result to a samples vector. 
   4. returns that vector (batch size long)
   5. continue until all samples have been used.

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
        historical_idx = 65 * 22 + 1 = 1431
        memory_spawners[0] = self.spawners[1431]
        temporal_mask[1] = 1.0  # ✓
    
    # i=1: historical_year=0 → EXISTS in current split ✓
    elif historical_year >= 0:
        historical_idx = 65 * 22 + 0 = 1430
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
2. Process each channel seperately
   1. apply a learned linear transformation to go from dim 25 to dim 128
      1. Using nn.Linear
   2. normalize the vector using LayerNorm -- $LN(x) = \gamma * ((x-\mu) / \sqrt(\sigma^2+\epsilon)) + \beta$, where $\gamma$ and $\beta$ are learnable scale and shift parameters, and $\epsilon$ is a small constant for numerical stability.
   3. apply dropout
3. each channel gets processed into a 12 embedding dimensionality vector. [B, 100, 25] -> [B, 100, 12]
4. multiply the channel by the mask (1.0 if valid, 0 if invalid)
5. concatonate all of the channel embeddings together so now we have [B, 100, 132]
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
5. Add the temporal encodiing using the TemporalEmbedding script
6. Apply dropout to the embeddings (what am I dropping out? which parameters? it just zeros out some of the parameters right? which ones?)
7. Run the full transformer block using the TransformerBlock script num_layers times
8. normalize the output of the transformer blocks 
   1. using layernorm
9.  Get output using the SpatialDecoder script
   1.  Before calling the decoder, first need to reshape into the correct order
10. Apply softplus
    1.  To constrain the output to be greater than 0 and uncapped.
    2.  Smooth function
11. return the predicted 50 x 50 grid

**Parameters**
- grid_size: 50
- patch_size: 5
- in_channels: 11
- embed_dim: 128
- num_heads: 8
- num_layers: 6
- d_ff: 512
- dropout: 0.1

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
4. if reduction = 'mean', return the mean of the loss across all pixels; if reduction = 'sum', return the sum of the loss across all pixels; if reduction = None, return the loss for each pixel seperately.

**Parameters**
- power: 1.5
- reduction: 'mean'

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
   4. This is how AdamW works: ...
9. Initialize the scheduler (OneCycleLR)
   1. It updates the learning rate for every batch, not just once per epoch. This creates a smooth curve for the learning rate, making training much more stable.
   2. Chose OneCycleLR to help the model avoid the identity trap, which is when the model gets stuck predicting the average density. The OnceCycleLR works by starting with a very low learning rate to find the correct direction, then ramps up to a high rate to jump over local minima and finally slows down to fine-tune pixels.
   3. This is how the scheduler actually works: ...
10. For every input, target pair in the training loader (so this would be for every pair in a batch)
    1.  zero out the gradient
    2.  Forward model pass
    3.  Compute the loss
    4.  Move backwards and compute the gradient
    5.  clip the gradients to avoid exploding gradients
    6.  optimize the weights
    7.  run the scheduler # what does this do?
11. compute the average training loss across all batches
12. change the model into evaluation mode (doesn't compute gradients)
13. for every input, target pair in the validation loader (every batch)
    1.  Forward model pass
    2.  Compute the loss
14. compute the average validation loss across all batches
15. save the learning rate, the training loss, the validation loss
16. save the model if the validation error is less than the current model's validation error
17. increase patience counter if no improvement has occured
18. Once all epochs have run or we stopped from earlystopping, load the best model
19. evaluate the model on the test data. 
20. plot the training curves

**Parameters**

data:
- batch_size: 8
- memory_years: 5
- train_years: 22
- val_years: 5
- test_years: 3
- transformation: 'log'

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
- reduction: 'mean'

optimizer:
- learning rate: 1e-4
- weight decay: 1e-4

scheduler:
- max learning rate: 5e-4
- epochs: 50
- steps_per_epoch: len(train_loader) # I think this might be batches?
- pct_start: 0.1
- anneal_strategy: 'cos'
- div_factor: 10 # inital_lr = max_lr / 10
- final_div_factor: 100 # final_lr = max_lr / 100

epochs:
- epochs: 50
- early_stop_patience: 10

gradient clipping: 
- max: 1.0






#### to-dos:
- Add 2 more in_channels representing lat and lon. Ideally would also add cold pool, but we won't have that for future predictions, unless we gave it summer (off-season) cold pool or something like that. 
- Add a spatial mask for when we actually use the real data. 