# Development Workflow Draft

## Data

## Model specs

## Training


## Pipeline skeleton
Dummy data -- file: generate_dummy_data.py; functions: create_dummy_crab_data(), create_smooth_random_field()

Data splits -- file: create_train_test_splits.py; functions: create_temporal_splits()




### generate_dummy_data.py
Goal of this script is to make dummy data to benchmark the pipeliine and make sure it works. 

#### create_dummy_crab_data() 
This function serves to create the dummy data. 
It uses random number generator to generate numbers between 0-100 to simulate crab densities. Optional spatial smoother to enfore spatial autocorrelation and give the model something to 'find'. 
Correlation between spawners and recruitment given by a weighted sum of the spawner grid (0.7) and the recruitment grid (0.3).
Uncertainty tries to mimic the edge effects of SPDE, where there is a decrease in data quality as we move away from our station survey area. 

Inputs: 
- n_years: number of years you want data for (int)
- n_bootstraps: number of bootstrap replicates you want (int)
- grid_size: how large of a grid you want (int)
- spatial_smoothing: T/F if you want spatial smoothing (boolean)

Outputs: 
spawners, recruitment, uncertainty

#### create_smooth_random_field()
This function serves to generate a spatioally autocorrelated field using a gaussian kernal with smoothness sigma, and mimics the SPDE matern covariance output. 

Inputs: 
- grid_size: how large of a grid you want (int)
- smoothness: The sigma of the gaussian kernel (int)

Outputs: 
spatial_field