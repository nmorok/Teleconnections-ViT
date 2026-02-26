# change log

#### 2/18/2026
Created the new_data_prep branch.
Changes:
- Changed the data process to mix the normalized GMRF before I perform non-linear transformations on them, so I could preserve the correlation between spawners and recruits. It was getting lost the way I had it previously. 
- Added 3 different data configs
- Removed the temporal mask from the attention because it wasn't adding anything. Attention is patch to patch, whereas the mask is at the channel level, so had more to do with the patch embedding
- updated the create splits script to handle the three levels of data
- updated the data helper to support multiple data difficulties.
- updated the training file to handle all 3 data difficulties.  

#### 2/19/2026
Changes:
- updated the training file plot scripts to include all three difficulties and loss criterion
- updated the training file to include loss criterion selection
- updated the training file's spatial decoder bias to reflect the data better. There was a problem where I was significantly underpredicting values. Mean of like 3 compared to mean of 8 (in log space)
- Added a softplus to the end of the decoder in the model.py file. 
- Updated the data helper to stop it from reseting the year index between train, val, and test. 

#### 2/20/2026
Thought:
- Given that there is an autocorrelation, I think it would be useful to give more to the model. In the sense that if there is a 5 year lag built in, I should give more than 5 years. since the 6th and 7th years have information that might be useful to the model. 
- Do I even need the recruits from previous year? Like all 5 years, or is it adding noise?
- What about that memory bank idea. Like all of the previous years and the spawners and recruits. 
- https://onlinelibrary.wiley.com/doi/full/10.1111/geb.70184 -- this is the paper Maia sent.