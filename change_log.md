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
- 