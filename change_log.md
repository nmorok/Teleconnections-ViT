# change log

#### 2/18/2026
Created the new_data_prep branch.
Changes:
- Changed the data process to mix the normalized GMRF before I perform non-linear transformations on them, so I could preserve the correlation between spawners and recruits. It was getting lost the way I had it previously. 
- Added 3 different data configs
- Removed the temporal mask from the attention because it wasn't adding anything. Attention is patch to patch, whereas the mask is at the channel level, so had more to do with the patch embedding
- updated the create splits script to handle the three levels of data
-  