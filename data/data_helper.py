"""
data_helper.py

Changes from previous version
------------------------------
* Added `use_spawners` and `use_recruits` boolean flags to both CrabDataset
  and get_dataloaders.  Combined with the existing `use_temp`, this gives 7
  channel-group configurations (all combinations of the three groups, minus
  the empty set).
* Input tensor is built dynamically from only the selected groups.
* CrabDataset.channel_mask_indices  (list[int], length = in_channels)
  is computed at __init__ time and stored on the dataset object.  Each entry
  is an index into the temporal mask vector (0 = current year, 1-5 = lookback
  1-5).  Pass this list directly to CrabTransformer so PatchEmbedding can
  apply masking without any hardcoded channel-range logic.
* get_channel_info() utility: given the three flags, returns
  (in_channels, channel_mask_indices).  Useful for constructing the model
  before building the dataset.

Spawner group  (6 channels when use_spawners=True):
  ch 0 : current spawner (zeroed + mask[0]=0 in one_year_ahead mode)
  ch 1-5: spawner t-1 … t-5

Recruit group  (5 channels when use_recruits=True):
  ch +0 : recruit t-1
  ch +1-4: recruit t-2 … t-5

Temperature group  (6 channels when use_temp=True and temp file exists):
  ch +0 : current temp  (zeroed when year_mask=0)
  ch +1-5: temp t-1 … t-5
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ============================================================
#  UTILITY: derive channel metadata from group flags
# ============================================================

def get_channel_info(use_spawners: bool, use_recruits: bool, use_temp: bool,
                     include_current: bool = True):
    """
    Returns (in_channels, channel_mask_indices) for the given combination of
    flags.

    include_current controls whether current-year channels are present in the
    tensor.  Set to False for one_year_ahead prediction mode.

    * Spawner group:  current (opt.) + 5 history  →  6ch or 5ch
    * Recruit group:  5 history only              →  always 5ch
                      (recruits have no current channel by design —
                       this group is inherently one-year-ahead)
    * Temperature group: current (opt.) + 5 history → 6ch or 5ch
                         mirrors spawner: if include_current=False, current
                         temp is also excluded because it belongs to the same
                         survey year as the current spawner.

    channel_mask_indices[c] = which slot of the temporal mask vector (length 6)
    to use when masking channel c in PatchEmbedding.
      0 → current year validity
      1 → t-1 validity  …  5 → t-5 validity

    Examples  (include_current=True)
    ----------------------------------
    all           : [0,1,2,3,4,5, 1,2,3,4,5, 0,1,2,3,4,5]  in_channels=17
    sp_rec        : [0,1,2,3,4,5, 1,2,3,4,5]                in_channels=11
    sp_temp       : [0,1,2,3,4,5, 0,1,2,3,4,5]              in_channels=12
    rec_temp      : [1,2,3,4,5,   0,1,2,3,4,5]              in_channels=11
    spawners_only : [0,1,2,3,4,5]                            in_channels=6
    recruits_only : [1,2,3,4,5]                              in_channels=5
    temp_only     : [0,1,2,3,4,5]                            in_channels=6

    Examples  (include_current=False  —  one_year_ahead mode)
    -----------------------------------------------------------
    all           : [1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5]       in_channels=15
    sp_rec        : [1,2,3,4,5, 1,2,3,4,5]                   in_channels=10
    sp_temp       : [1,2,3,4,5, 1,2,3,4,5]                   in_channels=10
    rec_temp      : [1,2,3,4,5, 1,2,3,4,5]                   in_channels=10  (same — rec has no current)
    spawners_only : [1,2,3,4,5]                               in_channels=5
    recruits_only : [1,2,3,4,5]                               in_channels=5   (unchanged)
    temp_only     : [1,2,3,4,5]                               in_channels=5
    """
    idx = []
    if use_spawners:
        if include_current:
            idx.append(0)                  # current spawner → mask slot 0
        idx.extend([1, 2, 3, 4, 5])       # spawner history
    if use_recruits:
        idx.extend([1, 2, 3, 4, 5])       # recruit history (no current channel)
    if use_temp:
        if include_current:
            idx.append(0)                  # current temp → mask slot 0
        idx.extend([1, 2, 3, 4, 5])       # temp history
    if not idx:
        raise ValueError("At least one channel group must be enabled.")
    return len(idx), idx


# ============================================================
#  DATASET
# ============================================================

class CrabDataset(Dataset):
    def __init__(self, spawner_path, recruit_path, temp_path=None,
                 n_years=30, memory_years=5,
                 historical_spawners=None,
                 historical_recruits=None,
                 historical_temps=None,
                 year_offset=0, mask_path=None,
                 year_mask=None, historical_year_mask=None,
                 include_current_spawner=True,
                 use_spawners=True,
                 use_recruits=True):
        """
        Parameters
        ----------
        use_spawners : bool
            Include the 6-channel spawner group (current + 5-year history).
        use_recruits : bool
            Include the 5-channel recruit history group (t-1 … t-5).
        use_temp is determined by whether temp_path is provided.

        After __init__, read:
            self.channel_mask_indices  — pass to CrabTransformer(channel_mask_indices=…)
            self.in_channels           — pass to CrabTransformer(in_channels=…)
        """
        self.year_offset            = year_offset
        self.n_years                = n_years
        self.memory_years           = memory_years
        self.include_current_spawner = include_current_spawner
        self.use_spawners           = use_spawners
        self.use_recruits           = use_recruits

        # -- Load arrays -------------------------------------------------------
        self.spawners = np.load(spawner_path).astype(np.float32)
        self.recruits = np.load(recruit_path).astype(np.float32)

        self.historical_spawners = historical_spawners
        self.historical_recruits = historical_recruits
        self.historical_temps    = historical_temps

        if temp_path is not None:
            self.temps = np.load(temp_path).astype(np.float32)
        else:
            self.temps = None

        self.use_temp = (self.temps is not None)

        # -- Validate -----------------------------------------------------------
        total_samples = len(self.spawners)
        assert total_samples % n_years == 0, \
            f"Total samples ({total_samples}) must be divisible by n_years ({n_years})"
        self.n_bootstraps = total_samples // n_years

        if self.historical_spawners is not None:
            assert self.historical_spawners.shape[0] == self.n_bootstraps
            assert self.historical_spawners.shape[1] == memory_years

        print(f"Loaded {self.n_bootstraps} bootstraps × {self.n_years} years | "
              f"spawners={use_spawners}  recruits={use_recruits}  temp={self.use_temp}")
        if self.historical_spawners is not None:
            print(f"  Using {memory_years} years of historical data from previous split")

        # -- Spatial mask -------------------------------------------------------
        if mask_path is not None:
            self.mask = np.load(mask_path).astype(np.float32)
            self.spawners[:, self.mask == 0] = 0.0
            self.recruits[:, self.mask == 0] = 0.0
            if self.temps is not None:
                self.temps[:, self.mask == 0] = 0.0
        else:
            self.mask = np.ones((50, 50), dtype=np.float32)

        # -- Log-transform (spawners and recruits only) -------------------------
        print("Applying log1p scaling to spawners / recruits.")
        self.spawners = np.log1p(self.spawners)
        self.recruits = np.log1p(self.recruits)
        # Temperatures are NOT log-transformed (negative values exist)

        # -- Temporal / year masks ----------------------------------------------
        self.year_mask = year_mask if year_mask is not None else \
                         np.ones(n_years, dtype=np.float32)
        self.historical_year_mask = historical_year_mask if historical_year_mask is not None else \
                                    np.ones(memory_years, dtype=np.float32)

        # -- Channel metadata ---------------------------------------------------
        # include_current_spawner drives whether current-year channels (spawner
        # AND temperature) appear in the tensor.  recruits_only is unaffected
        # since recruits never have a current channel.
        self.in_channels, self.channel_mask_indices = get_channel_info(
            self.use_spawners, self.use_recruits, self.use_temp,
            include_current=self.include_current_spawner,
        )
        print(f"  in_channels={self.in_channels}  "
              f"channel_mask_indices={self.channel_mask_indices}")

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.spawners)

    def __getitem__(self, idx):
        bootstrap_idx    = idx // self.n_years
        relative_year    = idx  % self.n_years
        year_idx         = self.year_offset + relative_year

        # ---- Memory banks -----------------------------------------------
        memory_spawners = np.zeros((self.memory_years, 50, 50), dtype=np.float32)
        memory_recruits = np.zeros((self.memory_years, 50, 50), dtype=np.float32)
        memory_temps    = np.zeros((self.memory_years, 50, 50), dtype=np.float32)
        # temporal_mask length = memory_years + 1 = 6
        # index 0 = current year, indices 1-5 = lookback 1-5
        temporal_mask = np.zeros(self.memory_years + 1, dtype=np.float32)

        # ---- Current spawner ------------------------------------------------
        # When include_current_spawner=True  → load real grid, set mask[0]
        # When include_current_spawner=False → current spawner is NOT added
        #   to the tensor at all (channel dropped, not zeroed).  current_temp
        #   is also dropped for the same reason — both belong to the current
        #   survey year which is unavailable in one-year-ahead mode.
        if self.use_spawners and self.include_current_spawner:
            current_spawner = self.spawners[idx]
            temporal_mask[0] = self.year_mask[relative_year]
        else:
            current_spawner = None  # excluded from tensor

        current_recruit = self.recruits[idx]  # always loaded (target)

        # ---- Historical window ------------------------------------------
        for i in range(self.memory_years):
            lookback = i + 1
            target_rel = relative_year - lookback

            if target_rel >= 0:
                if self.year_mask[target_rel] == 1.0:
                    local_idx = bootstrap_idx * self.n_years + target_rel
                    memory_spawners[i] = self.spawners[local_idx]
                    memory_recruits[i] = self.recruits[local_idx]
                    if self.temps is not None:
                        memory_temps[i] = self.temps[local_idx]
                    temporal_mask[i + 1] = 1.0
                # else: 2020 gap — leave as zeros, mask stays 0

            elif self.historical_spawners is not None:
                hist_idx = self.memory_years + target_rel
                if hist_idx >= 0 and self.historical_year_mask[hist_idx] == 1.0:
                    memory_spawners[i] = self.historical_spawners[bootstrap_idx, hist_idx]
                    memory_recruits[i] = self.historical_recruits[bootstrap_idx, hist_idx]
                    if self.historical_temps is not None:
                        memory_temps[i] = self.historical_temps[bootstrap_idx, hist_idx]
                    temporal_mask[i + 1] = 1.0

        # ---- Current temperature ----------------------------------------
        # Excluded from tensor when include_current_spawner=False (same rule
        # as spawners: we don't have this year's survey data in that mode).
        if self.use_temp and self.include_current_spawner:
            current_temp = self.temps[idx]
            if self.year_mask[relative_year] == 0.0:
                current_temp = np.zeros_like(current_temp)
            else:
                temporal_mask[0] = 1.0  # set current-year slot if spawners didn't
        else:
            current_temp = None

        # ---- Build input tensor dynamically from selected groups --------
        # current_spawner / current_temp are None when include_current_spawner=False
        # — those channels are simply absent from the tensor in that mode.
        tensor_list = []

        if self.use_spawners:
            if current_spawner is not None:        # normal mode
                tensor_list.append(
                    torch.tensor(current_spawner, dtype=torch.float32).unsqueeze(0)
                )
            tensor_list.append(                    # history always included
                torch.tensor(memory_spawners, dtype=torch.float32)
            )

        if self.use_recruits:
            tensor_list.append(
                torch.tensor(memory_recruits, dtype=torch.float32)
            )

        if self.use_temp:
            if current_temp is not None:           # normal mode
                tensor_list.append(
                    torch.tensor(current_temp, dtype=torch.float32).unsqueeze(0)
                )
            tensor_list.append(                    # history always included
                torch.tensor(memory_temps, dtype=torch.float32)
            )

        input_tensor        = torch.cat(tensor_list, dim=0)
        target_tensor       = torch.tensor(current_recruit,  dtype=torch.float32).unsqueeze(0)
        temporal_mask_tensor = torch.tensor(temporal_mask,   dtype=torch.float32)

        # per-sample validity flag
        # A sample is invalid (excluded from loss) only if it has NO data at all:
        # in one-year-ahead mode that means all history slots (mask[1..5]) are 0.
        # We check history slots only (mask[1:]) because slot 0 (current year)
        # is never set when include_current_spawner=False.
        valid_year = torch.tensor(self.year_mask[relative_year], dtype=torch.float32)
        if not self.include_current_spawner and temporal_mask[1:].sum() == 0:
            valid_year = torch.tensor(0.0, dtype=torch.float32)

        return (input_tensor, target_tensor, temporal_mask_tensor,
                torch.tensor(year_idx, dtype=torch.long),
                torch.tensor(self.mask, dtype=torch.float32),
                valid_year)


# ============================================================
#  HISTORICAL CONTEXT EXTRACTION
# ============================================================

def get_last_n_years(spawners, recruits, n_bootstraps, n_years_total,
                     n_years_to_extract, temps=None):
    """Extract the last n_years_to_extract years from a split for use as
    historical context in the next split."""
    h_sp = np.zeros((n_bootstraps, n_years_to_extract, 50, 50), dtype=np.float32)
    h_rc = np.zeros((n_bootstraps, n_years_to_extract, 50, 50), dtype=np.float32)
    h_tp = np.zeros((n_bootstraps, n_years_to_extract, 50, 50), dtype=np.float32) \
           if temps is not None else None

    start = n_years_total - n_years_to_extract
    for b in range(n_bootstraps):
        for y in range(n_years_to_extract):
            g = b * n_years_total + start + y
            h_sp[b, y] = spawners[g]
            h_rc[b, y] = recruits[g]
            if temps is not None:
                h_tp[b, y] = temps[g]

    return h_sp, h_rc, h_tp


# ============================================================
#  DATALOADER FACTORY
# ============================================================

def get_dataloaders(level='easy', batch_size=5, memory_years=5,
                    train_years=22, val_years=5, test_years=3,
                    data_type='dummy', include_current_spawner=True,
                    lag=0, use_temp=True,
                    use_spawners=True, use_recruits=True):
    """
    Parameters
    ----------
    use_spawners : include spawner channel group (6 ch)
    use_recruits : include recruit history channel group (5 ch)
    use_temp     : include temperature channel group (6 ch); ignored for dummy data
                   (no temp files exist for dummy data)
    """
    if data_type == 'real':
        lag_folder = f"lag{lag}" if lag > 0 else "nolag"
        data_dir   = f"data/real/splits/{lag_folder}/real/"
        mask_path  = "data/real/output/spatial_mask.npy"
        level      = 'real'
    else:
        data_dir  = f"data/dummy/splits/{level}/"
        mask_path = None
        use_temp  = False          # dummy data never has temp files

    # ---- Year masks ---------------------------------------------------
    train_year_mask = np.load(data_dir + f"train_year_mask_{level}.npy")
    val_year_mask   = np.load(data_dir + f"val_year_mask_{level}.npy")
    test_year_mask  = np.load(data_dir + f"test_year_mask_{level}.npy")

    n_bootstraps = len(np.load(data_dir + f"train_spawners_{level}.npy")) // train_years

    temp_train = data_dir + f"train_temp_{level}.npy" if use_temp else None
    temp_val   = data_dir + f"val_temp_{level}.npy"   if use_temp else None
    temp_test  = data_dir + f"test_temp_{level}.npy"  if use_temp else None

    ds_kwargs = dict(memory_years=memory_years, mask_path=mask_path,
                     include_current_spawner=include_current_spawner,
                     use_spawners=use_spawners, use_recruits=use_recruits)

    # 1. Train
    train_ds = CrabDataset(
        data_dir + f"train_spawners_{level}.npy",
        data_dir + f"train_recruits_{level}.npy",
        temp_train,
        n_years=train_years, year_offset=0,
        year_mask=train_year_mask,
        **ds_kwargs
    )

    # 2. Historical context for val
    t_h_sp, t_h_rc, t_h_tp = get_last_n_years(
        train_ds.spawners, train_ds.recruits,
        n_bootstraps, train_years, memory_years,
        train_ds.temps
    )
    train_hist_year_mask = train_year_mask[-memory_years:]

    # 3. Val
    val_ds = CrabDataset(
        data_dir + f"val_spawners_{level}.npy",
        data_dir + f"val_recruits_{level}.npy",
        temp_val,
        n_years=val_years, year_offset=train_years,
        historical_spawners=t_h_sp, historical_recruits=t_h_rc, historical_temps=t_h_tp,
        year_mask=val_year_mask, historical_year_mask=train_hist_year_mask,
        **ds_kwargs
    )

    # 4. Historical context for test
    v_h_sp, v_h_rc, v_h_tp = get_last_n_years(
        val_ds.spawners, val_ds.recruits,
        n_bootstraps, val_years, memory_years,
        val_ds.temps
    )
    val_hist_year_mask = val_year_mask[-memory_years:]

    # 5. Test
    test_ds = CrabDataset(
        data_dir + f"test_spawners_{level}.npy",
        data_dir + f"test_recruits_{level}.npy",
        temp_test,
        n_years=test_years, year_offset=train_years + val_years,
        historical_spawners=v_h_sp, historical_recruits=v_h_rc, historical_temps=v_h_tp,
        year_mask=test_year_mask, historical_year_mask=val_hist_year_mask,
        **ds_kwargs
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader