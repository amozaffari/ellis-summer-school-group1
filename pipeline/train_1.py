# train_sequence.py
import os, bisect, json
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler  # device-agnostic AMP for mixed-precision training

# ---------------- Dataset Helper Functions ----------------
def _open_da(path, varname, engine="netcdf4", robust_nan=True):
    """
    Utility to open a NetCDF file as an xarray.DataArray, perform basic cleaning,
    and ensure consistent formatting.
    """
    ds = xr.open_dataset(path, engine=engine)
    da = ds[varname]
    # If robust_nan is True, replace the dataset's specified fill value with NaN
    if robust_nan:
        fv = da.attrs.get("_FillValue", None)
        if fv is not None:
            da = da.where(da != fv)
    # Ensure latitude is sorted north-up for consistency
    lat_name = "lat" if "lat" in da.coords else "latitude"
    da = da.sortby(lat_name)
    # If there are 24 time steps (semi-monthly for a year), assign a simple integer coordinate
    if "time" in da.dims and da.sizes["time"] == 24:
        da = da.assign_coords(sample=("time", np.arange(24)))
    return da

def _infer_var(nc_path, engine="netcdf4"):
    """
    Infers the name of the main data variable in a NetCDF file, assuming it's
    the variable with at least 2 dimensions (e.g., lat, lon).
    """
    with xr.open_dataset(nc_path, engine=engine) as ds:
        for v in ds.data_vars:
            if ds[v].ndim >= 2:
                return v
    raise RuntimeError(f"No suitable data variable found in {nc_path}")

# ---------------- PyTorch Dataset Class ----------------
class ERA5LAISequenceWorld(Dataset):
    """
    Multi-year, whole-world, sliding window dataset for PyTorch.

    This class reads climate (ERA5) and Leaf Area Index (LAI) data, creates
    time-based sequences, and prepares them as tensors for a model.

    X (features) layout can be:
      - "time_channels": (T*C, H, W) -> Time steps are merged into the channel dimension.
      - "time_first":    (T, C, H, W) -> Time remains a separate dimension.
    y (target) can be:
      - "last": (1, H, W) -> Predict only the last time step.
      - "all":  (T, 1, H, W) -> Predict every time step in the sequence.

    A mask is generated with the same shape as y (True where the target is valid).
    """
    def __init__(
        self, years, seq_len=48, seq_stride=24,
        era5_mode="anom", lai_mode="raw",
        feature_layout="time_channels", target_mode="last",
        paths=None, engine="netcdf4", robust_nan=True,
    ):
        super().__init__()
        # --- Assertions to catch configuration errors early ---
        assert era5_mode in ("raw","anom","z")
        assert lai_mode in ("raw","anom")
        assert feature_layout in ("time_channels","time_first")
        assert target_mode in ("last","all")

        # --- Store configuration parameters ---
        self.years = list(years)
        self.seq_len = int(seq_len)
        self.seq_stride = int(seq_stride)
        self.era5_mode = era5_mode
        self.lai_mode = lai_mode
        self.feature_layout = feature_layout
        self.target_mode = target_mode
        self.engine = engine
        self.robust_nan = robust_nan

        # --- Define and update data file paths ---
        defaults = {
            "era5_root": "/ptmp/mp002/ellis/lai",
            "era5_anom_dir": "/ptmp/mp040/outputdir/era5/anom",
            "lai_root": "/ptmp/mp002/ellis/lai/lai",
            "lai_tmpl": "LAI.1440.720.{year}.nc",
            "lai_anom_dir": "/ptmp/mp002/ellis/lai/anom",
        }
        self.paths = defaults if paths is None else {**defaults, **paths}

        # --- Load all specified years of data into memory ---
        self._open_years()

        # --- Calculate indices for creating sequences ---
        self.samples_per_year = 24  # Semi-monthly data
        self.year_offsets = [i * self.samples_per_year for i in range(len(self.years))]
        self.total_samples = self.samples_per_year * len(self.years)

        # Generate a list of all possible start indices for the sliding windows
        self.starts = list(range(0, self.total_samples - self.seq_len + 1, self.seq_stride))
        if not self.starts:
            raise ValueError("seq_len is longer than total samples. Reduce seq_len or add more years.")

    def _open_years(self):
        """Loads all necessary xarray DataArrays for the given years into memory."""
        self.ssrd_y, self.t2m_y, self.tp_y, self.lai_y = [], [], [], []
        for y in self.years:
            # --- Load ERA5 features based on the specified mode ---
            if self.era5_mode == "raw":
                f_ssrd = os.path.join(self.paths["era5_root"], "ssrd", f"ssrd.15daily.fc.era5.1440.720.{y}.nc")
                f_t2m  = os.path.join(self.paths["era5_root"], "t2m",  f"t2m.15daily.an.era5.1440.720.{y}.nc")
                f_tp   = os.path.join(self.paths["era5_root"], "tp",   f"tp.15daily.fc.era5.1440.720.{y}.nc")
                ssrd = _open_da(f_ssrd, "ssrd", self.engine, self.robust_nan)
                t2m  = _open_da(f_t2m,  "t2m",  self.engine, self.robust_nan)
                tp   = _open_da(f_tp,   "tp",   self.engine, self.robust_nan)
            else: # "anom" or "z" mode
                suffix = "anom" if self.era5_mode == "anom" else "z"
                base = self.paths["era5_anom_dir"]
                ssrd = _open_da(os.path.join(base, f"ssrd_{suffix}_{y}.nc"), f"ssrd_{suffix}", self.engine, self.robust_nan)
                t2m  = _open_da(os.path.join(base, f"t2m_{suffix}_{y}.nc"),  f"t2m_{suffix}",  self.engine, self.robust_nan)
                tp   = _open_da(os.path.join(base, f"tp_{suffix}_{y}.nc"),   f"tp_{suffix}",   self.engine, self.robust_nan)

            # --- Load LAI target data based on the specified mode ---
            lai_file = os.path.join(self.paths["lai_root"], self.paths["lai_tmpl"].format(year=y))
            lai_var  = _infer_var(lai_file, self.engine)
            lai = _open_da(lai_file, lai_var, self.engine, self.robust_nan)
            if self.lai_mode == "anom":
                lai_anom_file = os.path.join(self.paths.get("lai_anom_dir", self.paths["lai_root"]), f"LAI_anom_{y}.nc")
                if os.path.exists(lai_anom_file):
                    lai = _open_da(lai_anom_file, "LAI_anom", self.engine, self.robust_nan)
                else:
                    raise FileNotFoundError(f"LAI anomaly file not found: {lai_anom_file}")

            # --- Data Validation ---
            for da, name in [(ssrd,"ssrd"),(t2m,"t2m"),(tp,"tp")]:
                assert "time" in da.dims and da.sizes["time"] == 24, f"{name} {y} expects 24 samples"

            self.ssrd_y.append(ssrd); self.t2m_y.append(t2m); self.tp_y.append(tp); self.lai_y.append(lai)

        # --- Grid Consistency Check ---
        self.lat_name = "lat" if "lat" in self.lai_y[0].coords else "latitude"
        self.lon_name = "lon" if "lon" in self.lai_y[0].coords else "longitude"
        H = int(self.lai_y[0].sizes[self.lat_name]); W = int(self.lai_y[0].sizes[self.lon_name])
        for j in range(len(self.years)):
            for da, name in [(self.ssrd_y[j],"ssrd"),(self.t2m_y[j],"t2m"),(self.tp_y[j],"tp")]:
                assert da.sizes[self.lat_name] == H and da.sizes[self.lon_name] == W, f"{name} grid mismatch in year {self.years[j]}"

    def __len__(self):
        """Returns the total number of sequences that can be generated."""
        return len(self.starts)

    def _slice_from_global_t(self, global_t):
        """Maps a global time index (across all years) to a (year_index, local_time_index) pair."""
        # Find the correct year index (j) using binary search for efficiency
        j = bisect.bisect_right(self.year_offsets, global_t) - 1
        j = max(0, min(j, len(self.year_offsets)-1))
        # The local time index is the global index minus the offset of that year
        return j, global_t - self.year_offsets[j]

    def __getitem__(self, idx):
        """Fetches a single training sequence (X, y, mask, meta) by its index."""
        start, Ts = self.starts[idx], self.seq_len
        X_list, y_list = [], []

        # --- Loop over the sequence length to gather all time steps ---
        for t in range(Ts):
            g = start + t
            yi, lt = self._slice_from_global_t(g) # Get year index (yi) and local time index (lt)
            x_ssrd = self.ssrd_y[yi].isel(time=lt).values
            x_t2m  = self.t2m_y [yi].isel(time=lt).values
            x_tp   = self.tp_y  [yi].isel(time=lt).values
            # Stack the 3 features (channels) for this time step
            X_list.append(np.stack([x_ssrd, x_t2m, x_tp], axis=0))
            # If target_mode is 'all', we collect a target for every input time step
            if self.target_mode == "all":
                y_map = self.lai_y[yi].isel(time=lt).values
                y_list.append(np.expand_dims(y_map, axis=0))

        # Stack all time steps to form the full input sequence
        X_seq = np.stack(X_list, axis=0)  # Shape: (T, 3, H, W)

        # --- Prepare the target (y) and mask ---
        if self.target_mode == "last":
            # Target is the LAI at the final time step
            yi, lt = self._slice_from_global_t(start + Ts - 1)
            y_map = self.lai_y[yi].isel(time=lt).values
            y = np.expand_dims(y_map, axis=0)            # (1,H,W)
            mask = ~np.isnan(y)                         # Mask is True where we have valid data
        else: # target_mode == "all"
            y = np.stack(y_list, axis=0)                 # (T,1,H,W)
            mask = ~np.isnan(y)

        # Replace any NaNs in the input features with 0.0
        X_seq = np.nan_to_num(X_seq, nan=0.0)

        # --- Reshape features based on the specified layout ---
        if self.feature_layout == "time_channels":
            # Reshape from (T,3,H,W) to (3T,H,W)
            X = X_seq.transpose(1,0,2,3).reshape(-1, X_seq.shape[2], X_seq.shape[3])
        else: # "time_first"
            X = X_seq

        # --- Convert numpy arrays to PyTorch tensors ---
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(np.nan_to_num(y, nan=0.0).astype(np.float32))
        m_t = torch.from_numpy(mask.astype(np.bool_))

        # --- Prepare metadata (optional, useful for debugging) ---
        t_indices = list(range(start, start+Ts))
        years_cov = sorted(set(self.years[self._slice_from_global_t(t)[0]] for t in t_indices))
        meta = {"start_index": int(start), "t_indices": t_indices, "years": years_cov}
        return X_t, y_t, m_t, meta

def masked_mse_loss(pred, target, mask):
    """Calculates Mean Squared Error only on valid pixels specified by the mask."""
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask.float() # Apply mask
    denom = mask.float().sum().clamp_min(1.0) # Avoid division by zero
    return diff2.sum() / denom

def collate_keep_meta(batch):
    """Custom collate function to handle the metadata dictionary from __getitem__."""
    Xs, ys, ms, metas = zip(*batch)
    return torch.stack(Xs), torch.stack(ys), torch.stack(ms), list(metas)

# ---------------- Tiny CNN Model ----------------
class TinyCNN(nn.Module):
    """
    A very simple 2D CNN. Expects input where time steps are concatenated
    along the channel dimension (feature_layout='time_channels').
    So, in_ch = seq_len * num_base_features (which is 3).
    """
    def __init__(self, in_ch, mid=16, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),   nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, kernel_size=1)
        )
    def forward(self, x):
        return self.net(x)

# ---------------- Main Training Script ----------------
if __name__ == "__main__":
    # --- Load Configuration ---
    # Resolve config path relative to this script for robustness
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: # Fallback for interactive environments like Jupyter
        script_dir = os.getcwd()
    config_path = os.path.join(script_dir, "..", "inputs", "training.json")

    with open(config_path, "r") as f:
        cfg = json.load(f)
        print("--- Configuration Loaded ---")
        print(json.dumps(cfg, indent=2))
        print("----------------------------")

    # --- Initialize Datasets ---
    print("Initializing training dataset...")
    train_ds = ERA5LAISequenceWorld(
        years=cfg["splits"]["train_years"],
        seq_len=cfg["training"]["seq_len_samples"],
        seq_stride=cfg["training"]["seq_stride"],
        era5_mode=cfg["data"]["era5_mode"],
        lai_mode=cfg["data"]["lai_mode"],
        feature_layout=cfg["training"]["feature_layout"],
        target_mode=cfg["training"]["target_mode"],
        paths=cfg["data"]["paths"],
        engine=cfg["data"]["engine"],
    )
    print("Initializing validation dataset...")
    val_ds = ERA5LAISequenceWorld(
        years=cfg["splits"]["val_years"],
        seq_len=cfg["training"]["seq_len_samples"],
        seq_stride=cfg["training"]["seq_stride"],
        era5_mode=cfg["data"]["era5_mode"],
        lai_mode=cfg["data"]["lai_mode"],
        feature_layout=cfg["training"]["feature_layout"],
        target_mode=cfg["training"]["target_mode"],
        paths=cfg["data"]["paths"],
        engine=cfg["data"]["engine"],
    )
    print(f"Training set has {len(train_ds)} samples. Validation set has {len(val_ds)} samples.")

    # --- Initialize DataLoaders ---
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=cfg["training"]["shuffle"], num_workers=0,
        collate_fn=collate_keep_meta, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=0,
        collate_fn=collate_keep_meta, pin_memory=False
    )

    # --- Initialize Model & Device ---
    # Get the input shape from the first batch
    X0, _, _, _ = next(iter(train_loader))
    in_channels = X0.shape[1]
    model = TinyCNN(in_ch=in_channels)
    print(f"Model initialized with {in_channels} input channels.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True # Performance optimization
    print(f"Using device: {device}")


    # --- Optimizer and Automatic Mixed Precision (AMP) Scaler ---
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # GradScaler is used for mixed-precision training to prevent underflow
    scaler = GradScaler() if device.type == "cuda" else None

    # --- Training Loop ---
    EPOCHS = 2
    ckpt_path = os.path.join(script_dir, "tinycnn_ckpt.pt")
    loss_history = {"train": [], "val": []} # To store loss values

    def epoch_loop(loader, train=True):
        """A full pass over the dataset (an epoch)."""
        model.train(train) # Set model to train or evaluation mode
        total_loss, n_batches = 0.0, 0
        for X, y, mask, metas in loader:
            # Move data to the target device
            non_blocking = (device.type == "cuda")
            X = X.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
            mask = mask.to(device, non_blocking=non_blocking)

            # Context manager to enable/disable gradients
            with torch.set_grad_enabled(train):
                # Autocast for mixed-precision on CUDA
                with autocast(device_type=device.type, enabled=(scaler is not None)):
                    pred = model(X)
                    loss = masked_mse_loss(pred, y, mask)
                
                # If training, perform backpropagation and update weights
                if train:
                    opt.zero_grad(set_to_none=True)
                    if scaler is not None:
                        scaler.scale(loss).backward() # Scale loss
                        scaler.step(opt)              # Unscale gradients and update optimizer
                        scaler.update()               # Update scale for next iteration
                    else: # If not using CUDA/scaler
                        loss.backward()
                        opt.step()

            total_loss += float(loss.detach()) # Accumulate loss
            n_batches += 1
        return total_loss / max(n_batches, 1) # Return average loss for the epoch

    print("\n--- Starting Training ---")
    for epoch in range(1, EPOCHS + 1):
        # Run one epoch of training
        train_loss = epoch_loop(train_loader, train=True)
        loss_history["train"].append(train_loss)
        
        # Run one epoch of validation
        val_loss = epoch_loop(val_loader, train=False)
        loss_history["val"].append(val_loss)

        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # --- Save Final Model ---
    torch.save({"model": model.state_dict(), "in_channels": in_channels}, ckpt_path)
    print(f"\n--- Training Complete ---")
    print(f"Saved final model checkpoint to: {ckpt_path}")

    # --- Display Loss Progression ---
    print("\n--- Loss Progression ---")
    print("Epoch | Train Loss | Val Loss")
    print("----------------------------")
    for i in range(EPOCHS):
        print(f"{i+1:^5} | {loss_history['train'][i]:<10.6f} | {loss_history['val'][i]:<10.6f}")
    print("----------------------------")
