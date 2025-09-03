# train_optimized.py
import os, bisect, json, time
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# ---------------- Dataset Helper Functions (No changes here) ----------------
def _open_da(path, varname, engine="netcdf4", robust_nan=True):
    ds = xr.open_dataset(path, engine=engine)
    da = ds[varname]
    if robust_nan:
        fv = da.attrs.get("_FillValue", None)
        if fv is not None:
            da = da.where(da != fv)
    lat_name = "lat" if "lat" in da.coords else "latitude"
    da = da.sortby(lat_name)
    if "time" in da.dims and da.sizes["time"] == 24:
        da = da.assign_coords(sample=("time", np.arange(24)))
    return da

def _infer_var(nc_path, engine="netcdf4"):
    with xr.open_dataset(nc_path, engine=engine) as ds:
        for v in ds.data_vars:
            if ds[v].ndim >= 2:
                return v
    raise RuntimeError(f"No suitable data variable found in {nc_path}")

# ---------------- PyTorch Dataset Class (MODIFIED FOR PERFORMANCE) ----------------
class ERA5LAISequenceWorld(Dataset):
    def __init__(
        self, years, seq_len=48, seq_stride=24,
        era5_mode="anom", lai_mode="raw",
        feature_layout="time_channels", target_mode="last",
        paths=None, engine="netcdf4", robust_nan=True,
    ):
        super().__init__()
        # ... (initial parameters are the same) ...
        self.years = list(years)
        self.seq_len = int(seq_len)
        self.seq_stride = int(seq_stride)
        self.era5_mode = era5_mode
        self.lai_mode = lai_mode
        self.feature_layout = feature_layout
        self.target_mode = target_mode
        self.engine = engine
        self.robust_nan = robust_nan
        defaults = {
            "era5_root": "/ptmp/mp002/ellis/lai",
            "era5_anom_dir": "/ptmp/mp040/outputdir/era5/anom",
            "lai_root": "/ptmp/mp002/ellis/lai/lai",
            "lai_tmpl": "LAI.1440.720.{year}.nc",
            "lai_anom_dir": "/ptmp/mp002/ellis/lai/anom",
        }
        self.paths = defaults if paths is None else {**defaults, **paths}
        
        # This method now loads all data into RAM for speed
        self._open_and_load_years() # <-- MODIFIED

        self.samples_per_year = 24
        self.total_samples = self.lai_data.shape[0] # <-- MODIFIED to use the loaded data shape

        self.starts = list(range(0, self.total_samples - self.seq_len + 1, self.seq_stride))
        if not self.starts:
            raise ValueError("seq_len is longer than total samples. Reduce seq_len or add years.")

    def _open_and_load_years(self): # <-- MODIFIED: New optimized loading function
        """
        Loads all necessary data for the given years, concatenates them,
        and then loads everything into memory as NumPy arrays to prevent I/O bottlenecks.
        """
        ssrd_y, t2m_y, tp_y, lai_y = [], [], [], []
        print(f"    - Loading data for years: {self.years}...") # <-- NEW progress indicator
        for y in self.years:
            # ... (the logic to find and open individual files is the same) ...
            if self.era5_mode == "raw":
                f_ssrd = os.path.join(self.paths["era5_root"], "ssrd", f"ssrd.15daily.fc.era5.1440.720.{y}.nc")
                f_t2m  = os.path.join(self.paths["era5_root"], "t2m",  f"t2m.15daily.an.era5.1440.720.{y}.nc")
                f_tp   = os.path.join(self.paths["era5_root"], "tp",   f"tp.15daily.fc.era5.1440.720.{y}.nc")
                ssrd = _open_da(f_ssrd, "ssrd", self.engine, self.robust_nan)
                t2m  = _open_da(f_t2m,  "t2m",  self.engine, self.robust_nan)
                tp   = _open_da(f_tp,   "tp",   self.engine, self.robust_nan)
            else:
                suffix = "anom" if self.era5_mode == "anom" else "z"
                base = self.paths["era5_anom_dir"]
                ssrd = _open_da(os.path.join(base, f"ssrd_{suffix}_{y}.nc"), f"ssrd_{suffix}", self.engine, self.robust_nan)
                t2m  = _open_da(os.path.join(base, f"t2m_{suffix}_{y}.nc"),  f"t2m_{suffix}",  self.engine, self.robust_nan)
                tp   = _open_da(os.path.join(base, f"tp_{suffix}_{y}.nc"),   f"tp_{suffix}",   self.engine, self.robust_nan)
            lai_file = os.path.join(self.paths["lai_root"], self.paths["lai_tmpl"].format(year=y))
            lai_var  = _infer_var(lai_file, self.engine)
            lai = _open_da(lai_file, lai_var, self.engine, self.robust_nan)
            if self.lai_mode == "anom":
                lai_anom_file = os.path.join(self.paths.get("lai_anom_dir", self.paths["lai_root"]), f"LAI_anom_{y}.nc")
                lai = _open_da(lai_anom_file, "LAI_anom", self.engine, self.robust_nan)
            
            ssrd_y.append(ssrd); t2m_y.append(t2m); tp_y.append(tp); lai_y.append(lai)
        
        # --- NEW OPTIMIZATION STEP ---
        # Concatenate all years into single xarray DataArrays
        print("    - Concatenating yearly data...")
        full_ssrd = xr.concat(ssrd_y, dim="time")
        full_t2m  = xr.concat(t2m_y, dim="time")
        full_tp   = xr.concat(tp_y, dim="time")
        full_lai  = xr.concat(lai_y, dim="time")

        # Now, load all the data from disk into RAM. This is the key performance gain.
        # This will be slow once, but makes training much faster.
        print("    - Loading all data into memory (this may take a moment)...")
        self.ssrd_data = full_ssrd.load().values
        self.t2m_data  = full_t2m.load().values
        self.tp_data   = full_tp.load().values
        self.lai_data  = full_lai.load().values
        print("    - ✅ Data loaded.")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx): # <-- MODIFIED: New optimized __getitem__
        """Fetches a single training sequence by slicing the pre-loaded NumPy arrays."""
        start = self.starts[idx]
        end = start + self.seq_len

        # --- SLICE pre-loaded NumPy arrays (extremely fast) ---
        # Shape of each is (T, H, W) where T=seq_len
        x_ssrd_seq = self.ssrd_data[start:end, :, :]
        x_t2m_seq  = self.t2m_data[start:end, :, :]
        x_tp_seq   = self.tp_data[start:end, :, :]

        # Stack the features to get (T, 3, H, W)
        X_seq = np.stack([x_ssrd_seq, x_t2m_seq, x_tp_seq], axis=1)

        # --- Prepare the target (y) and mask from pre-loaded data ---
        if self.target_mode == "last":
            # Target is the LAI at the final time step of the sequence
            y = self.lai_data[end - 1, :, :] # Get the last slice
            y = np.expand_dims(y, axis=0)    # Add channel dim -> (1, H, W)
            mask = ~np.isnan(y)
        else: # target_mode == "all"
            y = self.lai_data[start:end, :, :]
            y = np.expand_dims(y, axis=1) # Add channel dim -> (T, 1, H, W)
            mask = ~np.isnan(y)
        
        X_seq = np.nan_to_num(X_seq, nan=0.0)

        # --- Reshape features based on layout (no changes here) ---
        if self.feature_layout == "time_channels":
            X = X_seq.transpose(0, 2, 3, 1).reshape(X_seq.shape[2], X_seq.shape[3], -1)
            X = X.transpose(2, 0, 1) # to get C,H,W -> (3T, H, W)
        else:
            X = X_seq

        # --- Convert to PyTorch tensors (no changes here) ---
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(np.nan_to_num(y, nan=0.0).astype(np.float32))
        m_t = torch.from_numpy(mask.astype(np.bool_))

        # We can simplify the metadata as we don't need to track years anymore
        meta = {"start_index": int(start)}
        return X_t, y_t, m_t, meta

# ... (The rest of the script, including TinyCNN, collate_fn, and the main training loop,
#      remains exactly the same as the previous version with all the print statements.) ...

def masked_mse_loss(pred, target, mask):
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return diff2.sum() / denom

def collate_keep_meta(batch):
    Xs, ys, ms, metas = zip(*batch)
    return torch.stack(Xs), torch.stack(ys), torch.stack(ms), list(metas)

class TinyCNN(nn.Module):
    def __init__(self, in_ch, mid=16, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),   nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, kernel_size=1)
        )
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    config_path = os.path.join(script_dir, "..", "inputs", "training.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    print("--- Configuration Loaded ---")
    print(json.dumps(cfg, indent=2))
    print("-" * 28)
    print("\n--- Initializing Datasets ---")
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
    print(f"✅ Training set has {len(train_ds)} samples.")
    print(f"✅ Validation set has {len(val_ds)} samples.")
    print("-" * 29)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=cfg["training"]["shuffle"], num_workers=0,
        collate_fn=collate_keep_meta, pin_memory=True # Can set pin_memory=True now
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=0,
        collate_fn=collate_keep_meta, pin_memory=True # Can set pin_memory=True now
    )
    print("\n--- Initializing Model & Device ---")
    DEVICE_ID = 1
    if torch.cuda.is_available():
        if DEVICE_ID >= torch.cuda.device_count():
            print(f"⚠️ WARNING: Device ID {DEVICE_ID} is not available. Found {torch.cuda.device_count()} devices.")
            DEVICE_ID = 0
        device = torch.device(f"cuda:{DEVICE_ID}")
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(device)
        print(f"✅ Using GPU: {gpu_name} (Device {DEVICE_ID})")
    else:
        device = torch.device("cpu")
        print("⚠️ WARNING: CUDA not available. Using CPU.")
    X0, _, _, _ = next(iter(train_loader))
    in_channels = X0.shape[1]
    model = TinyCNN(in_ch=in_channels)
    model.to(device)
    print(f"✅ Model initialized with {in_channels} input channels.")
    print("-" * 33)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler() if device.type == "cuda" else None
    EPOCHS = 2
    ckpt_path = os.path.join(script_dir, "tinycnn_ckpt.pt")
    loss_history = {"train": [], "val": []}
    def epoch_loop(loader, train=True, epoch_num=0, total_epochs=0):
        model.train(train)
        total_loss, n_batches = 0.0, 0
        mode = "Train" if train else "Val"
        loader_len = len(loader)
        for batch_idx, (X, y, mask, metas) in enumerate(loader):
            non_blocking = (device.type == "cuda")
            X = X.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
            mask = mask.to(device, non_blocking=non_blocking)
            with torch.set_grad_enabled(train):
                with autocast(device_type=device.type, enabled=(scaler is not None)):
                    pred = model(X)
                    loss = masked_mse_loss(pred, y, mask)
                if train:
                    opt.zero_grad(set_to_none=True)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
            total_loss += float(loss.detach())
            n_batches += 1
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == loader_len:
                progress = (batch_idx + 1) / loader_len
                print(f"\rEpoch {epoch_num}/{total_epochs} [{mode}] | Batch {batch_idx+1}/{loader_len} ({progress:.0%})", end="")
        print()
        return total_loss / max(n_batches, 1)
    print("\n--- Starting Training ---")
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss = epoch_loop(train_loader, train=True, epoch_num=epoch, total_epochs=EPOCHS)
        loss_history["train"].append(train_loss)
        val_loss = epoch_loop(val_loader, train=False, epoch_num=epoch, total_epochs=EPOCHS)
        loss_history["val"].append(val_loss)
        print(f"Epoch {epoch}/{EPOCHS} Summary | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print("-" * 25)
    total_time = time.time() - start_time
    print(f"\n--- ✅ Training Complete in {total_time:.2f}s ---")
    torch.save({"model": model.state_dict(), "in_channels": in_channels}, ckpt_path)
    print(f"💾 Saved final model checkpoint to: {ckpt_path}")
    print("\n--- Loss Progression ---")
    print("Epoch | Train Loss | Val Loss")
    print("----------------------------")
    for i in range(EPOCHS):
        print(f"{i+1:^5} | {loss_history['train'][i]:<10.6f} | {loss_history['val'][i]:<10.6f}")
    print("----------------------------")
