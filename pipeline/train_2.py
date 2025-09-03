# train_sequence.py
import os, bisect, json, time
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

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
    """
    def __init__(
        self, years, seq_len=48, seq_stride=24,
        era5_mode="anom", lai_mode="raw",
        feature_layout="time_channels", target_mode="last",
        paths=None, engine="netcdf4", robust_nan=True,
    ):
        super().__init__()
        assert era5_mode in ("raw","anom","z")
        assert lai_mode in ("raw","anom")
        assert feature_layout in ("time_channels","time_first")
        assert target_mode in ("last","all")
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
        self._open_years()
        self.samples_per_year = 24
        self.year_offsets = [i * self.samples_per_year for i in range(len(self.years))]
        self.total_samples = self.samples_per_year * len(self.years)
        self.starts = list(range(0, self.total_samples - self.seq_len + 1, self.seq_stride))
        if not self.starts:
            raise ValueError("seq_len is longer than total samples. Reduce seq_len or add years.")

    def _open_years(self):
        self.ssrd_y, self.t2m_y, self.tp_y, self.lai_y = [], [], [], []
        for y in self.years:
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
                if os.path.exists(lai_anom_file):
                    lai = _open_da(lai_anom_file, "LAI_anom", self.engine, self.robust_nan)
                else:
                    raise FileNotFoundError(f"LAI anomaly file not found: {lai_anom_file}")
            for da, name in [(ssrd,"ssrd"),(t2m,"t2m"),(tp,"tp")]:
                assert "time" in da.dims and da.sizes["time"] == 24, f"{name} {y} expects 24 samples"
            self.ssrd_y.append(ssrd); self.t2m_y.append(t2m); self.tp_y.append(tp); self.lai_y.append(lai)
        self.lat_name = "lat" if "lat" in self.lai_y[0].coords else "latitude"
        self.lon_name = "lon" if "lon" in self.lai_y[0].coords else "longitude"
        H = int(self.lai_y[0].sizes[self.lat_name]); W = int(self.lai_y[0].sizes[self.lon_name])
        for j in range(len(self.years)):
            for da, name in [(self.ssrd_y[j],"ssrd"),(self.t2m_y[j],"t2m"),(self.tp_y[j],"tp")]:
                assert da.sizes[self.lat_name] == H and da.sizes[self.lon_name] == W, f"{name} grid mismatch in year {self.years[j]}"

    def __len__(self):
        return len(self.starts)

    def _slice_from_global_t(self, global_t):
        j = bisect.bisect_right(self.year_offsets, global_t) - 1
        j = max(0, min(j, len(self.year_offsets)-1))
        return j, global_t - self.year_offsets[j]

    def __getitem__(self, idx):
        start, Ts = self.starts[idx], self.seq_len
        X_list, y_list = [], []
        for t in range(Ts):
            g = start + t
            yi, lt = self._slice_from_global_t(g)
            x_ssrd = self.ssrd_y[yi].isel(time=lt).values
            x_t2m  = self.t2m_y [yi].isel(time=lt).values
            x_tp   = self.tp_y  [yi].isel(time=lt).values
            X_list.append(np.stack([x_ssrd, x_t2m, x_tp], axis=0))
            if self.target_mode == "all":
                y_map = self.lai_y[yi].isel(time=lt).values
                y_list.append(np.expand_dims(y_map, axis=0))
        X_seq = np.stack(X_list, axis=0)
        if self.target_mode == "last":
            yi, lt = self._slice_from_global_t(start + Ts - 1)
            y_map = self.lai_y[yi].isel(time=lt).values
            y = np.expand_dims(y_map, axis=0)
            mask = ~np.isnan(y)
        else:
            y = np.stack(y_list, axis=0)
            mask = ~np.isnan(y)
        X_seq = np.nan_to_num(X_seq, nan=0.0)
        if self.feature_layout == "time_channels":
            X = X_seq.transpose(1,0,2,3).reshape(-1, X_seq.shape[2], X_seq.shape[3])
        else:
            X = X_seq
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(np.nan_to_num(y, nan=0.0).astype(np.float32))
        m_t = torch.from_numpy(mask.astype(np.bool_))
        t_indices = list(range(start, start+Ts))
        years_cov = sorted(set(self.years[self._slice_from_global_t(t)[0]] for t in t_indices))
        meta = {"start_index": int(start), "t_indices": t_indices, "years": years_cov}
        return X_t, y_t, m_t, meta

def masked_mse_loss(pred, target, mask):
    """Calculates Mean Squared Error only on valid pixels specified by the mask."""
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return diff2.sum() / denom

def collate_keep_meta(batch):
    """Custom collate function to handle the metadata dictionary from __getitem__."""
    Xs, ys, ms, metas = zip(*batch)
    return torch.stack(Xs), torch.stack(ys), torch.stack(ms), list(metas)

# ---------------- Tiny CNN Model ----------------
class TinyCNN(nn.Module):
    """A very simple 2D CNN."""
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
    # --- ⚙️ Load Configuration ---
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

    # --- 💾 Initialize Datasets ---
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

    # --- 🧠 Initialize DataLoaders ---
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

    # --- 🚀 Initialize Model & Device ---
    print("\n--- Initializing Model & Device ---")
    # Set your desired GPU ID here. 0 is the first GPU, 1 is the second, etc.
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

    # --- Optimizer and AMP Scaler ---
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler() if device.type == "cuda" else None

    # --- 🔥 Training Loop ---
    EPOCHS = 2
    ckpt_path = os.path.join(script_dir, "tinycnn_ckpt.pt")
    loss_history = {"train": [], "val": []}
    
    def epoch_loop(loader, train=True, epoch_num=0, total_epochs=0):
        """A full pass over the dataset (an epoch) with progress reporting."""
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
            
            # Print progress every 20 batches or on the last batch
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == loader_len:
                progress = (batch_idx + 1) / loader_len
                print(f"\rEpoch {epoch_num}/{total_epochs} [{mode}] | Batch {batch_idx+1}/{loader_len} ({progress:.0%})", end="")

        print() # Newline after the progress bar is complete
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