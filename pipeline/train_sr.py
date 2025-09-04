# train_optimized.py  (PySR version)
import os, json, time
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from pysr import PySRRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ---------------- Dataset Helpers (unchanged) ----------------
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

# ---------------- Dataset Class (optimized version for PySR) ----------------
class ERA5LAISequenceWorld(Dataset):
    def __init__(
        self, years, seq_len=48, seq_stride=24,
        era5_mode="anom", lai_mode="raw",
        feature_layout="time_channels", target_mode="last",
        paths=None, engine="netcdf4", robust_nan=True,
    ):
        super().__init__()
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
        self._open_and_load_years()

        self.samples_per_year = 24
        self.total_samples = self.lai_data.shape[0]
        self.starts = list(range(0, self.total_samples - self.seq_len + 1, self.seq_stride))
        if not self.starts:
            raise ValueError("seq_len is longer than total samples. Reduce seq_len or add years.")

    def _open_and_load_years(self):
        ssrd_y, t2m_y, tp_y, lai_y = [], [], [], []
        print(f"    - Loading data for years: {self.years}...")
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
                lai = _open_da(lai_anom_file, "LAI_anom", self.engine, self.robust_nan)
            ssrd_y.append(ssrd); t2m_y.append(t2m); tp_y.append(tp); lai_y.append(lai)

        print("    - Concatenating yearly data...")
        full_ssrd = xr.concat(ssrd_y, dim="time")
        full_t2m  = xr.concat(t2m_y, dim="time")
        full_tp   = xr.concat(tp_y, dim="time")
        full_lai  = xr.concat(lai_y, dim="time")

        print("    - Loading all data into memory (this may take a moment)...")
        self.ssrd_data = full_ssrd.load().values  # (T,H,W)
        self.t2m_data  = full_t2m.load().values  # (T,H,W)
        self.tp_data   = full_tp.load().values   # (T,H,W)
        self.lai_data  = full_lai.load().values  # (T,H,W)
        print("    - ✅ Data loaded.")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.seq_len
        x_ssrd_seq = self.ssrd_data[start:end, :, :]
        x_t2m_seq  = self.t2m_data[start:end, :, :]
        x_tp_seq   = self.tp_data[start:end, :, :]
        X_seq = np.stack([x_ssrd_seq, x_t2m_seq, x_tp_seq], axis=1)  # (T,3,H,W)

        if self.target_mode == "last":
            y = self.lai_data[end - 1, :, :]
            y = np.expand_dims(y, axis=0)    # (1,H,W)
            mask = ~np.isnan(y)
        else:
            y = self.lai_data[start:end, :, :]  # (T,H,W)
            y = np.expand_dims(y, axis=1)       # (T,1,H,W)
            mask = ~np.isnan(y)

        X_seq = np.nan_to_num(X_seq, nan=0.0)

        if self.feature_layout == "time_channels":
            # -> (3T, H, W)
            X = X_seq.transpose(0, 2, 3, 1).reshape(X_seq.shape[2], X_seq.shape[3], -1).transpose(2, 0, 1)
        else:
            # -> (T, 3, H, W)
            X = X_seq

        X_t = torch.from_numpy(X.astype(np.float32))                               # (C,H,W) or (T,3,H,W)
        y_t = torch.from_numpy(np.nan_to_num(y, nan=0.0).astype(np.float32))       # (1,H,W) or (T,1,H,W)
        m_t = torch.from_numpy(mask.astype(np.bool_))
        meta = {"start_index": int(start)}
        return X_t, y_t, m_t, meta

def collate_keep_meta(batch):
    Xs, ys, ms, metas = zip(*batch)
    return torch.stack(Xs), torch.stack(ys), torch.stack(ms), list(metas)

# ---------------- PySR helpers ----------------
def _batch_to_flat_xy(X, y, mask):
    """
    Convert a batch (B, C, H, W) + (B, 1, H, W) to tabular X (n, C) and y (n,)
    using the valid-target mask.
    Works also if X is (B, C, H, W) or (B, T*3, H, W) — we just treat C as features.
    """
    # Ensure on CPU
    X = X.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy().astype(bool)

    B, C, H, W = X.shape
    X_flat = X.transpose(0, 2, 3, 1).reshape(-1, C)    # (B*H*W, C)
    y_flat = y.reshape(B, -1).reshape(-1)              # (B*H*W,)
    m_flat = mask.reshape(B, -1).reshape(-1)           # (B*H*W,)

    X_valid = X_flat[m_flat]
    y_valid = y_flat[m_flat]
    return X_valid, y_valid

def collect_samples(loader, max_points=10000, seed=0, log_prefix="Train"):
    """
    Walk the loader, flatten each batch to (n, C), subsample to gather at most max_points.
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    total = 0
    loader_len = len(loader)
    for bidx, (X, y, mask, metas) in enumerate(loader, 1):
        Xb, yb = _batch_to_flat_xy(X, y, mask)
        if yb.size == 0:
            continue

        # If adding all would exceed max_points, sample the remainder.
        remaining = max_points - total
        if remaining <= 0:
            break
        if yb.size > remaining:
            sel = rng.choice(yb.size, size=remaining, replace=False)
            Xb = Xb[sel]
            yb = yb[sel]

        X_list.append(Xb)
        y_list.append(yb)
        total += yb.size

        progress = min(total / max_points, 1.0)
        print(f"\r[{log_prefix}] Collected {total}/{max_points} samples ({progress:.0%})  |  Batch {bidx}/{loader_len}", end="")

    print()  # newline
    if total == 0:
        raise RuntimeError(f"No valid samples collected from {log_prefix.lower()} loader.")
    return np.vstack(X_list), np.concatenate(y_list)

# ---------------- Main ----------------
if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    # Load config
    config_path = os.path.join(script_dir, "..", "inputs", "training_pysr.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    print("--- Configuration Loaded ---")
    print(json.dumps(cfg, indent=2))
    print("-" * 28)

    # Datasets & loaders
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
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=cfg["training"]["shuffle"],
        num_workers=0,
        collate_fn=collate_keep_meta,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_keep_meta,
        pin_memory=True,
    )

    # ---------------- PySR: collect samples ----------------
    max_train = int(cfg["training"].get("pysr_max_train_points", 10000))
    max_val   = int(cfg["training"].get("pysr_max_val_points",   5000))
    seed      = int(cfg["training"].get("seed", 0))

    print("\n--- Collecting Samples for PySR ---")
    X_train, y_train = collect_samples(train_loader, max_points=max_train, seed=seed, log_prefix="Train")
    X_val,   y_val   = collect_samples(val_loader,   max_points=max_val,   seed=seed+1, log_prefix="Val")
    in_channels = X_train.shape[1]
    print(f"✅ Collected {X_train.shape[0]} train samples with {in_channels} features each.")
    print(f"✅ Collected {X_val.shape[0]} val samples.")
    print("-" * 33)

    # ---------------- Initialize PySR ----------------
    variable_names = [f"x{i}" for i in range(in_channels)]  # optional, for readability

    model = PySRRegressor(
        # Search settings — tweak as needed
        niterations=int(cfg["training"].get("pysr_niterations", 100)),
        maxsize=int(cfg["training"].get("pysr_maxsize", 20)),
        binary_operators=cfg["training"].get("pysr_binary_ops", ["+", "-", "*", "/"]),
        unary_operators=cfg["training"].get("pysr_unary_ops", ["exp", "log", "sin", "cos"]),
        elementwise_loss=cfg["training"].get("pysr_loss", "loss(x, y) = (x - y)^2"),
        model_selection=cfg["training"].get("pysr_model_selection", "best"),
        variable_names=variable_names,
        progress=True,
        verbosity=1,
    )

    print("\n--- Starting PySR ---")
    start_time = time.time()
    model.fit(X_train, y_train)
    total_time = time.time() - start_time
    print(f"\n--- ✅ Training Complete in {total_time:.2f}s ---")

    # ---------------- Evaluate ----------------
    print("\n--- Evaluation (Validation Set) ---")
    y_val_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    r2   = r2_score(y_val, y_val_pred)
    print("Best equation found:")
    print(model.get_best())
    print(f"\nValidation RMSE: {rmse:.6f}")
    print(f"Validation R²  : {r2:.6f}")
    print("-" * 25)

    # ---------------- Save ----------------
    ckpt_dir = script_dir
    model_path = os.path.join(ckpt_dir, "pysr_model.pkl")
    eqn_path   = os.path.join(ckpt_dir, "pysr_equations.txt")

    joblib.dump(model, model_path)
    with open(eqn_path, "w") as f:
        f.write(str(model))       # full Pareto front
        f.write("\n\nBest:\n")
        f.write(str(model.get_best()))

    print(f"💾 Saved PySR model to: {model_path}")
    print(f"🧮 Saved equations to:  {eqn_path}")

    # ---------------- Summary table (like epoch table) ----------------
    print("\n--- Summary ---")
    print("Split | Samples | Features | RMSE     | R^2    ")
    print("------------------------------------------------")
    print(f"Train | {X_train.shape[0]:>7} | {in_channels:^8} |    -     |   -    ")
    print(f"Val   | {X_val.shape[0]:>7} | {in_channels:^8} | {rmse:7.4f} | {r2:6.3f}")
    print("------------------------------------------------")
