# train_sequence.py
import os, bisect, json
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler  # device-agnostic AMP

# ---------------- Dataset ----------------
def _open_da(path, varname, engine="netcdf4", robust_nan=True):
    ds = xr.open_dataset(path, engine=engine)
    da = ds[varname]
    if robust_nan:
        fv = da.attrs.get("_FillValue", None)
        if fv is not None:
            da = da.where(da != fv)
    lat_name = "lat" if "lat" in da.coords else "latitude"
    da = da.sortby(lat_name)  # north-up
    if "time" in da.dims and da.sizes["time"] == 24:
        da = da.assign_coords(sample=("time", np.arange(24)))
    return da

def _infer_var(nc_path, engine="netcdf4"):
    with xr.open_dataset(nc_path, engine=engine) as ds:
        for v in ds.data_vars:
            if ds[v].ndim >= 2:
                return v
    raise RuntimeError(f"No data variable in {nc_path}")

class ERA5LAISequenceWorld(Dataset):
    """
    Multi-year, whole-world, sliding window dataset.

    X layout:
      - feature_layout="time_channels": (T*C, H, W)
      - feature_layout="time_first":    (T, C, H, W)
    y:
      - target_mode="last": (1, H, W)
      - target_mode="all":  (T, 1, H, W)
    mask has same shape as y (True where target valid).
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

        X_seq = np.stack(X_list, axis=0)  # (T, 3, H, W)

        if self.target_mode == "last":
            yi, lt = self._slice_from_global_t(start + Ts - 1)
            y_map = self.lai_y[yi].isel(time=lt).values
            y = np.expand_dims(y_map, axis=0)            # (1,H,W)
            mask = ~np.isnan(y)
        else:
            y = np.stack(y_list, axis=0)                 # (T,1,H,W)
            mask = ~np.isnan(y)

        X_seq = np.nan_to_num(X_seq, nan=0.0)

        if self.feature_layout == "time_channels":
            # (T,3,H,W) -> (3T,H,W)
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
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return diff2.sum() / denom

def collate_keep_meta(batch):
    Xs, ys, ms, metas = zip(*batch)
    return torch.stack(Xs), torch.stack(ys), torch.stack(ms), list(metas)

# ---------------- Tiny model ----------------
class TinyCNN(nn.Module):
    """Simple CNN for feature_layout='time_channels' (in_ch = seq_len * 3)."""
    def __init__(self, in_ch, mid=16, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1),   nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, 1)
        )
    def forward(self, x):
        return self.net(x)

# ---------------- Train script ----------------
if __name__ == "__main__":
    # Resolve config relative to this script (Jupyter-safe fallback)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    config_path = os.path.join(script_dir, "..", "inputs", "training.json")

    with open(config_path, "r") as f:
        cfg = json.load(f)

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

    # Model & device
    X0, _, _, _ = next(iter(train_loader))
    in_channels = X0.shape[1]
    model = TinyCNN(in_ch=in_channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # AMP: use GradScaler only on CUDA
    scaler = GradScaler(device.type) if device.type == "cuda" else None

    EPOCHS = 2
    ckpt_path = os.path.join(script_dir, "tinycnn_ckpt.pt")

    def epoch_loop(loader, train=True):
        model.train(train)
        total_loss, n = 0.0, 0
        for X, y, mask, metas in loader:
            non_blocking = (device.type == "cuda")
            X = X.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
            mask = mask.to(device, non_blocking=non_blocking)

            with torch.set_grad_enabled(train):
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
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
            n += 1
        return total_loss / max(n, 1)

    for epoch in range(1, EPOCHS + 1):
        tr = epoch_loop(train_loader, True)
        va = epoch_loop(val_loader, False)
        print(f"epoch {epoch}/{EPOCHS}  train_loss={tr:.4f}  val_loss={va:.4f}")

    torch.save({"model": model.state_dict(), "in_channels": in_channels}, ckpt_path)
    print(f"Saved checkpoint to: {ckpt_path}")