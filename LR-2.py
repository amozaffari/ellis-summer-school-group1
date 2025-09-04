import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xarray as xr
import glob
import os
from sklearn.model_selection import train_test_split
import pandas as pd

# -----------------------------
# 0. Define years and train/test split
# -----------------------------
years = list(range(1995, 2007))
train_years, test_years = train_test_split(years, test_size=0.2, random_state=42)
# "train_years": [1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
#     "val_years":   [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
# train_years = [1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994,
#                1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004,
#                2005, 2006, 2007, 2008, 2009]

# test_years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
print("Training years:", train_years)
print("Testing years:", test_years)

# -----------------------------
# 1. Define paths and file patterns
# -----------------------------
y_path = "/ptmp/mp002/ellis/lai/lai"

x_paths = {
    "ssrd": "/ptmp/mp002/ellis/lai/ssrd",
    "swvl1": "/ptmp/mp002/ellis/lai/swvl1",
    "swvl2": "/ptmp/mp002/ellis/lai/swvl2",
    "swvl3": "/ptmp/mp002/ellis/lai/swvl3",
    "swvl4": "/ptmp/mp002/ellis/lai/swvl4",
    "t2m": "/ptmp/mp002/ellis/lai/t2m",
    "tp": "/ptmp/mp002/ellis/lai/tp"
}
x_patterns = {
    "ssrd": "ssrd.15daily.fc.era5.1440.720.*.nc",
    "swvl1": "swvl1.15daily.an.era5.1440.720.*.nc",
    "swvl2": "swvl2.15daily.an.era5.1440.720.*.nc",
    "swvl3": "swvl3.15daily.an.era5.1440.720.*.nc",
    "swvl4": "swvl4.15daily.an.era5.1440.720.*.nc",
    "t2m": "t2m.15daily.an.era5.1440.720.*.nc",
    "tp": "tp.15daily.fc.era5.1440.720.*.nc"
}

# -----------------------------
# 2. Load aligned X and Y with masks/shapes
# -----------------------------
def load_aligned_data(x_paths, x_patterns, y_path, years_filter):
    X_list, Y_list, masks, shapes, years_out = [], [], [], [], []

    for year in years_filter:
        y_file = glob.glob(os.path.join(y_path, f"LAI.1440.720.{year}.nc"))
        if not y_file:
            continue
        y_ds = xr.open_dataset(y_file[0])
        y_var = y_ds["LAI"]
        y_data = y_var.values  # shape (time, lat, lon)
        lat = y_ds["latitude"].values
        lon = y_ds["longitude"].values
        time = y_ds["time"].values
        y_ds.close()

        # Load all X variables
        x_vars = []
        for var, path in x_paths.items():
            x_file = glob.glob(os.path.join(path, x_patterns[var].replace("*", str(year))))
            if not x_file:
                continue
            x_ds = xr.open_dataset(x_file[0])
            x_var_data = x_ds[var].values
            x_ds.close()
            x_vars.append(x_var_data)

        # Stack: (time, lat, lon, n_vars)
        X_stack = np.stack(x_vars, axis=-1)

        # Mask invalid values
        mask = ~np.isnan(y_data)
        for i in range(X_stack.shape[-1]):
            mask &= ~np.isnan(X_stack[..., i])

        # Save mask & shape
        masks.append(mask)
        shapes.append(y_data.shape)
        years_out.append((year, lat, lon, time))

        # Flatten
        X_flat = X_stack[mask]
        Y_flat = y_data[mask].reshape(-1, 1)

        X_list.append(X_flat)
        Y_list.append(Y_flat)

    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)
    return X_all, Y_all, masks, shapes, years_out

# Prepare training and testing data
X_train, Y_train, _, _, _ = load_aligned_data(x_paths, x_patterns, y_path, train_years)
X_test, Y_test, test_masks, test_shapes, test_years_meta = load_aligned_data(x_paths, x_patterns, y_path, test_years)

print("Aligned X_train shape:", X_train.shape)
print("Aligned Y_train shape:", Y_train.shape)
print("Aligned X_test shape:", X_test.shape)
print("Aligned Y_test shape:", Y_test.shape)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# -----------------------------
# 3. Define linear regression model
# -----------------------------
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(input_dim=X_train_tensor.shape[1])

# -----------------------------
# 4. Loss and optimizer
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 5. Training loop
# -----------------------------
num_epochs = 500
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# -----------------------------
# 6. Evaluate and reconstruct predictions
# -----------------------------
with torch.no_grad():
    Y_pred_test = model(X_test_tensor)
    test_loss = criterion(Y_pred_test, Y_test_tensor)
    print(f"Test MSE Loss: {test_loss.item():.4f}")

Y_pred_np = Y_pred_test.numpy()

# -----------------------------
# 7. Save predictions back into NetCDF (per year)
# -----------------------------
output_dir = "./predictions_B"
os.makedirs(output_dir, exist_ok=True)

start = 0
for (year, lat, lon, time), mask, shape in zip(test_years_meta, test_masks, test_shapes):
    count = mask.sum()
    preds_year = Y_pred_np[start:start+count]
    start += count

    # Reconstruct into original grid
    Y_reconstructed = np.full(shape, np.nan, dtype=np.float32)
    Y_reconstructed[mask] = preds_year.ravel()

    # Save to NetCDF
    ds_out = xr.Dataset(
        {"LAI_pred": (("time", "latitude", "longitude"), Y_reconstructed)},
        coords={"time": time, "latitude": lat, "longitude": lon}
    )
    out_file = os.path.join(output_dir, f"LAI_pred_{year}.nc")
    ds_out.to_netcdf(out_file)
    print(f"Saved prediction for {year} -> {out_file}")
    print("Prediction shape:", Y_reconstructed.shape)
# Get learned weights (flatten since shape is (1, n_features))
weights = model.linear.weight.detach().numpy().flatten()

# Match with predictor names
predictor_names = list(x_paths.keys())
importance_df = pd.DataFrame({
    "predictor": predictor_names,
    "weight": weights,
    "abs_weight": np.abs(weights)
})

# Sort by absolute weight (importance)
importance_df = importance_df.sort_values(by="abs_weight", ascending=False)

print("\n=== Predictor Importance (by learned weight) ===")
print(importance_df)

# Optional: plot
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.bar(importance_df["predictor"], importance_df["abs_weight"])
plt.ylabel("Absolute Weight (Importance)")
plt.title("Predictor Importance")
plt.show()