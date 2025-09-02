import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xarray as xr
import glob
import os
from sklearn.model_selection import train_test_split


# -----------------------------
# 0. Define years and train/test split
# -----------------------------
years = list(range(1985, 2017))
train_years, test_years = train_test_split(years, test_size=0.2, random_state=42)

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
# 2. Load X variables filtered by years
# -----------------------------
def load_x_variables(x_paths, x_patterns, years_filter):
    x_data_list = []
    for var, path in x_paths.items():
        for year in years_filter:
            pattern = x_patterns[var].replace("*", str(year))
            files = sorted(glob.glob(os.path.join(path, pattern)))
            for f in files:
                ds = xr.open_dataset(f)
                var_data = ds[var].values.flatten()
                var_data = var_data[~np.isnan(var_data)]
                x_data_list.append(var_data)
                ds.close()
    # Stack all variables horizontally
    X = np.stack(x_data_list, axis=1)
    return X

# -----------------------------
# 3. Load Y variable filtered by years
# -----------------------------
def load_y_variable(y_path, years_filter):
    y_data_list = []
    for year in years_filter:
        pattern = f"LAI.1440.720.{year}.nc"
        files = sorted(glob.glob(os.path.join(y_path, pattern)))
        for f in files:
            ds = xr.open_dataset(f)
            y_var = ds["LAI"].values.flatten()
            y_var = y_var[~np.isnan(y_var)]
            y_data_list.append(y_var)
            ds.close()
    Y = np.concatenate(y_data_list, axis=0).reshape(-1,1)
    return Y

# -----------------------------
# 4. Prepare training and testing data
# -----------------------------
X_train = load_x_variables(x_paths, x_patterns, train_years)
Y_train = load_y_variable(y_path, train_years)

X_test = load_x_variables(x_paths, x_patterns, test_years)
Y_test = load_y_variable(y_path, test_years)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# -----------------------------
# 5. Define linear regression model
# -----------------------------
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(input_dim=X_train_tensor.shape[1])

# -----------------------------
# 6. Loss and optimizer
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 7. Training loop
# -----------------------------
num_epochs = 500
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# -----------------------------
# 8. Evaluate on test set
# -----------------------------
with torch.no_grad():
    Y_pred_test = model(X_test_tensor)
    test_loss = criterion(Y_pred_test, Y_test_tensor)
    print(f"Test MSE Loss: {test_loss.item():.4f}")

print("First 5 test predictions:\n", Y_pred_test[:5])