# ellis-summer-school-group1

## Environment Setup
```bash
conda create -n my_env python=3.10 pandas xarray netCDF4 scipy matplotlib
```

## Data Location
The LAI (Leaf Area Index) data is located in: `/ptmp/mp002/ellis/lai/lai/`
- Files: `LAI.1440.720.{year}.nc` for years 1982-2017
- Each file contains 24 half-monthly composites (15-day periods)
- Resolution: 1440x720 global grid

## Produced Data

### Generated Outputs
All generated outputs are stored in `/ptmp/mp040/outputdir/` and are accessible to everyone:
- **`lai/`**:
        - **`clim/`** - Contains climatology data (1982-2017 average LAI values)
        - **`anom/`** - Contains anomaly and z-score data for LAI relative to the climatology

- **`era5/`**:
        - **`clim/`** - Contains climatology data (1982-2017 average) for ERA5 climate variables
        - **`anom/`** - Contains anomaly and z-score data for ERA5 variables relative to the climatology
        
## Project Accomplishments

### Data Analysis (`lai_prep.ipynb`)
- Computing LAI climatology (1982-2017 average) for 24 half-monthly periods
- Calculating anomalies and z-scores relative to climatology
- Hemispheric time series analysis with area-weighted means

### ERA5 Climate Data Analysis (`era5_prep.ipynb`)
- Processing ERA5 climate variables: SSRD (surface solar radiation), T2M (2m temperature), TP (total precipitation)
- Computing ERA5 climatology (1982-2017 average) for 24 half-monthly periods
- Calculating anomalies and z-scores for climate variables relative to climatology
- Generating visualizations for climate data analysis and anomaly detection

## How to access the resources and data / compute and etc
https://pad.gwdg.de/s/2KKJPuo1W