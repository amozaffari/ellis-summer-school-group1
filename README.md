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

## Project Accomplishments

### Data Analysis (`climatology.ipynb`)
- Computing LAI climatology (1982-2017 average) for 24 half-monthly periods
- Calculating anomalies and z-scores relative to climatology
- Hemispheric time series analysis with area-weighted means

## How to access the resources and data / compute and etc
https://pad.gwdg.de/s/2KKJPuo1W