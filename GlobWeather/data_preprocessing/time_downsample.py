import os
import argparse
import xarray as xr
import numpy as np
from tqdm import tqdm
from GlobWeather.utils.data_utils import NAME_TO_VAR, VAR_TO_NAME

# change as needed
VARS = [
    # constants
    # "angle_of_sub_gridscale_orography.nc",
    # "geopotential_at_surface.nc",
    # "high_vegetation_cover.nc",
    # "lake_cover.nc",
    # "lake_depth.nc",
    # "land_sea_mask.nc",
    # "low_vegetation_cover.nc",
    # "slope_of_sub_gridscale_orography.nc",
    # "soil_type.nc",
    # "standard_deviation_of_filtered_subgrid_orography.nc",
    # "standard_deviation_of_orography.nc",
    # "type_of_high_vegetation.nc",
    # "type_of_low_vegetation.nc",

    # surface variables
    # "2m_temperature",
    # "10m_u_component_of_wind",
    # "10m_v_component_of_wind",
    # "total_cloud_cover",
    # "toa_incident_solar_radiation",
    # "10m_wind_speed",
    # "mean_sea_level_pressure",

    # pressure level variables
    # "geopotential",
    # "specific_humidity",
    # "temperature",
    # "u_component_of_wind",
    "v_component_of_wind",
    # "vertical_velocity",
]

def parse_args():
    parser = argparse.ArgumentParser(description='Regridding NetCDF files.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing input data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save regridded files.')
    parser.add_argument('--start_year', type=int, default=1979, help='Start year for the data range.')
    parser.add_argument('--end_year', type=int, default=2018, help='End year for the data range.')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    root_dir = args.root_dir
    save_dir = args.save_dir
    start_year = args.start_year
    end_year = args.end_year
    
    years = list(range(start_year, end_year + 1))
    os.makedirs(save_dir, exist_ok=True)
    
    var_dirs = [os.path.join(root_dir, v) for v in VARS]
    
    for dir in tqdm(var_dirs, desc='vars', position=0):
        var_name = os.path.basename(dir)
        
        os.makedirs(os.path.join(save_dir, var_name), exist_ok=True)
        for year in tqdm(years, desc='years', position=1, leave=False):
            ds_in = xr.open_dataset(os.path.join(dir, f'{year}.nc'))
            ds_out = ds_in.resample(time='6h').first()
            ds_out.attrs = {}
            ds_out = ds_out.rename({'lat': 'latitude', 'lon': 'longitude', NAME_TO_VAR[var_name]: var_name})
            ds_out.to_netcdf(os.path.join(save_dir, var_name, f'{year}.nc'))


if __name__ == "__main__":
    main()