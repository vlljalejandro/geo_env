from netCDF4 import Dataset
import xarray as xr
import numpy as np
import os
from collections import namedtuple
from scipy.stats import norm
import matplotlib.pyplot as plt

# ## ================================ ##
# ## Wet Bulb Temperature Calculation ##
# ## ================================ ##

def calculate_wet_bulb_temperature(temp_k, rh_percent):
    """
    Calculate wet bulb temperature from air temperature and relative humidity.
    
    Args:
        temp_k: Temperature in Kelvin
        rh_percent: Relative humidity in percent
        
    Returns:
        Wet bulb temperature in Kelvin
    """
    # Convert temperature from Kelvin to Celsius for calculations
    temp_c = temp_k - 273.15
    
    # Calculation using Stull's method (2011) - accurate to within 0.3°C
    wbt_c = temp_c * np.arctan(0.151977 * (rh_percent + 8.313659)**0.5) + \
            np.arctan(temp_c + rh_percent) - np.arctan(rh_percent - 1.676331) + \
            0.00391838 * (rh_percent)**(3/2) * np.arctan(0.023101 * rh_percent) - 4.686035
    
    # Convert back to Kelvin
    wbt_k = wbt_c + 273.15
    
    return wbt_k

def main():
    # Input file paths
    temp_file = r"C:\Users\vllja\Documents\VS Code\geo_env\data\ISIMIP_Data\SSP370\Temp_370\Temp_370.nc"
    rh_file = r"C:\Users\vllja\Documents\VS Code\geo_env\data\ISIMIP_Data\SSP370\Humudity_370\RH370.nc"
    
    # Output directory and file
    output_dir = r"C:\Users\vllja\Documents\VS Code\geo_env\data\ISIMIP_Data\wet_bulb_370"
    output_file = os.path.join(output_dir, "wb_370.nc")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the datasets
    ds_temp = xr.open_dataset(temp_file)
    ds_rh = xr.open_dataset(rh_file)
    
    # Extract temperature and humidity data
    temp_k = ds_temp['tas']  # Assuming 'tas' is temperature variable
    rh_percent = ds_rh['hurs']  # Assuming 'hurs' is relative humidity
    
    # Calculate wet bulb temperature
    wbt_k = calculate_wet_bulb_temperature(temp_k, rh_percent)
    
    # Create a new dataset for the output
    ds_output = xr.Dataset(
        {
            'wet_bulb_temp': (['time', 'lat', 'lon'], wbt_k.values),
        },
        coords={
            'time': ds_temp['time'],
            'lat': ds_temp['lat'],
            'lon': ds_temp['lon'],
        },
        attrs={
            'description': 'Wet bulb temperature calculated from temperature and relative humidity',
            'units': 'K',
            'calculation_method': "Stull's method (2011)",
        }
    )
    
    # Save to NetCDF
    ds_output.to_netcdf(output_file)
    print(f"Wet bulb temperature saved to: {output_file}")
    
    # Close the datasets
    ds_temp.close()
    ds_rh.close()

if __name__ == "__main__":
    main()