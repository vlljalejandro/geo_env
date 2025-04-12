from netCDF4 import Dataset
import xarray as xr
import numpy as np
import os
from collections import namedtuple
from scipy.stats import norm
import matplotlib.pyplot as plt


###########################################
## Part 1: Data Download and Preparation ##

# # Open the NetCDF file
# nc_file = Dataset("data\ISIMIP_Data\SSP126\Humidity_126\RH126.nc", "r")  # "r" means read mode
# # Print all variable names
# print(nc_file.variables.keys())
# # Close the file
# nc_file.close()

# Create the variables
prec_SSP126 = xr.open_dataset(r"C:\Users\vllja\Documents\VS Code\geo_env\data\ISIMIP_Data\SSP126\Precipitation_126\PR_126.nc")
temp_SSP126 = xr.open_dataset(r"C:\Users\vllja\Documents\VS Code\geo_env\data\ISIMIP_Data\SSP126\Temp_126\Temp126.nc")
wetb_SSP126 = xr.open_dataset(r"C:\Users\vllja\Documents\VS Code\geo_env\data\ISIMIP_Data\wet_bulb_126\wb_126.nc")
prec_SSP370 = xr.open_dataset(r"C:\Users\vllja\Documents\VS Code\geo_env\data\ISIMIP_Data\SSP370\Precipitation_370\pr370.nc")
temp_SSP370 = xr.open_dataset(r"C:\Users\vllja\Documents\VS Code\geo_env\data\ISIMIP_Data\SSP370\Temp_370\Temp_370.nc")
wetb_SSP370 = xr.open_dataset(r"C:\Users\vllja\Documents\VS Code\geo_env\data\ISIMIP_Data\wet_bulb_370\wb_370.nc")   



###########################################
## Part 2: Climate Change Trend Analysis ##

# Time array (for slope calculation)
time_years = np.arange(0, 86)

## ========================================================= ##
## TREND ANALYSIS FUNCTIONS (Hamed & Rao 1998 + Sen's Slope) ##
## ========================================================= ##

def hamed_rao_mk_test(x, alpha=0.05):
    """Modified MK test with autocorrelation correction (Hamed & Rao 1998)"""
    n = len(x)
    s = 0
    for k in range(n-1):
        for j in range(k+1, n):
            s += np.sign(x[j] - x[k])
    
    # Calculate variance with autocorrelation correction
    var_s = n*(n-1)*(2*n+5)/18
    ties = np.unique(x, return_counts=True)[1]
    for t in ties:
        var_s -= t*(t-1)*(2*t+5)/18
    
    # Correct for autocorrelation
    n_eff = n
    if n > 10:
        acf = [1] + [np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, n//4)]
        n_eff = n / (1 + 2 * sum((n-i)/n * acf[i] for i in range(1, len(acf))))
        var_s *= n_eff / n
    
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    p = 2 * (1 - norm.cdf(abs(z)))
    h = abs(z) > norm.ppf(1-alpha/2)
    
    Trend = namedtuple('Trend', ['trend', 'h', 'p', 'z', 's'])
    trend = 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no trend'
    return Trend(trend=trend, h=h, p=p, z=z, s=s)

def sens_slope(x, y):
    """Sen's slope estimator"""
    slopes = []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    return np.median(slopes)

# Calculate the annual averages for SSP126 and SSP370
avg_temp_SSP126 = temp_SSP126['tas'].resample(time='YE').mean(dim='time')
avg_temp_SSP126 = avg_temp_SSP126.mean(dim=['lat', 'lon'])
avg_temp_SSP126 = avg_temp_SSP126 - 273.15
avg_temp_SSP370 = temp_SSP370['tas'].resample(time='YE').mean(dim='time')
avg_temp_SSP370 = avg_temp_SSP370.mean(dim=['lat', 'lon'])
avg_temp_SSP370 = avg_temp_SSP370 - 273.15

avg_prec_SSP126 = prec_SSP126['pr'].resample(time='YE').mean(dim='time')
avg_prec_SSP126 = avg_prec_SSP126.mean(dim=['lat', 'lon'])
avg_prec_SSP370 = prec_SSP370['pr'].resample(time='YE').mean(dim='time')
avg_prec_SSP370 = avg_prec_SSP370.mean(dim=['lat', 'lon'])

# Get the trends
trend_temp_SSP126 = hamed_rao_mk_test(avg_temp_SSP126.values)
trend_temp_SSP370 = hamed_rao_mk_test(avg_temp_SSP370.values)
trend_prec_SSP126 = hamed_rao_mk_test(avg_prec_SSP126.values)
trend_prec_SSP370 = hamed_rao_mk_test(avg_prec_SSP370.values)
print(f"Trend for temperature SSP126: {trend_temp_SSP126.trend}, p-value: {trend_temp_SSP126.p}")
print(f"Trend for temperature SSP370: {trend_temp_SSP370.trend}, p-value: {trend_temp_SSP370.p}")
print(f"Trend for precipitation SSP126: {trend_prec_SSP126.trend}, p-value: {trend_prec_SSP126.p}")
print(f"Trend for precipitation SSP370: {trend_prec_SSP370.trend}, p-value: {trend_prec_SSP370.p}")

# Get the slopes
slope_temp_SSP126 = sens_slope(time_years, avg_temp_SSP126.values)
slope_temp_SSP370 = sens_slope(time_years, avg_temp_SSP370.values)
slope_prec_SSP126 = sens_slope(time_years, avg_prec_SSP126.values)
slope_prec_SSP370 = sens_slope(time_years, avg_prec_SSP370.values)
print(f"Slope for temperature SSP126: {slope_temp_SSP126} °C/year")
print(f"Slope for temperature SSP370: {slope_temp_SSP370} °C/year") 
print(f"Slope for precipitation SSP126: {slope_prec_SSP126} mm/year")
print(f"Slope for precipitation SSP370: {slope_prec_SSP370} mm/year")

# All plots for SSP126 and SSP370 with trend line
plt.figure(figsize=(12, 6))
plt.plot(avg_temp_SSP126['time'], avg_temp_SSP126, label='Annual Average Temperature (SSP126)', color='red', marker='o', markersize=5)
plt.plot(avg_temp_SSP126['time'], slope_temp_SSP126 * time_years + avg_temp_SSP126.values[0], color='blue', linestyle='--', label='Trend: 0.00443 °C/year (p = 0.0)')
plt.title('Annual Average Temperature for SSP126')
plt.xlabel('Year')
plt.ylabel('Temperature (K)')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(avg_temp_SSP370['time'], avg_temp_SSP370, label='Annual Average Temperature (SSP370)', color='red', marker='o', markersize=5)
plt.plot(avg_temp_SSP370['time'], slope_temp_SSP370 * time_years + avg_temp_SSP370.values[0], color='blue', linestyle='--', label='Trend: 0.04371 °C/year (p = 0.0)')
plt.title('Annual Average Temperature for SSP370')
plt.xlabel('Year')
plt.ylabel('Temperature (K)')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(avg_prec_SSP126['time'], avg_prec_SSP126, label='Annual Average Precipitation (SSP126)', color='blue', marker='o', markersize=5)
plt.plot(avg_prec_SSP126['time'], slope_prec_SSP126 * time_years + avg_prec_SSP126.values[0], color='red', linestyle='--', label='Trend: 0.0 mm/year (p = 0.52932)')
plt.title('Annual Average Precipitation for SSP126')
plt.xlabel('Year')
plt.ylabel('Precipitation (mm)')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(avg_prec_SSP370['time'], avg_prec_SSP370, label='Annual Average Precipitation (SSP370)', color='blue', marker='o', markersize=5)
plt.plot(avg_prec_SSP370['time'], slope_prec_SSP370 * time_years + avg_prec_SSP370.values[0], color='red', linestyle='--', label='Trend: 0.0 mm/year (p = 0.54625)')
plt.title('Annual Average Precipitation for SSP370')
plt.xlabel('Year')
plt.ylabel('Precipitation (mm)')
plt.grid()
plt.legend()
plt.show()



##########################################
## Part 3: Analysis of climate extremes ##

# Resample to daily means
daily_temp_SSP126 = temp_SSP126['tas'].resample(time='D').mean(dim='time')
daily_temp_SSP370 = temp_SSP370['tas'].resample(time='D').mean(dim='time')
daily_prec_SSP126 = prec_SSP126['pr'].resample(time='D').mean(dim='time')
daily_prec_SSP370 = prec_SSP370['pr'].resample(time='D').mean(dim='time')

# Get the maximum event per year for temperature and precipitation
max_temp_SSP126 = daily_temp_SSP126.resample(time='YE').max(dim='time')
max_temp_SSP126 = max_temp_SSP126.mean(dim=['lat', 'lon'])
max_temp_SSP126 = max_temp_SSP126 - 273.15
max_temp_SSP370 = daily_temp_SSP370.resample(time='YE').max(dim='time')
max_temp_SSP370 = max_temp_SSP370.mean(dim=['lat', 'lon'])
max_temp_SSP370 = max_temp_SSP370 - 273.15
max_prec_SSP126 = daily_prec_SSP126.resample(time='YE').max(dim='time')
max_prec_SSP126 = max_prec_SSP126.mean(dim=['lat', 'lon'])
max_prec_SSP370 = daily_prec_SSP370.resample(time='YE').max(dim='time')
max_prec_SSP370 = max_prec_SSP370.mean(dim=['lat', 'lon'])

# Get the trends for max temperature and precipitation
trend_max_temp_SSP126 = hamed_rao_mk_test(max_temp_SSP126.values)
trend_max_temp_SSP370 = hamed_rao_mk_test(max_temp_SSP370.values)
trend_max_prec_SSP126 = hamed_rao_mk_test(max_prec_SSP126.values)
trend_max_prec_SSP370 = hamed_rao_mk_test(max_prec_SSP370.values)
print(f"Trend for max temperature SSP126: {trend_max_temp_SSP126.trend}, p-value: {trend_max_temp_SSP126.p}")
print(f"Trend for max temperature SSP370: {trend_max_temp_SSP370.trend}, p-value: {trend_max_temp_SSP370.p}")
print(f"Trend for max precipitation SSP126: {trend_max_prec_SSP126.trend}, p-value: {trend_max_prec_SSP126.p}")
print(f"Trend for max precipitation SSP370: {trend_max_prec_SSP370.trend}, p-value: {trend_max_prec_SSP370.p}")

# Get the slopes for max temperature and precipitation
slope_max_temp_SSP126 = sens_slope(time_years, max_temp_SSP126.values)
slope_max_temp_SSP370 = sens_slope(time_years, max_temp_SSP370.values)
slope_max_prec_SSP126 = sens_slope(time_years, max_prec_SSP126.values)
slope_max_prec_SSP370 = sens_slope(time_years, max_prec_SSP370.values)
print(f"Slope for max temperature SSP126: {slope_max_temp_SSP126} °C/year")
print(f"Slope for max temperature SSP370: {slope_max_temp_SSP370} °C/year")
print(f"Slope for max precipitation SSP126: {slope_max_prec_SSP126} mm/year")
print(f"Slope for max precipitation SSP370: {slope_max_prec_SSP370} mm/year")

# Plot the maximum temperature and precipitation events
plt.figure(figsize=(12, 6))
plt.plot(max_temp_SSP126['time'], max_temp_SSP126, label='Max Temperature (SSP126)', color='red', marker='o', markersize=5)
plt.plot(max_temp_SSP126['time'], slope_max_temp_SSP126 * time + max_temp_SSP126.values[0], color='blue', linestyle='--', label='Trend: 0.00664 °C/year (p = 0.0)')
plt.title('Max Temperature for SSP126')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(max_temp_SSP370['time'], max_temp_SSP370, label='Max Temperature (SSP370)', color='red', marker='o', markersize=5)
plt.plot(max_temp_SSP370['time'], slope_max_temp_SSP370 * time_years + max_temp_SSP370.values[0], color='blue', linestyle='--', label='Trend: 0.05023 °C/year (p = 0.0)')
plt.title('Max Temperature for SSP370')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(max_prec_SSP126['time'], max_prec_SSP126, label='Max Precipitation (SSP126)', color='blue', marker='o', markersize=5)
plt.plot(max_prec_SSP126['time'], slope_max_prec_SSP126 * time_years + max_prec_SSP126.values[0], color='red', linestyle='--', label='Trend: 0.0 mm/year (p = 0.61624)')
plt.title('Max Precipitation for SSP126')
plt.xlabel('Year')
plt.ylabel('Precipitation (mm)')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(max_prec_SSP370['time'], max_prec_SSP370, label='Max Precipitation (SSP370)', color='blue', marker='o', markersize=5)
plt.plot(max_prec_SSP370['time'], slope_max_prec_SSP370 * time_years + max_prec_SSP370.values[0], color='red', linestyle='--', label='Trend: 0.0 mm/year (p = 0.05023)')
plt.title('Max Precipitation for SSP370')
plt.xlabel('Year')
plt.ylabel('Precipitation (mm)')
plt.grid()
plt.legend()
plt.show()



##############################################
## Part 4: Wet Bulb Temperature Calculation ##

# Clip the wet bulb temperature data to the same time period as the other datasets
wetb_SSP126 = wetb_SSP126['wet_bulb_temp'].resample(time='YE').mean(dim='time')
wetb_SSP370 = wetb_SSP370['wet_bulb_temp'].resample(time='YE').mean(dim='time')
wetb_SSP126 = wetb_SSP126.sel(time=slice('2015-01-01', '2100-12-31'))
wetb_SSP370 = wetb_SSP370.sel(time=slice('2015-01-01', '2100-12-31'))
wetb_SSP126 = wetb_SSP126.mean(dim=['lat', 'lon'])
wetb_SSP370 = wetb_SSP370.mean(dim=['lat', 'lon'])
wetb_SSP126 = wetb_SSP126 - 273.15
wetb_SSP370 = wetb_SSP370 - 273.15



############################################################
## Part 5: Wet Bulb Temperature Trend Analysis & Extremes ##

# Get the trends for wet bulb temperature
trend_wetb_SSP126 = hamed_rao_mk_test(wetb_SSP126.values)
trend_wetb_SSP370 = hamed_rao_mk_test(wetb_SSP370.values)
print(f"Trend for wet bulb temperature SSP126: {trend_wetb_SSP126.trend}, p-value: {trend_wetb_SSP126.p}")
print(f"Trend for wet bulb temperature SSP370: {trend_wetb_SSP370.trend}, p-value: {trend_wetb_SSP370.p}")

# Get the slopes for wet bulb temperature
slope_wetb_SSP126 = sens_slope(time_years, wetb_SSP126.values)
slope_wetb_SSP370 = sens_slope(time_years, wetb_SSP370.values)
print(f"Slope for wet bulb temperature SSP126: {slope_wetb_SSP126} °C/year")
print(f"Slope for wet bulb temperature SSP370: {slope_wetb_SSP370} °C/year")

# Plot the wet bulb temperature
plt.figure(figsize=(12, 6))
plt.plot(wetb_SSP126['time'], wetb_SSP126, label='Wet Bulb Temperature (SSP126)', color='green', marker='o', markersize=5)
plt.plot(wetb_SSP126['time'], slope_wetb_SSP126 * time_years + wetb_SSP126.values[0], color='blue', linestyle='--', label='Trend: 0.00137 °C/year (p = 0.016)')
plt.title('Wet Bulb Temperature for SSP126')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(wetb_SSP370['time'], wetb_SSP370, label='Wet Bulb Temperature (SSP370)', color='green', marker='o', markersize=5)
plt.plot(wetb_SSP370['time'], slope_wetb_SSP370 * time_years + wetb_SSP370.values[0], color='blue', linestyle='--', label='Trend: 0.03299 °C/year (p = 0.000)')
plt.title('Wet Bulb Temperature for SSP370')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.grid()
plt.legend()
plt.show()