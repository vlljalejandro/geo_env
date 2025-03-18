from netCDF4 import Dataset
import xarray as xr
import cdsapi
from multiprocessing import Pool
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
import rioxarray



# ###########################
# ## PART 1: DATA DOWNLOAD ##

# ## Define the function to download data for a specific year
# def download_year_data(year):
#     print(year)
#     dataset = "reanalysis-era5-single-levels"
#     request = {
#         "product_type": ["reanalysis"],
#         "year": [str(year)],
#         "month": [
#             "01", "02", "03",
#             "04", "05", "06",
#             "07", "08", "09",
#             "10", "11", "12"],
#         "day": [
#             "01", "02", "03",
#             "04", "05", "06",
#             "07", "08", "09",
#             "10", "11", "12",
#             "13", "14", "15",
#             "16", "17", "18",
#             "19", "20", "21",
#             "22", "23", "24",
#             "25", "26", "27",
#             "28", "29", "30",
#             "31"],
#         "time": [
#             "00:00", "01:00", "02:00",
#             "03:00", "04:00", "05:00",
#             "06:00", "07:00", "08:00",
#             "09:00", "10:00", "11:00",
#             "12:00", "13:00", "14:00",
#             "15:00", "16:00", "17:00",
#             "18:00", "19:00", "20:00",
#             "21:00", "22:00", "23:00"],
#         "data_format": "netcdf",
#         "download_format": "unarchived",
#         "variable": ["total_precipitation"],  #change your variable
#         "area": [33, 34, 16, 56]  #North West South East
#     }
#     # Specify the path where the files should be downloaded
#     download_path = r'C:\Users\vllja\Documents\VS Code\geo_env\data\Precipitation'
#     filename = f"{download_path}era5_OLR_{year}_total_precipitation.nc" #change the name
#     client = cdsapi.Client()
#     client.retrieve(dataset, request).download(filename)
#     print(f"Downloaded data for year {year} to {filename}.")

# ## Main block to parallelize the process
# if __name__ == "__main__":
#     # Define the years range
#     years = range(2010,2020)
#     # Number of processors to use
#     num_processors = 2
#     # Use Pool for parallel processing
#     with Pool(num_processors) as pool:
#         pool.map(download_year_data, years)


################################################
## PART 2: DATA EXPLORATION AND VISUALIZATION ##

## Open the NetCDF file
nc_file = Dataset(r"C:\Users\vllja\Documents\VS Code\geo_env\data\lab07\Precipitation\era5_OLR_2000_total_precipitation.nc", "r")  # "r" means read mode

## Print all variable names
print(nc_file.variables.keys())

## Close the file
nc_file.close()

## Shapefile ##
shapefile = gpd.read_file(r"C:\Users\vllja\Documents\VS Code\geo_env\data\lab07\saudi_area\Saudi_Shape.shp")  
shapefile = shapefile.to_crs("EPSG:4326")

##==================================================================================##
## Total precipitation against time for the years 2000 to 2020 based on monthly sum ##
monthly_sum = []

for i in range(2000,2021):
    precipitation = xr.open_dataset(rf"C:\Users\vllja\Documents\VS Code\geo_env\data\lab07\Precipitation\era5_OLR_{i}_total_precipitation.nc")
    precipitation.rio.write_crs("EPSG:4326", inplace=True)
    clipped_precipitation = precipitation.rio.clip(shapefile.geometry.values, shapefile.crs, drop=True)
    monthly_data = clipped_precipitation['tp'].resample(valid_time='ME').sum()
    monthly_data = monthly_data.mean(dim=['latitude', 'longitude']) * 1000 # Convert to mm
    monthly_sum.append(monthly_data)

tp_concatenated = xr.concat(monthly_sum, dim="valid_time")
tp_yearly_sum = tp_concatenated.resample(valid_time='YS').sum()

## Plot the total monthly precipitation from 2000 to 2020
fig, ax = plt.subplots(figsize=(12, 6))
tp_concatenated.plot(ax=ax, label='Monthly Precipitation')
tp_yearly_sum.plot(ax=ax, color='black', linestyle='--', label='Yearly Precipitation', marker='o')
ax.set_xticks(pd.date_range(start="2000", end="2021", freq="YS"))
ax.set_xticklabels([str(year) for year in range(2000, 2022)], rotation=45)
ax.set_title("Total Monthly Precipitation from 2000 to 2020")
ax.set_xlabel("Time")
ax.set_ylabel("Total Precipitation (mm)")
ax.grid(ls="--", alpha=0.5)
ax.legend()
plt.show()


##================================================================================##
## Total evaporation against time for the years 2000 to 2020 based on monthly sum ##
monthly_sum = []

for i in range(2000,2021):
    evaporation = xr.open_dataset(rf"C:\Users\vllja\Documents\VS Code\geo_env\data\lab07\Total_Evaporation\era5_OLR_{i}_total_evaporation.nc")
    evaporation.rio.write_crs("EPSG:4326", inplace=True)
    clipped_evaporation = evaporation.rio.clip(shapefile.geometry.values, shapefile.crs, drop=True)
    monthly_data = clipped_evaporation['e'].resample(valid_time='ME').sum()
    monthly_data = monthly_data.mean(dim=['latitude', 'longitude']) * (-1000) # Convert to mm
    monthly_sum.append(monthly_data)

e_concatenated = xr.concat(monthly_sum, dim="valid_time")
e_yearly_sum = e_concatenated.resample(valid_time='YS').sum()

## Plot the total monthly precipitation from 2000 to 2020
fig, ax = plt.subplots(figsize=(12, 6))
e_concatenated.plot(ax=ax, color='red', label='Monthly Evaporation')
e_yearly_sum.plot(ax=ax, color='black', linestyle='--', label='Yearly Evaporation', marker='o')
ax.set_xticks(pd.date_range(start="2000", end="2021", freq="YS"))
ax.set_xticklabels([str(year) for year in range(2000, 2022)], rotation=45)
ax.set_title("Total Monthly Evaporation from 2000 to 2020")
ax.set_xlabel("Time")
ax.set_ylabel("Total Evaporation (mm)")
ax.grid(ls="--", alpha=0.5)
ax.legend()
plt.show()

##===========================================================================##
## Total runoff against time for the years 2000 to 2020 based on monthly sum ##
monthly_sum = []

for i in range(2000,2021):
    runoff = xr.open_dataset(rf"C:\Users\vllja\Documents\VS Code\geo_env\data\lab07\Runoff\ambientera5_OLR_{i}_total_runoff.nc")
    runoff.rio.write_crs("EPSG:4326", inplace=True)
    clipped_runoff = runoff.rio.clip(shapefile.geometry.values, shapefile.crs, drop=True)
    monthly_data = clipped_runoff['ro'].resample(valid_time='ME').sum()
    monthly_data = monthly_data.mean(dim=['latitude', 'longitude']) * 1000 # Convert to mm
    monthly_sum.append(monthly_data)

ro_concatenated = xr.concat(monthly_sum, dim="valid_time")

## Plot the total monthly precipitation from 2000 to 2020
fig, ax = plt.subplots(figsize=(12, 6))
ro_concatenated.plot(ax=ax, color='green')
ax.set_xticks(pd.date_range(start="2000", end="2021", freq="YS"))
ax.set_xticklabels([str(year) for year in range(2000, 2022)], rotation=45)
ax.set_title("Total Monthly Runoff from 2000 to 2020")
ax.set_xlabel("Time")
ax.set_ylabel("Total Runoff (mm)")
ax.grid(ls="--", alpha=0.5)	
plt.show()


##==========================================##
## Plot all the variables from 2000 to 2020 ##
fig, ax = plt.subplots(figsize=(12, 6))
ro_concatenated.plot(ax=ax, label="Runoff", color='green')
e_concatenated.plot(ax=ax, label="Evaporation", color='red')
tp_concatenated.plot(ax=ax, label="Precipitation")
ax.set_xticks(pd.date_range(start="2000", end="2021", freq="YS"))
ax.set_xticklabels([str(year) for year in range(2000, 2022)], rotation=45)
ax.set_title("Total Monthly Runoff, Evapotranspiration, and Precipitation from 2000 to 2020")
ax.set_xlabel("Time")
ax.set_ylabel("Total (mm)")
ax.legend()
ax.grid(ls="--", alpha=0.5)	
plt.show()

# Plot the difference of Precipitation – (Total Evaporation + Runoff)
diff_concatenated = tp_concatenated - (e_concatenated + ro_concatenated)
fig, ax = plt.subplots(figsize=(12, 6))
diff_concatenated.plot(ax=ax, color='purple', label = 'Precipitation - (Evaporation + Runoff)')
ax.set_xticks(pd.date_range(start="2000", end="2021", freq="YS"))
ax.set_xticklabels([str(year) for year in range(2000, 2022)], rotation=45)
ax.set_title("Water Balance from 2000 to 2020")
ax.set_xlabel("Time")
ax.set_ylabel("Water Balance (mm)")
ax.grid(ls="--", alpha=0.5)	
ax.legend()
plt.show()

## Plot the Runoff estimate
bal_concatenated = tp_concatenated - e_concatenated
fig, ax = plt.subplots(figsize=(12, 6))
bal_concatenated.plot(ax=ax, label = 'Precipitation - Evaporation' , color='orange')
ro_concatenated.plot(ax=ax, label = 'Runoff', color='green')
ax.set_xticks(pd.date_range(start="2000", end="2021", freq="YS"))
ax.set_xticklabels([str(year) for year in range(2000, 2022)], rotation=45)
ax.set_title("Water Surplus from 2000 to 2020")
ax.set_xlabel("Time")
ax.set_ylabel("Runoff (mm)")
ax.legend()
ax.grid(ls="--", alpha=0.5)	
plt.show()