import cdsapi
import xarray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tools


## Part 1: Data Download ##

# dataset = "reanalysis-era5-single-levels"
# request = {
#     "product_type": ["reanalysis"],
#     "variable": [
#         "2m_temperature",
#         "total_precipitation"
#     ],
#     "year": [
#         "2019", "2020", "2021",
#         "2022", "2023", "2024"
#     ],
#     "month": [
#         "01", "02", "03",
#         "04", "05", "06",
#         "07", "08", "09",
#         "10", "11", "12"
#     ],
#     "day": [
#         "01", "02", "03",
#         "04", "05", "06",
#         "07", "08", "09",
#         "10", "11", "12",
#         "13", "14", "15",
#         "16", "17", "18",
#         "19", "20", "21",
#         "22", "23", "24",
#         "25", "26", "27",
#         "28", "29", "30",
#         "31"
#     ],
#     "time": [
#         "00:00", "01:00", "02:00",
#         "03:00", "04:00", "05:00",
#         "06:00", "07:00", "08:00",
#         "09:00", "10:00", "11:00",
#         "12:00", "13:00", "14:00",
#         "15:00", "16:00", "17:00",
#         "18:00", "19:00", "20:00",
#         "21:00", "22:00", "23:00"
#     ],
#     "data_format": "netcdf",
#     "download_format": "zip",
#     "area": [23, 39, 22, 40]
# }
# client = cdsapi.Client()
# client.retrieve(dataset, request).download()

# # Merge the data
# ds1 = xarray.open_dataset(r"C:\Users\vllja\Documents\VS Code\geo_env\data\ERA5\data_stream-oper_stepType-accum.nc")
# ds2 = xarray.open_dataset(r"C:\Users\vllja\Documents\VS Code\geo_env\data\ERA5\data_stream-oper_stepType-instant.nc")
# ds_merged = xarray.merge([ds1, ds2])
# ds_merged.to_netcdf(r"C:\Users\vllja\Documents\VS Code\geo_env\data\ERA5\era5_merged.nc")


# ## Part 2: Data Pre-Processing ##

dset = xarray.open_dataset(r"C:\Users\vllja\Documents\VS Code\geo_env\data\ERA5\era5_merged.nc")
t2m = np.array(dset.variables['t2m'])
tp = np.array(dset.variables['tp'])
latitude = np.array(dset.variables['latitude'])
longitude = np.array(dset.variables['longitude'])
time_dt = np.array(dset.variables['valid_time'])

t2m = t2m- 273.15
tp = tp * 1000

if t2m.ndim == 4:
 t2m = np.nanmean(t2m, axis=1)
 tp = np.nanmean(tp, axis=1)

# Air temperature and precipitation
df_era5 = pd.DataFrame(index=time_dt)
df_era5['t2m'] = t2m[:,3,2]
df_era5['tp'] = tp[:,3,2]
ylabel = 'Temperature (°C) / Precipitation (mm)'
legend = ['Temperature (°C)', 'Precipitation (mm)'] 
df_era5.plot(legend=legend, ylabel=ylabel) 
plt.legend(legend)
plt.show()

# Average annual precipitation
annual_precip = df_era5['tp'].resample('YE').mean()*24*365.25
mean_anual_precip = np.nanmean(annual_precip)
print(f"Mean annual precipitation: {mean_anual_precip} mm") 


## Part 3: Calculation of Potential Evaporation (PE) ##

tmin = df_era5['t2m'].resample('D').min().values
tmax = df_era5['t2m'].resample('D').max().values
tmean = df_era5['t2m'].resample('D').mean().values
lat = 21.25
doy = df_era5['t2m'].resample('D').mean().index.dayofyear
pe = tools.hargreaves_samani_1982(tmin, tmax, tmean, lat, doy)

# Mean PE
mean_pe = np.nanmean(pe)
print(f"Mean potential evaporation: {mean_pe} mm/day")

# Plot the PE time series
ts_index = df_era5['t2m'].resample('D').mean().index
plt.figure()
plt.plot(ts_index, pe, label='Potential Evaporation')
plt.xlabel('Time')
plt.ylabel('Potential evaporation (mm/day)')
plt.show()  