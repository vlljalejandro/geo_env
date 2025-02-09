import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr
import os

# Access the file and inspect contents
dset = xr.open_dataset(r'C:\Users\vllja\Documents\VS Code\Course_Data\Climate_Model_Data\tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc')
print((dset.keys()))
print(dset['tas'])
print(dset['tas'].dtype)

# Temporal span of each netCDF file
path = r'C:\Users\vllja\Documents\VS Code\Course_Data\Climate_Model_Data'
contents = os.listdir(path)
for i in contents:
    dset = xr.open_dataset(os.path.join(path, i))
    print(i, ' == ', dset['time'].values[0], ' == ', dset['time'].values[-1])

# Creation of Climate Change Maps
dset0 = xr.open_dataset(r'C:\Users\vllja\Documents\VS Code\Course_Data\Climate_Model_Data\tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc')
mean_1850_1900 = np.mean(dset0['tas'].sel(time=slice('18500101','19001231')), axis=0)

dset1 = xr.open_dataset(r'C:\Users\vllja\Documents\VS Code\Course_Data\Climate_Model_Data\tas_Amon_GFDL-ESM4_ssp119_r1i1p1f1_gr1_201501-210012.nc')
dset2 = xr.open_dataset(r'C:\Users\vllja\Documents\VS Code\Course_Data\Climate_Model_Data\tas_Amon_GFDL-ESM4_ssp245_r1i1p1f1_gr1_201501-210012.nc')
dset3 = xr.open_dataset(r'C:\Users\vllja\Documents\VS Code\Course_Data\Climate_Model_Data\tas_Amon_GFDL-ESM4_ssp585_r1i1p1f1_gr1_201501-210012.nc')

scenario1_2071_2100 = np.mean(dset1['tas'].sel(time=slice('20710101','21001231')), axis=0)
scenario2_2071_2100 = np.mean(dset2['tas'].sel(time=slice('20710101','21001231')), axis=0)
scenario3_2071_2100 = np.mean(dset3['tas'].sel(time=slice('20710101','21001231')), axis=0)

scenario1_2071_2100 = np.array(scenario1_2071_2100)-mean_1850_1900
scenario2_2071_2100 = np.array(scenario2_2071_2100)-mean_1850_1900
scenario3_2071_2100 = np.array(scenario3_2071_2100)-mean_1850_1900

lat_min, lat_max = -90, 90
lon_min, lon_max = -180, 180

plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
plt.imshow(mean_1850_1900, extent=[lon_min, lon_max, lat_min, lat_max],cmap='coolwarm',origin='lower')
plt.title('Temperature Mean 1850-1900')
cbar1 = plt.colorbar()
cbar1.set_label('True Mean Temperature (K)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.subplot(2, 2, 2)
plt.imshow(scenario1_2071_2100, extent=[lon_min, lon_max, lat_min, lat_max],cmap='coolwarm',origin='lower')
plt.title('Scenario 1')
cbar2 = plt.colorbar()
cbar2.set_label('Temperature Difference (K)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.subplot(2, 2, 3)
plt.imshow(scenario2_2071_2100,  extent=[lon_min, lon_max, lat_min, lat_max],cmap='coolwarm',origin='lower')
plt.title('Scenario 2')
cbar3 = plt.colorbar()
cbar3.set_label('Temperature Difference (K)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.subplot(2, 2, 4)
plt.imshow(scenario3_2071_2100,  extent=[lon_min, lon_max, lat_min, lat_max],cmap='coolwarm',origin='lower')
plt.title('Scenario 3')
cbar4 = plt.colorbar()
cbar4.set_label('Temperature Difference (K)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()