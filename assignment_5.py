import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import geopandas as gpd

## Data visualization and inspection ##

dset = xr.open_dataset(r'C:\Users\vllja\Documents\VS Code\geo_env\data\Gridsat_data\GRIDSAT-B1.2009.11.25.06.v02r01.nc')
IR = np.array(dset.variables['irwin_cdr']).squeeze()
IR = np.flipud(IR)
IR = IR * 0.01 + 200
IR = IR - 273.15

plt.figure(1)
plt.imshow(IR, extent=[-180.035, 180.035,-70.035, 70.035], aspect='auto')
cbar = plt.colorbar()
cbar.set_label('Brightness temperature (degrees Celsius)')
jeddah_lat = 21.5
jeddah_lon = 39.2
plt.scatter(jeddah_lon, jeddah_lat, color='red', marker='o', label ='Jeddah')
plt.show()
plt.clf()

## Rainfall estimation ##

folderPath = r'C:\Users\vllja\Documents\VS Code\geo_env\data\Gridsat_data'
files = os.listdir(folderPath)
cumulate = None

for i in files:
    dset = xr.open_dataset(fr'C:\Users\vllja\Documents\VS Code\geo_env\data\Gridsat_data\{i}')
    jeddah_ir = dset['irwin_cdr'].sel(lat=slice(18, 28), lon=slice(35,45)).squeeze()
    jeddah_ir = np.flipud(jeddah_ir)
    jeddah_ir = jeddah_ir*0.01 + 200
    temp = -3.6382* 0.01 * xr.apply_ufunc(np.power, jeddah_ir, 1.2)
    rainfall = 3*(1.1183* 10**11 * xr.apply_ufunc(np.exp, temp))
    print(f'The maximum rainfall in {i} is {rainfall.max()} mm')
    if cumulate is None:
        cumulate = rainfall
    else:
        cumulate = cumulate + rainfall

plt.figure(1)
plt.imshow(cumulate,extent=[35, 45 ,18 , 28], aspect='equal') 
x_label = 'Longitude'
y_label = 'Latitude'
plt.xlabel(x_label)
plt.ylabel(y_label)
cbar = plt.colorbar()
cbar.set_label('Cumulate Rainfall (mm)')
jeddah_lat = 21.5
jeddah_lon = 39.2
plt.scatter(jeddah_lon, jeddah_lat, color='red', marker='o', label ='Jeddah')
plt.show()
