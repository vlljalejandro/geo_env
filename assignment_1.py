import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr
dset = xr.open_dataset(r'C:\Users\vllja\OneDrive\[2] Geo-Environmental Modeling\Labs and Assignments\Course_Data\SRTMGL1_NC.003_Data\N21E039.SRTMGL1_NC.nc') # Local direction of the file

DEM = np.array(dset.variables['SRTMGL1_DEM'])
DEM.shape  #To determine data dimensions

plt.imshow(DEM)
cbar = plt.colorbar()
cbar.set_label('Elevation (m asl)')

plt.savefig('assignment_1.png',dpi=300)  # To save the image