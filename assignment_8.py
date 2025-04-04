import xarray as xr
import geopandas as gpd
import numpy as np
import scipy.optimize as opt
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#os.chdir(os.path.abspath(''))
os.chdir(os.path.dirname(__file__))
print(os.getcwd())


## ---Part 1: Pre-Processing ERA5 dataset ---

# Clip each variable using the shapefile
def load_and_clip(nc_file, var_name, gdf):
    ds = xr.open_dataset(nc_file)
    ds = ds.rio.write_crs("EPSG:4326")  # Ensure correct CRS
    clipped = ds.rio.clip(gdf.geometry, gdf.crs, drop=True)
    return clipped[var_name]

# Load the watershed shapefile
shapefile_path = r"C:\Users\vllja\Documents\VS Code\geo_env\data\lab08\WS_3.shp"
gdf = gpd.read_file(shapefile_path)

# Load the NetCDF files (precipitation, ET, runoff)
precip_file = r"C:\Users\vllja\Documents\VS Code\geo_env\data\lab07\Precipitation\era5_OLR_2001_total_precipitation.nc"
et_file = r"C:\Users\vllja\Documents\VS Code\geo_env\data\lab07\Total_Evaporation\era5_OLR_2001_total_evaporation.nc"
runoff_file = r"C:\Users\vllja\Documents\VS Code\geo_env\data\lab07\Runoff\ambientera5_OLR_2001_total_runoff.nc"
precip_file2 = r"C:\Users\vllja\Documents\VS Code\geo_env\data\lab07\Precipitation\era5_OLR_2002_total_precipitation.nc"
et_file2 = r"C:\Users\vllja\Documents\VS Code\geo_env\data\lab07\Total_Evaporation\era5_OLR_2002_total_evaporation.nc"
runoff_file2 = r"C:\Users\vllja\Documents\VS Code\geo_env\data\lab07\Runoff\ambientera5_OLR_2002_total_runoff.nc"

# Load and clip each dataset,unit conversion: meters to mm
P_grid = load_and_clip(precip_file, "tp", gdf) * 1000.0
ET_grid = load_and_clip(et_file, "e", gdf) * 1000.0
Q_grid = load_and_clip(runoff_file, "ro", gdf) * 1000.0
P_grid2 = load_and_clip(precip_file2, "tp", gdf) * 1000.0
ET_grid2 = load_and_clip(et_file2, "e", gdf) * 1000.0
Q_grid2 = load_and_clip(runoff_file2, "ro", gdf) * 1000.0

# Compute area-averaged values
P = P_grid.mean(dim=["latitude", "longitude"]).values
ET = ET_grid.mean(dim=["latitude", "longitude"]).values
Q_obs = Q_grid.mean(dim=["latitude", "longitude"]).values
P2 = P_grid2.mean(dim=["latitude", "longitude"]).values
ET2 = ET_grid2.mean(dim=["latitude", "longitude"]).values
Q_obs2 = Q_grid2.mean(dim=["latitude", "longitude"]).values

# Ensure ET is positive
ET = np.where(ET < 0.0, -ET, 0.0) 
ET2 = np.where(ET2 < 0.0, -ET2, 0.0)

# Plot the data against time (2001 and 2002)
ET3 = np.concatenate((ET, ET2))
P3 = np.concatenate((P, P2))
Q_obs3 = np.concatenate((Q_obs, Q_obs2))

time = np.arange(len(P3))
plt.figure(figsize=(12, 6))
plt.plot(time, P3, label='Precipitation (mm)', color='blue')
plt.plot(time, ET3, label='Evaporation (mm)', color='orange')
plt.plot(time, Q_obs3, label='Runoff (mm)', color='green')
print(np.arange(0, len(P3), ))
plt.xticks(np.arange(0, len(P3), 730), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlim(0, len(P3))
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('Water (mm)')
plt.title('Precipitation, Evaporation, and Runoff Time Series for 2001 and 2002')
plt.legend()
plt.grid()
plt.show()

## --- Part 2: Model setup and validation ---

# Rainfall-runoff model (finite difference approximation)
def simulate_runoff(k, P, ET, dt=1):
    n = len(P)
    Q_sim = np.zeros(n)
    Q_sim[0] = Q_obs[0]
    
    for t in range(2, n):
        Q_t = (Q_sim[t-1] + (P[t] - ET[t]) * dt) / (1 + dt/k)
        Q_sim[t] = max(0,Q_t) # Ensure non-negative runoff

    return (Q_sim)

# Define the objective (KGE) function
def kge(Q_obs, Q_sim):
    r = np.corrcoef(Q_obs, Q_sim)[0, 1]
    alpha = np.std(Q_sim) / np.std(Q_obs)
    beta = np.mean(Q_sim) / np.mean(Q_obs)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return (kge, r, alpha, beta)

# --- Validation 2001 (k = 0.015) ---
Index = np.empty([1 , 5]) 
k_test = 0.15
Q_sim = simulate_runoff(k_test, P, ET)
PerfIndex = kge(Q_obs, Q_sim)
Index[0,0] = k_test
Index[0,1:] = PerfIndex  #for k, kge, r, alpha, beta
print (f'Validation 2001 (k = 0.15) → {Index}')

# Plot Q_obs and Q_sim_all against time 2001 (k = 0.015)
time = np.arange(len(P))
plt.figure(figsize=(12, 6))
plt.plot(time, Q_obs, label='Observed Runoff (mm)', color='green')
plt.plot(time, Q_sim, label='Simulated Runoff (mm)', linestyle='--')
plt.xticks(np.arange(0, len(P), 730), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlim(0, len(P))
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('Runoff (mm)')
plt.title('Observed and Simulated Runoff Time Series (k = 0.15, 2001)')
plt.legend()
plt.show()

# Plot scatterplot of Q_obs vs Q_sim 2001 (k = 0.015)
plt.figure(figsize=(7, 7))
plt.scatter(Q_obs, Q_sim, label='k = 0.15', color='blue', alpha=0.5)
plt.xlabel('Observed Runoff (mm)')
plt.ylabel('Simulated Runoff (mm)')
plt.title('Observed vs Simulated Runoff (k = 0.15, 2001)')
plt.plot([0, max(Q_obs)], [0, max(Q_obs)], color='red', linestyle='--', label='1:1 Line')
plt.legend()
plt.grid()
plt.show()

# ## --- Part 3: Model calibration ---

# ## KGE Optimization ##

# Objective function for optimization
def objective(k, P, ET, Q_obs):
    Q_sim = simulate_runoff(k, P, ET)
    kge_model = kge(Q_obs, Q_sim)
    return (1.0 - kge_model[0])

# Optimize k using KGE
res = opt.minimize_scalar(objective, bounds=(0.1, 2), args=(P, ET, Q_obs), method='bounded')
print(res)

# Best k value
best_k = res.x
print(f"Optimized k: {best_k:.3f}")

# --- Validation 2001 (k = 0.509) ---
Index_best = np.empty([1 , 5]) 
Q_sim_best = simulate_runoff(best_k, P, ET)
PerfIndex_best = kge(Q_obs, Q_sim_best)
Index_best[0,0] = best_k
Index_best[0,1:] = PerfIndex_best #for k, kge, r, alpha, beta
print (f'Validation 2001 (k = 0.509) → {Index_best}')

# Plot Q_obs and Q_sim against time 2001 (k = 0.509)
time = np.arange(len(P))
plt.figure(figsize=(12, 6))
plt.plot(time, Q_obs, label='Observed Runoff (mm)', color='green')
plt.plot(time, Q_sim_best, label='Simulated Runoff (mm)', linestyle='--')
plt.xticks(np.arange(0, len(P), 730), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlim(0, len(P))
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('Runoff (mm)')
plt.title('Observed and Simulated Runoff Time Series (k = 0.509, 2001)')
plt.legend()
plt.show()

# Plot scatterplot of Q_obs vs Q_sim 2001 (k = 0.509)
plt.figure(figsize=(7, 7))
plt.scatter(Q_obs, Q_sim_best, label='k = 0.509', color='blue', alpha=0.5)
plt.xlabel('Observed Runoff (mm)')
plt.ylabel('Simulated Runoff (mm)')
plt.title('Observed vs Simulated Runoff (k = 0.509, 2001)')
plt.plot([0, max(Q_obs)], [0, max(Q_obs)], color='red', linestyle='--', label='1:1 Line')
plt.legend()
plt.grid()
plt.show()

# --- Validation 2002 (k = 0.509) ---
Index2_best = np.empty([1 , 5]) 
Q2_sim_best = simulate_runoff(best_k, P2, ET2)
PerfIndex2_best = kge(Q_obs, Q2_sim_best)
Index2_best[0,0] = best_k
Index2_best[0,1:] = PerfIndex2_best #for k, kge, r, alpha, beta
print (f'Validation 2002 (k = 0.509) → {Index2_best}')

# Plot Q_obs and Q_sim against time 2002 (k = 0.509)
time = np.arange(len(P2))
plt.figure(figsize=(12, 6))
plt.plot(time, Q_obs, label='Observed Runoff (mm)', color='green')
plt.plot(time, Q2_sim_best, label='Simulated Runoff (mm)', linestyle='--')
plt.xticks(np.arange(0, len(P), 730), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlim(0, len(P))
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('Runoff (mm)')
plt.title('Observed and Simulated Runoff Time Series (k = 0.509, 2002)')
plt.legend()
plt.show()

# Plot scatterplot of Q_obs vs Q_sim 2002 (k = 0.509)
plt.figure(figsize=(7, 7))
plt.scatter(Q_obs2, Q2_sim_best, label='k = 0.509', color='blue', alpha=0.5)
plt.xlabel('Observed Runoff (mm)')
plt.ylabel('Simulated Runoff (mm)')
plt.title('Observed vs Simulated Runoff (k = 0.509, 2002)')
plt.plot([0, max(Q_obs2)], [0, max(Q_obs2)], color='red', linestyle='--', label='1:1 Line')
plt.legend()
plt.grid()
plt.show()