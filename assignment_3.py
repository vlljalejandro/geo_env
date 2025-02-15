import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr
import tools

## Part 1: Downloading and Importing Jeddah Weather Data ##
df_isd = tools.read_isd_csv(r'C:\Users\vllja\Documents\VS Code\geo_env\data\41024099999.csv')
plot = df_isd.plot(title="ISD data for Jeddah")
# plt.show()

## Part 2: Heat Index (HI) Calculation ##
df_isd['RH'] = tools.dewpoint_to_rh(df_isd['DEW'].values,df_isd['TMP'].values)
df_isd['HI'] = tools.gen_heat_index(df_isd['TMP'].values,df_isd['RH'].values)
print(df_isd.max())
print(df_isd.idxmax())
print(df_isd.loc[["2023-08-21 10:00:00"]])

# Check for other dates
print(df_isd.loc[["2023-08-19 10:00:00"]])
print(df_isd.loc[["2023-08-20 10:00:00"]]) #Heatwave Event
print(df_isd.loc[["2023-08-21 10:00:00"]]) #Heatwave Event
print(df_isd.loc[["2023-08-22 10:00:00"]]) #Heatwave Event
print(df_isd.loc[["2023-08-23 10:00:00"]])

# Daily mean of HI
df_isd_resampled = df_isd.resample('D').mean()
df_isd_resampled['RH'] = tools.dewpoint_to_rh(df_isd_resampled['DEW'].values,df_isd_resampled['TMP'].values)
df_isd_resampled['HI'] = tools.gen_heat_index(df_isd_resampled['TMP'].values,df_isd_resampled['RH'].values)
dailyMean = df_isd['HI'].resample('D').mean()
hourly_vs_mean = df_isd_resampled['HI'].plot(label='Resampled Daily Data')
hourly_vs_mean = dailyMean.plot(title="Hourly vs Daily Mean Heat Index for Jeddah 2023", label='Daily Mean')
plt.ylabel('Heat Index')
plt.legend()
plt.savefig('HourlyvsDailyMean.png',dpi = 300)

# Time series of HI
HISeries = df_isd['HI'].plot(label='Hourly')
HISeries = dailyMean.plot(title="Heat Index for Jeddah 2023", label='Daily Mean')
plt.ylabel('Heat Index')
plt.legend()
plt.savefig('HeatIndex.png',dpi = 300)

## Part 3: Potential Impact of Climate Change ##
df_isd_projected = tools.read_isd_csv(r'C:\Users\vllja\Documents\VS Code\geo_env\data\41024099999.csv')
df_isd_projected['TMP'] = df_isd_projected['TMP'] + 3
df_isd_projected['RH'] = tools.dewpoint_to_rh(df_isd_projected['DEW'].values,df_isd_projected['TMP'].values)
df_isd_projected['HI'] = tools.gen_heat_index(df_isd_projected['TMP'].values,df_isd_projected['RH'].values)
print(df_isd_projected.max())