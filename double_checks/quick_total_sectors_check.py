import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
""" 
Outputs maps of reported Total emissions,
reported individual sector emissions,
and a map of % diff between reported Total and sum of sectors.
"""

#totalfile = '/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES/2021/Month01/total/weekdy/GRA2PESv1.0_total_202101_weekdy_00to11Z.nc'
totalfile = '/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES/2021/Month01/GRA2PESv1.0_total_202101_subset_regrid.nc4'

ddir = '/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES/2021/Month01'
#filebase = 'GRA2PESv1.0_sector_202101_weekdy_00to11Z.nc'
filebase = 'GRA2PESv1.0_sector_202101_subset_regrid.nc4'
#sectors = ['EGU','Industrial','Onroad','OFFROAD','RAIL','AVIATION','SHIPPING','RES','COMM','COOKING','WASTE','VCP','AG','FUG']
sectors = ['EGU','Industrial','Onroad','Nonroad','Other','Residential_Commercial']


sectorfiles = []
for sector in sectors:
    #sectorfiles.append(f"{ddir}/{sector}/weekdy/{filebase.replace('sector',sector)}")
    sectorfiles.append(f"{ddir}/{filebase.replace('sector',sector)}")

total_data = xr.open_dataset(totalfile)
total_co2 = total_data['CO2'].values
wnn = total_co2 == -9999
total_co2[wnn] = np.nan
#vals = total_co2[0,0,:,:]
vals = total_co2
#plt.pcolormesh(total_data.XLONG,total_data.XLAT,vals,vmin=0,vmax=np.nanmean(vals)+2*np.nanstd(vals))
plt.pcolormesh(total_data.lon,total_data.lat,vals,vmin=0,vmax=np.nanmean(vals)+2*np.nanstd(vals))
plt.title('total')
plt.colorbar()
plt.savefig('/discover/nobackup/projects/gmao/geos_carb/embell/images/total.png')

print('Reading component CO2 fields')
components = []
for s,sector in zip(sectorfiles,sectors):
    #if sector != 'EGU':
    #    continue
    sector_data = xr.open_dataset(s)
    components.append(sector_data['CO2'].values)
    #vals = sector_data['CO2'].values[0,0,:,:]
    vals = sector_data['CO2'].values
    wnnv = vals == -9999
    vals[wnnv] = np.nan
    plt.close('all')
    #plt.pcolormesh(sector_data.XLONG,sector_data.XLAT,vals,vmin=0,vmax=np.nanmean(vals)+2*np.nanstd(vals))
    plt.pcolormesh(sector_data.lon,sector_data.lat,vals,vmin=0,vmax=np.nanmean(vals)+2*np.nanstd(vals))
    plt.title(sector)
    plt.colorbar()
    plt.savefig(f'/discover/nobackup/projects/gmao/geos_carb/embell/images/{sector}.png')
    #breakpoint()
components = np.array(components)
wnnc = components == -9999
components[wnnc] = np.nan
components_sum = np.sum(components,axis=0)

diff = (total_co2 - components_sum)/total_co2
print(np.nanmean(diff))
print(np.nanmax(diff))
plt.close('all')
plt.pcolormesh(total_data.lon,total_data.lat,diff,vmin=0,vmax=0.05)
plt.title('diff as fraction of Total')
plt.colorbar()
plt.savefig(f'/discover/nobackup/projects/gmao/geos_carb/embell/images/diff.png')



    
