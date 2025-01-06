#import h5py
import netCDF4 as nc
from matplotlib import pyplot as plt
import numpy as np
import glob
import cartopy.crs as ccrs
import cartopy.feature as cf
from datetime import datetime
import xarray
import rasterio
#---
# FUNCTIONS
# ---
# Convert moles of gas to grams of gas.
# Inputs: Emissions in moles, species name
def mol_to_g(val,s):
    grams_per_mole = {
        'CO': 12.0096+15.99903,
        'CO2':12.0096+2.*15.99903,
        'CH4':12.0096+4.*1.00784,
        'NO':14.007+15.99903,
        'NO2':14.007+2.*15.99903
    }
    return val*grams_per_mole[s]

def area_weight(lon,lat,z):
    # NOTE may not be in correct units yet - this is copy/pasted from Tom's LPJ notebook
    # cell area in m2
    global_cellarea = 111.1 * 111.1 * 0.5 * 0.5 * np.cos(np.radians(lat)) * 1000 * 1000
    global_cellarea = np.tile(global_cellarea, (len(lon),1)).T
    # weight the data with the cell area - units are g/m2
    weights = global_cellarea/sum(global_cellarea)
    weighted = z*weights
    return weighted 

def map_gas():
    # map with coastlines
    fig = plt.figure(1,figsize=(12,6))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    if region == '':
        ax.set_extent([lon.min(),lon.max(),lat.min(),50])   # [x0,x1,y0,y1]
    else:
        ax.set_extent(bounds[region])

    p = ax.pcolormesh(lon,lat,gas,cmap=cmap,vmin=0,vmax=40,transform=ccrs.PlateCarree())

    ax.coastlines(lw=0.3,color='gray')
    ax.add_feature(cf.BORDERS,lw=0.3,color='gray')

    # add and format colorbar
    cb = plt.colorbar(p)
    cb.set_label(
        #label=data[species].attrs['units'].decode(),
        #label="g m$^{-2}$ hr$^{-1}$",
        label = ds[species].units,
        fontsize=14,
        labelpad=15
    )
    cb.ax.tick_params(labelsize=13)
            
    # add gridlines
    grid = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.15,color='gray')
    grid.top_labels=False
    grid.right_labels=False
    grid.xlabel_style={'size':14}
    grid.ylabel_style={'size':14}
    
    # add title
    ti = f"{f.split('/')[-5]} {species} {'/'.join(f.split('/')[-3:-1])} {t.split('_')[-1][0:2]}Z"
    if region != '':
        ti = ti+f' {region}'
    plt.title(ti,size=18)
    # Example title: onroad_gasoline CO2 Month01/weekdy 00Z' 
    # construct output filename
    fn = dataset+'_'+ti.replace(' ','_').replace('/','_')
    # Example output filename: COVID-AQS_onroad_gasoline_CO2_Month01_weekdy_00Z

    plt.savefig(f'{image_dir}{fn}.png',dpi=300)
    print(f'Saved {image_dir}{fn}.png')
    
    plt.close()
    return
# ---------
# ---------
# ---------
# MAIN CODE
# ---------
# ---------
# ---------
dataset = 'GRAAPES'

region = 'New York'
bounds = {
    'Los Angeles':[-119.36,-116.70,33.35,34.52],
    'New York':[-74.72,-72.85,40.34,41.26],
    'Baltimore':[-76.9,-76.32,39.14,39.46]
}

months = ['Month'+datetime(2020,x,1).strftime('%m') for x in np.arange(1,13,1)]
days = ['weekdy','satdy','sundy']      # satdy, sundy, weekdy
species = ['CO2']   # hourly emissions in mol km-2 hr-1
                      # to start: CO2, CO, NO, NO2
keep =['Times','E_CO2','E_CO','E_NO2','XLAT','XLONG'] # variables from original NetCDF files that we're interested in for R2

cmap = 'plasma'
image_dir = f'/discover/nobackup/projects/gmao/geos_carb/embell/images/{dataset}/'

for month in months:
    for day in days:

        ff = glob.glob(f"/discover/nobackup/projects/gmao/geos_carb/embell/data/GRAAPES/{month}/{day}/*")
        #ff = glob.glob(f"/discover/nobackup/projects/gmao/geos_carb/embell/data/COVID-AQS/emissions/FIVE/onroad_gasoline/2021/{month}/{day}/*")
        if len(ff) == 0:
            print(f"{month}/{day} not found - skipping.")
            continue
        
        for i,f in enumerate(ff):
            #if i > 0: continue
        
            #ds = nc.Dataset(f)
            ds = xarray.open_dataset(f)

            # Before any calculations, remove variables we don't want to keep
            remove = [v not in keep for v in ds.variables]
            remove_vars = np.array([v for v in ds.variables])[remove]
            ds = ds.drop_vars(remove_vars)
    
            # Average across z dimension
            ds = ds.mean(dim='emissions_zdim')
            # Average across time dimension
            ds = ds.mean(dim='Time')

            # Update variable attributes
            for s in keep[1:4]:
                subset[s].attrs = ds[s].attrs

            # Native units are mol km-2 hr-1 - convert to g m-2 hr-1
            # Molecular masses from https://webbook.nist.gov/chemistry/form-ser/
            #subset['E_CO2'][:] = (subset['E_CO2'][:]*44.0095)/(1000.*1000.)
            #subset['E_CO'][:] = (subset['E_CO'][:]*28.0101)/(1000.*1000.)
            #subset['E_NO2'][:] = (subset['E_NO2'][:]*46.0055)/(1000.*1000.)
    
            #lat = ds['XLAT'][:]
            #lon = ds['XLONG'][:]
        
            # Convert {moles per km2 per hr} to {grams per m2 per hr}
            #gas = mol_to_g(gas,species)/(1000.*1000.)
        
            # Call our function
            #map_gas()
            breakpoint() 
        #ds.close()
    
