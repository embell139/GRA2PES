import matplotlib
matplotlib.use('Agg')
import xarray as xr
from cartopy import crs as ccrs
import cartopy.feature as cf
import contextily as cx
import numpy as np
import datetime
from matplotlib import pyplot as plt
"""
Spot-check! Anything crazy in these maps?
"""

def map_gas(lon,lat,vals,title=None,savename=None,cmap='viridis'):
    fig = plt.figure(figsize=(15,8)) 
    ax = fig.add_subplot(projection = ccrs.PlateCarree())
    vmax = np.nanmean(vals) + 2*np.nanstd(vals)
    #vmax = 150
    #breakpoint()
    mapp = ax.pcolormesh(lon,lat,vals,vmin=0,vmax=vmax,alpha=0.7,transform=ccrs.PlateCarree(),cmap=cmap)
    cx.add_basemap(ax,crs=ccrs.PlateCarree(),source=cx.providers.CartoDB.Positron)
    ax.coastlines(lw=0.25)
    ax.add_feature(cf.BORDERS,lw=0.25)
    ax.add_feature(cf.STATES,lw=0.25)
    plt.title(title)
    fig.colorbar(mapp)
    if savename:
        plt.savefig(savename)
        print('Saved ',savename)
    plt.close('all')
    return


if __name__ == '__main__':
    base_dir = '/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES'
    image_dir = '/discover/nobackup/projects/gmao/geos_carb/embell/images/GRA2PES/basic_maps'
    years = [2021]
    months = ['Month'+datetime.datetime(1993,month,1).strftime('%m') for month in np.arange(1,13)]
    sectors = ['total']
    version = 'GRA2PESv1.0'

    cmaps = {
        'CO2':'Spectral_r',
        'CO':'Blues',
        'PM25-PRI':'YlOrBr',
        'NOX':'RdPu',
        'SO2':'Greens'
    }

    for sector in sectors:
        for year in years:
            for month in months:
                fn = f"{base_dir}/{year}/{month}/{version}_{sector}_{datetime.datetime(year,int(month[-2::]),1).strftime('%Y%m')}_subset_regrid.nc4"
                data = xr.open_dataset(fn)
                for v in data.drop_vars('crs').data_vars:
                #for v in ['CO2']:
                    title = f'{sector} {v}, {month} {year}' 
                    sn = f'{image_dir}/{sector}_{month}_{year}_{v}_map.png' 
                    vals = data[v].values
                    wnn = vals == -9999
                    vals[wnn] = np.nan 
                    print(np.nanmean(vals))
                    #breakpoint()
                    map_gas(data['lon'].values,data['lat'].values,data[v].values,title=title,savename=sn,cmap=cmaps[v])
