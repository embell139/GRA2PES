import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import glob
import xarray as xr
import pandas as pd
import os
import datetime

class inputs:
    sectors = ['Industrial','EGU','Other','Residential_Commercial','Onroad','Nonroad','total']
    years = [2021]
    months = ['Month'+datetime.datetime(1993,m,1).strftime('%m') for m in np.arange(1,13)]
    species = ['CO2','CO','NOX','SO2','PM25-PRI']
    data_dir = '/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES'
     
def check_totals(inputs):
    for year in inputs.years:
        print(year)
        for month in inputs.months:
            print(month)
            dirthis = f"{inputs.data_dir}/{year}/{month}"
            totfn = glob.glob(f"{dirthis}/*total*subset_regrid.nc4") 
            totals = xr.open_dataset(totfn[0])
            for s in inputs.species:
                print(s)
                tot = totals[s].values.copy()
                wnn = tot == -9999
                tot[wnn] = np.nan
                components = []
                for sector in inputs.sectors[0:-1]:
                    sfn = glob.glob(f"{dirthis}/*{sector}*subset_regrid.nc4")
                    print(sector)
                    sect = xr.open_dataset(sfn[0])
                    sect_vals = sect[s].values
                    wnns = sect_vals == -9999
                    sect_vals[wnns] = np.nan
                    print(np.nanmean(sect_vals))
                    components.append(sect_vals)
                    sect.close()
                components = np.array(components)
                components_sum = np.nansum(components,axis=0)
                diff = (tot - components_sum)/tot
                plt.pcolormesh(totals.lon,totals.lat,diff,)
                plt.colorbar()
                plt.savefig('/discover/nobackup/projects/gmao/geos_carb/embell/images/test.png')
                print('Median percent difference between totals:')
                print(f'    {np.nanmean(diff)}')
                breakpoint()
                
                    

if __name__ == '__main__':
    check_totals(inputs)
