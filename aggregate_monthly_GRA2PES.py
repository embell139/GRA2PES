"""
Author: Emily Bell
Contact: emily.i.bell@nasa.gov
Description: This code takes native GRA2PES data from specified sectors and aggregates it 
to get average monthly emission rates. Currently, it's built to handle five species:
CO2, CO, NOX, PM2.5, and SO2. 

Steps, not necessarily in order:

- Native GRA2PES emission contain 20 vertical levels, which we sum to get total column emissions.

- Data are regridded from native Lambert Conformal projection and 4000m resolution to EPSG 4326
  and spatial resolution defined by <inputs.degx> and <inputs.degy>.

- Moles are converted to metric tons for consistency where applicable.

- Native hourly emissions rates are averaged for each 12-hour file.

-  0-11Z and 12-23Z emission rates are averaged for a daily emissions rate, for each represented day: satdy, sundy, weekdy.

- Satdy, sundy, and weekdy emissions rates are weighted as appropriate to the given calendar month to calculate
  an average hourly emissions rate for that month.

- The average hourly emissions rate is multiplied by # hours in a month, for units of metric tons km^-2 month^-1. 

- Appropriate metadata is added.

- Output is saved in NetCDF4 format.
"""

import numpy as np
import glob
import os
import xarray as xr
# for mapping
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import traceback
# for regridding
import pyproj
import gc
import xesmf as xe
# for aggregating in time 
import calendar
from calendar import monthrange
from datetime import datetime

class inputs():   
    dataset = 'GRA2PES'
    example_fn = f'/discover/nobackup/projects/gmao/geos_carb/embell/data/{dataset}/2021/Month08/total/weekdy/GRA2PESv1.0_total_202108_weekdy_00to11Z.nc'
    data_dir_base = f'/discover/nobackup/projects/gmao/geos_carb/embell/data/{dataset}'
    output_dir_base = data_dir_base
    years = ['2021']
    months = ['Month'+datetime(1993,m,1).strftime('%m') for m in np.arange(1,13)]
    #months = ['Month04'] 
    days = ['weekdy','satdy','sundy']
    halves = ['00to11Z','12to23Z']
    input_file = 'GRA2PESv1.0_[SECT]_[YYYYMM]_[DD]_[HH].nc'
    output_fn = f"{os.path.basename(example_fn).split('_')[0]}_[SECT]_[YYYYMM]_subset_regrid.nc4"
    sectors = ['Other']

    # for producing subset files
    keep = ['Times','CO2','CO','XLAT','XLONG','NOX','PM25-PRI','SO2'] # variables from original NetCDF files that we're interested in for R2
    #sums = {
    #    'SOX': ['SO2']
    #}
    descriptions = {
        'CO2': 'Carbon Dioxide',
        'CO': 'Carbon Monoxide',
        'NOX': 'Nitrogen Oxides (NOX)',
        'SO2': 'Sulfur Oxides (SOX)'
    }
    degx, degy = 0.036, 0.036     # resolution to regrid to - estimated to match native 4km (111.1km ~ 1 deg) 
    reuse_weights = False       # when regridding

    # for plotting
    region = 'New York'
    bounds = {
        'Los Angeles':[-119.36,-116.70,33.35,34.52],
        'New York':[-74.72,-72.85,40.34,41.26],
        'Baltimore':[-76.9,-76.32,39.14,39.46]
    }
    cmap = 'plasma'
    image_dir = f'/discover/nobackup/projects/gmao/geos_carb/embell/images/{dataset}/'

# ---------
# REGRIDDING
# these functions pulled from regrid_wrfchemi_v3.py
# ---------
def make_grid_RCM(domain_example_fn):
    out_example = xr.open_dataset(domain_example_fn, chunks={'Time': 1}, engine='netcdf4')
    wrf_proj = pyproj.Proj(
        proj='lcc', # projection type: Lambert Conformal Conic
        lat_1=out_example.TRUELAT1, lat_2=out_example.TRUELAT2, # Cone intersects with the sphere 33,45
        lat_0=out_example.MOAD_CEN_LAT, lon_0=out_example.STAND_LON, # Center point :MOAD_CEN_LAT = 39.34594f ; :STAND_LON = -97.f ;
        a=6370000, b=6370000
    ) # Radius#a=6378137, b=6378137) # Radius
    # More info here: https://fabienmaussion.info/2018/01/06/wrf-projection/
    latlon_proj = pyproj.Proj(proj='latlong',ellps='WGS84',datum ='WGS84')
                   
    # Construct Grid for WRF
    e, n = pyproj.transform(latlon_proj,wrf_proj,out_example.CEN_LON, out_example.CEN_LAT)
                   
    # Grid parameters
    dx_wrf, dy_wrf = out_example.DX, out_example.DY
    nx_wrf, ny_wrf = out_example.dims['west_east'], out_example.dims['south_north']
    # Down left corner of the domain
    x0_wrf = -(nx_wrf-1) / 2. * dx_wrf + e
    y0_wrf = -(ny_wrf-1) / 2. * dy_wrf + n
                   
    # Grid of Grid Centers
    xx_wrf, yy_wrf = np.meshgrid(np.arange(nx_wrf) * dx_wrf + x0_wrf, np.arange(ny_wrf) * dy_wrf + y0_wrf)
    #Transformation of Center X-Y to Center Lat-Lon
    lon_wrf, lat_wrf = pyproj.transform(wrf_proj,latlon_proj,xx_wrf,yy_wrf)
                   
    # Calculating the boundary X-Y Coordinates
    x_b_wrf, y_b_wrf = np.meshgrid(np.arange(nx_wrf+1) * dx_wrf + x0_wrf -dx_wrf/2, np.arange(ny_wrf+1) * dy_wrf + y0_wrf -dy_wrf/2)
    #Transformation of Boundary X-Y to Boundary Lat_Lon
    lon_b_wrf, lat_b_wrf = pyproj.transform(wrf_proj,latlon_proj,x_b_wrf,y_b_wrf)
    grid = {'lat': lat_wrf, #Center Point Spacing Lat
        'lon': lon_wrf, #Center Point Spacing Lon
        'lat_b': lat_b_wrf, # Boundary Spacing Lat
        'lon_b': lon_b_wrf, # Boundary Spacing Lon
    }
                    
    return grid 

def make_grid_out():
                   
    # minlat = 38.4    
    # maxlat = 39.60   
    # minlon = -77.8   
    # maxlon = -76.2   
    # dlat = 0.01 degrees
    # dlon = 0.01 degrees
    # The southwest corner of the domain is 38.4, -77.8, so the center of the southwest corner grid cell is 38.405, -77.795.

    #TODO: really shouldn't hard code lon/lat min/mix
    # should pull this from the data
    lon = np.arange(-137.2963,-58.576263,inputs.degx)
    lat = np.arange(18.191376,52.22797,inputs.degy)
                   
    nx = len(lon)  
    ny = len(lat)  
                   
    lon_b = np.zeros([nx+1])
    lon_b[0:nx] = lon-inputs.degx/2
    lon_b[nx] = lon[nx-1]+inputs.degx/2
                   
    lat_b = np.zeros([ny+1])
    lat_b[0:ny] = lat-inputs.degy/2
    lat_b[ny] = lat[ny-1]+inputs.degy/2

    grid =  {'lat': lat, #Center Point Spacing Lat
                    'lon': lon, #Center Point Spacing Lon
                    'lat_b': lat_b, # Boundary Spacing Lat
                    'lon_b': lon_b, # Boundary Spacing Lon
                    }
                    
    return grid    
                   
def save_ncf(ds,out_fn,mode='w'):
    # Write to NetCDF
    print('----Writing to NetCDF')
    #for c, var in enumerate(ds_month.data_vars):
    #    ds_out = ds_month[var]
    #    ds_out[var].attrs = ds_month[var].attrs
    #    ds_out.attrs = {}
    #    del ds_out[var].attrs['MemoryOrder'],ds_out[var].attrs['grid_mapping'],ds_out[var].attrs['FieldType']
    #    
   #     if c == 0:
   #         print('Creating File: ', out_fn)
   #         print('Appending '+var)
   #         save_ncf(ds_out,out_fn,data_vars=ds_out.data_vars,mode='w')
   #     else:
   #         print('Appending '+var)
   #         save_ncf(ds_out,out_fn,data_vars=ds_out.data_vars,mode='a'

    for var in ds.data_vars:
        # Set all encoding settings properly
        if var != 'spatial_ref':
            encoding_dict = {'dtype': 'float32', 'chunksizes':(ds.sizes['lat'], ds.sizes['lon']),
                'zlib': True, 'complevel': 1, '_FillValue': None }
        ds[var].attrs['encoding']=encoding_dict
#        else:      
#            ds['Times'].encoding={'char_dim_name':'DateStrLen'}
#                   
#    #print('W massriting file: ', out_fn)
    ds.to_netcdf(out_fn,format='netCDF4',engine='netcdf4',mode=mode)
                   
def add_var(data,regridder,ds):
    data.append(regridder(ds))

def reformat(ds,regridder):
    change_units = ['CO2','CO','SO2','NOX'] 
    # Convert to metric tons per km2 per hour
    for c in change_units:
        c_tons = moles_to_tons(ds[c].values,c)
        dims = ds[c].dims
        # Can't just change existing values, 
        # so we'll delete the old c and make a new one
        ds = ds.drop_vars(c)
        ds[c] = (dims,c_tons)

    # Only keep relevant variables
    # This also calculates combination parameters, like SOx
    ds_in = subset(ds)
     
    # Renaming after regridder to match wrfchemi dimensions and coordinate names
    rename_dict = {'south_north':'y',
                    'west_east':'x',
                    'XLAT':'lat',
                    'XLONG':'lon'
                    }
    ds_in = ds_in.rename(rename_dict)
    ds_in = ds_in.set_coords(names=('lat','lon'))
    ds_in = ds_in.transpose('Time','bottom_top','y','x')
    
    # Aggregate in z, time
    ds_day = aggregate_z_t(ds_in)
    # Regrid
    print('----Regridding')
    ds_day = regridder(ds_day)
    for v in ds_day.variables:
        ds_day[v].attrs = ds_in[v].attrs
        ds_day[v].attrs['units'] = 'metric tons km^-2 month^-1'
        if v in inputs.descriptions.keys():
            ds_day[v].attrs['description'] = inputs.descriptions[v]
    #for s in inputs.sums.keys():
    #    if s == 'SOX':
    #        ds_day[s].attrs = ds[inputs.sums[s][1]].attrs
    #        ds_day[s].attrs['description'] = f"Sulfur Oxides (SOX)"

    return ds_day

def day_weights(year,month):
    # calculate no. of weekdays, saturdays, and sundays in given month
    mcal = calendar.monthcalendar(int(year),int(month[-2::])) 
    # calendar indexes monday-sunday as 0-6
    n_weekdays = 0
    for n in range(0,5):
        n_weekdays += sum(1 for x in mcal if x[n] != 0) 
    n_saturdays = sum(1 for x in mcal if x[5] != 0)
    n_sundays = sum(1 for x in mcal if x[6] != 0)
    n_days = np.sum([n_weekdays,n_saturdays,n_sundays])
    # weights, for weighted monthly total
    weights = {
        'weekdy':n_weekdays/n_days,
        'satdy':n_saturdays/n_days,
        'sundy':n_sundays/n_days            
    }
    
    #breakpoint()
    return weights

def monthly_avg(ds_out,avgs,weights):
    # Apply weights to get average daily emissions for the month
    print('----Calculating total emissions for this month by weighting weekday, satdy, sundy totals')
    for v in ds_out.drop_vars(['lat','lon']).variables:
        ds_out[v].values = sum(weights[day]*avgs[day][v].values for day in inputs.days)
        vals_test = [avgs[day][v].values for day in inputs.days]
        weights_test = [weights[day] for day in inputs.days]
        #breakpoint()
        #if (np.nanmean(ds_out[v].values) - np.nanmean(np.average(vals_test,weights=weights_test,axis=0))): 
        #    breakpoint()    # these two weighted averages SHOULD be the same
        #else: 
        #    print(' ()()')
        #    print('\(..)/ Passed math check!')
    return ds_out

def set_metadata(ds):
    # Add some crucial metadata
    ds.attrs['TITLE'] = 'Average hourly GRA2PES emissions for GHG Center' 
    ds.attrs['DESCRIPTION'] = f"Derived from {'_'.join(os.path.basename(inputs.example_fn).split('_')[0:2])} files. Native emissions on 20 vertical levels are summed in Z space to get the total emissions in the column. Native units are in tons or moles km^-2 hour-1, and reported in hourly time steps for weekdays, saturdays, and sundays. Moles, where used, are converted to metric tons for consistency. The average hourly emissions rate for the month is calculated by weighting the hourly and daily emissions accordingly. This number is then multiplied by the number of hours in the month to get an emissions rate in metric tons km^-2 month^-1. Data have also been regridded from native Lambert Conformal projection and 4000m resolution to EPSG 4326 and {inputs.degx} longitude x {inputs.degy} latitude resolution."

    ds.attrs['TIME_AVG'] = '[00:00 - 24:00)Z'
    ds.attrs['GHGC_CONTACT'] = 'eibell@nasa.gov'
    ds.attrs['GRA2PES_CONTACT'] = 'colin.harkins@noaa.gov'
    ds.attrs['FILE_CREATION_DATETIME'] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    
    # Variable-specific
    ds_ex = xr.open_dataset(inputs.example_fn) 

    # Geospatial information
    # This format should be CF-compliant
    for v in ds.data_vars:
        ds[v].attrs['grid_mapping']='crs'
        try:
            del ds[v].attrs['FieldType'],ds[v].attrs['MemoryOrder'],ds[v].attrs['stagger']
        except:
           pass 
    ds = ds.assign(variables={'crs':''})
    ds['crs'].attrs['standard_name'] = 'crs'
    ds['crs'].attrs['grid_mapping_name'] = 'latitude_longitude'
    ds['crs'].attrs['longitude_of_prime_meridian'] = 0.0
    ds['crs'].attrs['semi_major_axis'] = 6378137.0
    ds['crs'].attrs['inverse_flattening'] = 298.257223563
    ds['crs'].attrs['crs_wkt'] = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'

    return ds

def subset(ds):
    # Before any calculations, remove variables we don't want to keep
    #breakpoint()
    print('----Selecting subset of original dataset')
    remove = [v not in inputs.keep for v in ds.variables]
    remove_vars = np.array([v for v in ds.variables])[remove] 
    ds_out = ds.drop_vars(remove_vars)
    # Add up fields to get SOX total 
    # Actually, Brian says just SO2 is fine per the EPA
    #for v in inputs.sums.keys():
    #    sumthis = sum(ds[vv].values for vv in inputs.sums[v])
    #    #ds_out[v] = (tuple(dict(ds.dims).keys()),sumthis)
    #    ds_out = ds_out.assign(variables={v:(ds[inputs.sums[v][1]].dims,sumthis)})

    return ds_out

def set_fill_value(ds):
    for v in ds.data_vars:
        # This is a specific VEDA requirement, per Sid August 2024 - NaN values throw errors
        ds[v].values[np.isnan(ds[v].values)] = -9999
    return ds
    
def aggregate_z_t(ds):
    # Sum across z dimension
    dsnew = ds.sum(dim='bottom_top')
    # Sum across time dimension
    dsnew = dsnew.mean(dim='Time') # this gets us metric tons km^-2 hr^-2
    # Coordinates should now be 2D: [south_north,west_east]
    #breakpoint()
    
    print('----Returning data aggregated in time, z')
    return dsnew


def main():
    grid_RCM = make_grid_RCM(inputs.example_fn)
    grid_out = make_grid_out()
          
    regridder = xe.Regridder(grid_RCM, grid_out, method='conservative', reuse_weights=inputs.reuse_weights,unmapped_to_nan=True)
    print(regridder)
          
    for sector in inputs.sectors:
        
        for year in inputs.years:
    
            for month in inputs.months:
                print(f'\n* MONTH: {month} *\n')
                weights = day_weights(year,month)
                # daily_averages will have the average emissions
                # as an xarray dataset for weekdy, satdy, sundy
                daily_avgs = {}
    
                for d,day in enumerate(inputs.days):
                    print(f'* DAY: {day} *')
    
                    for h,half in enumerate(inputs.halves):
                        filebase = inputs.input_file.replace('[HH]',half)
                        filebase = filebase.replace('[YYYYMM]',datetime(int(year),int(month[-2::]),1).strftime('%Y%m'))
                        filebase = filebase.replace('[DD]',day)
                        filebase = filebase.replace('[SECT]',sector)
                        fn_in = os.path.join(inputs.data_dir_base,year,month,sector,day,filebase)
                        try: 
                            ds_in = xr.open_dataset(fn_in,cache=False,engine='netcdf4')
                        except:
                            continue
                        print(f'* {half}  *')
                        print('===>\n===> Reading '+fn_in)
                        ds_day = reformat(ds_in,regridder)
                        if h == 0:
                            ds_old = ds_day.copy()
                            attrs = ds_in.attrs
                            del ds_day
                        else:
                            print('----Averaging AM + PM')
                            for v in ds_day.drop_vars(['lat','lon']).variables:
                                testt = np.nanmean(np.array([ds_day[v].values,ds_old[v].values]),axis=0) 
                                testt2 = ds_day[v].values.copy()
                                diff = ds_day[v].values - ds_old[v].values
                                testt3 = 0.5*ds_day[v].values + 0.5*ds_old[v].values 
                                ds_day[v].values = 0.5*ds_day[v].values + 0.5*ds_old[v].values
                                #breakpoint()
                                # Units will still be in tons CO2/km2/hour
                                # Double checking...
                                if np.nanmean(ds_day[v].values) != np.nanmean(testt): 
                                    print('Two means are not the same!')
                                    breakpoint() # these two means should get the same result!
                                else: 
                                    print(' ()()')
                                    print('\(..)/ Passed math check!')
                                cond1 = np.nanmean(ds_day[v].values) == np.nanmean(testt2)  # make sure values have changed (e.g., were overwritten correctly)
                                cond3 = np.nanmean(diff) != 0.0  # if the fields going into ds_day[v].values were different, then cond1 should not be true. if both cond1 and cond3 are true, we have a problem. 
                                if cond1 and cond3:
                                    print('Check if AM and PM fields are the same')
                                    breakpoint() # this would mean i'm not overwriting the values properly with the NEW mean
                                else: 
                                    print(' ()()')
                                    print('\(..)/ Passed math check!')
                        del ds_in
                        gc.collect()
                        # End halves loop
    
                    # Last bit of day loop:
                    if 'ds_day' in locals():
                        print(f'----Adding {day} to daily_avgs')
                        daily_avgs[day] = ds_day
                        print(f'Deleting AM + PM for {day}')
                        del ds_day
                        del ds_old
                    else:
                        #print('\n>>>> No halves to total up to a day; continuing.\n')
                        continue
                    #breakpoint()
                    # End days loop
    
                print('----Reached end of weekdy/satdy/sundy loop')
                if len(daily_avgs.keys()) > 0:
                    #breakpoint()
                    ds_month = monthly_avg(daily_avgs['sundy'].copy(),daily_avgs,weights)
                    # still in units of tons CO2/km2/hour
                    ds_month = hourly_to_monthly(ds_month,int(year),int(month[-2::]))
                    # NOW we're in units of tons CO2/km2/month
                    # Replace nans with -9999
                    ds_month = set_fill_value(ds_month)
                    out_fn = inputs.output_fn.replace('[YYYYMM]',datetime(int(year),int(month[-2::]),1).strftime('%Y%m'))
                    out_fn = out_fn.replace('[SECT]',sector)
                    out_fn = os.path.join(inputs.output_dir_base,year,month,out_fn)
                    # if file already exists then delete the file
                    if os.path.exists(out_fn):
                        os.remove(out_fn)
                    ds_month = set_metadata(ds_month)
                    #breakpoint()
                    #save_ncf(ds_month,out_fn,)
                    print('|| Saving to ',out_fn,' ||')
                    ds_month.to_netcdf(out_fn,format='netCDF4',engine='netcdf4',mode='w')
                    del ds_month
                    del daily_avgs
                    gc.collect()
                else:
                    print('\n>>>> No days to average up to a month; continuing.\n')
                    continue
                # End months loop
        # End years loop
    # End sectors loop
                    

# ---------
# MISC.
# ---------
def moles_to_tons(val,s):
    # Convert moles of gas to metric tons of gas.
    # Inputs: Emissions in moles, species name
    # Molecular weights from NIST at 
    # https://webbook.nist.gov/chemistry/form-ser/
    grams_per_mole = {
        'CO': 28.0101,
        'CO2':44.0095,
        'CH4':16.0425,
        'NOX':46.0055,  # Per Colin, just NO2 mass
        'SO2':64.064
    }

    grams_km2_hr = val * grams_per_mole[s]
    # 1000 grams per kilogram
    kg_km2_hr = grams_km2_hr/1000.
    # 1000 kilograms in a metric ton
    tons_km2_hr = kg_km2_hr/1000.
    return tons_km2_hr  

def hourly_to_monthly(ds,year,month):
    # units input will be tons km^-2 hour^-1
    # multiply by hours in the month to get tons km^-2 month^-1
    # (hours in month = days in month * 24.)
    ndays_in_month = monthrange(year,month)[1]
    for v in ds.data_vars:
        ds[v].values = ds[v].values * ndays_in_month * 24.
        ds[v].attrs['units'] = 'metric tons km^-2 month^-1'

    return ds

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
if __name__ == "__main__":
   main() 
