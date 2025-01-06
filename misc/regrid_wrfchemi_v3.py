import xarray as xr
import os
import xesmf as xe
#import ESMF
import pyproj
import numpy as np
import gc

class inputs():
    #example_fn = '/wrk/users/charkins/emissions/V7_GRA2PES/wrfchemi/CONUS4km_append_extra/2021/Month08/satdy/wrfchemi_00z_d01'
    #data_dir_base = '/wrk/users/charkins/emissions/V7_GRA2PES/wrfchemi/CONUS4km_append_extra'
    #output_dir_base = '/wrk/users/charkins/emissions/for_chris_loughner'
    #years = ['2021']
    #months = ['Month08']
    #days = ['satdy','sundy']
    
    example_fn = '/discover/nobackup/projects/gmao/geos_carb/embell/data/GRAAPES/2021/Month08/weekdy/wrfchemi_00z_d01'
    data_dir_base = '/discover/nobackup/projects/gmao/geos_carb/embell/data/GRAAPES'
    output_dir_base = data_dir_base
    years = ['2021']
    months = ['Month01','Month02','Month03','Month03','Month05','Month06','Month07','Month08','Month09','Month10','Month11','Month12']
    days = ['weekdy','satdy','sundy']
    halves = ['00','12']
    input_file = 'wrfchemi_[HH]z_d01'
    output_fn = 'wrfchemi_[HH]z_d01_regrid.nc4'
    
    
    reuse_weights = False
# end inputs class

def make_grid_RCM(domain_example_fn):
    out_example = xr.open_dataset(domain_example_fn, chunks={'Time': 1}, engine='netcdf4')
    wrf_proj = pyproj.Proj(proj='lcc', # projection type: Lambert Conformal Conic
                    lat_1=out_example.TRUELAT1, lat_2=out_example.TRUELAT2, # Cone intersects with the sphere 33,45
                    lat_0=out_example.MOAD_CEN_LAT, lon_0=out_example.STAND_LON, # Center point :MOAD_CEN_LAT = 39.34594f ; :STAND_LON = -97.f ;
                    a=6370000, b=6370000) # Radius#a=6378137, b=6378137) # Radius
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
    
    degx, degy = 0.01, 0.01
    
    lon = np.arange(-77.795,-76.205,degx)
    lat = np.arange(38.405,39.605,degy)
    
    nx = len(lon)
    ny = len(lat)
    
    lon_b = np.zeros([nx+1])
    lon_b[0:nx] = lon-degx/2
    lon_b[nx] = lon[nx-1]+degx/2
    
    lat_b = np.zeros([ny+1])
    lat_b[0:ny] = lat-degy/2
    lat_b[ny] = lat[ny-1]+degy/2
    
    
    grid =  {'lat': lat, #Center Point Spacing Lat
                    'lon': lon, #Center Point Spacing Lon
                    'lat_b': lat_b, # Boundary Spacing Lat
                    'lon_b': lon_b, # Boundary Spacing Lon
                    }
                    
    return grid
    
def save_ncf(ds,out_fn,data_vars,mode='w'):
    
    
    
    ncf_vars_all = data_vars
    for var in ncf_vars_all:
        if var != 'Times':
            # Set all encoding settings properly
            encoding_dict = {'dtype': 'float32', 'chunksizes':(1,1, ds.sizes['lat'], ds.sizes['lon']),
              'zlib': True, 'complevel': 1, '_FillValue': None }
            ds[var].encoding=encoding_dict
        else:
            ds['Times'].encoding={'char_dim_name':'DateStrLen'}
    
    #print('Writing file: ', out_fn)
    ds.to_netcdf(out_fn,format='netCDF4',engine='netcdf4',mode=mode)
    
# end save_ncf

def add_var(data,regridder,ds):
    data.append(regridder(ds))
    

def main():
    grid_RCM = make_grid_RCM(inputs.example_fn)
    grid_out = make_grid_out()
    
    regridder = xe.Regridder(grid_RCM, grid_out, method='conservative', reuse_weights=inputs.reuse_weights)
    print(regridder)
    
    for year in inputs.years:
        for month in inputs.months:
            for day in inputs.days:
                for half in inputs.halves:
                    
                    fn_in = os.path.join(inputs.data_dir_base,year,month,day,inputs.input_file.replace('[HH]',half))
                   
                    try: 
                        ds_in = xr.open_dataset(fn_in,cache=False,engine='netcdf4')
                    except:
                        print('')
                        print('>> ',fn_in,' not found. Skipping.')
                        continue
                    
                    # %% Renaming after regridder to match wrfchemi dimentions and coordinate names
                    rename_dict = {'south_north':'y',
                                    'west_east':'x',
                                    'XLAT':'lat',
                                    'XLONG':'lon'
                                    }
                    ds_in = ds_in.rename(rename_dict)
                    ds_in = ds_in.set_coords(names=('lat','lon'))
                    ds_in = ds_in.transpose('Time','emissions_zdim','y','x')
                    
                    #print(ds_in)
                    
                    
                    out_fn = os.path.join(inputs.output_dir_base,year,month,day,'regrid',inputs.output_fn.replace('[HH]',half))
                    # if file already exists then delete the file
                    if os.path.exists(out_fn):
                        os.remove(out_fn)
                    
                    for c, var in enumerate(ds_in.drop('Times').data_vars):
                        
                        ds_out = regridder(ds_in[[var]])
                        ds_out[var].attrs = ds_in[var].attrs
                        ds_out.attrs = {}
                        del ds_out[var].attrs['MemoryOrder'],ds_out[var].attrs['grid_mapping'],ds_out[var].attrs['FieldType']
                        
                        if c == 0:
                            print('Creating File: ', out_fn)
                            print('Appending '+var)
                            save_ncf(ds_out,out_fn,data_vars=ds_out.data_vars,mode='w')
                        else:
                            print('Appending '+var)
                            save_ncf(ds_out,out_fn,data_vars=ds_out.data_vars,mode='a')
                        #breakpoint()    
                        del ds_out
                        gc.collect()
                    
                    
                    ds_out = ds_in[['Times']]
                    print('Appending '+'Times')
                    save_ncf(ds_out,out_fn,data_vars=['Times'],mode='a')
                    print('Finish appending')
                    
                    del ds_in
                    gc.collect()

# End Main

if __name__ == "__main__":
    main()

    



    

