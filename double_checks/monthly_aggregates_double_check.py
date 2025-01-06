import numpy as np
import xarray as xr
import glob
import os
import datetime
import calendar
from calendar import monthrange
import xesmf as xe
import pyproj
from matplotlib import pyplot as plt

class inputs:
    #sectors =['Nonroad','Residential_Commercial','Other','EGU','Onroad','total','Industrial']
    dataset='GRA2PES'
    sectors = ['total']
    species = ['CO','CO2','NOX','SO2','PM25-PRI']
    days = ['weekdy','satdy','sundy']
    halves = ['00to11Z','12to23Z']
    data_dir = f'/discover/nobackup/projects/gmao/geos_carb/embell/data/{dataset}'
    img_dir = '/discover/nobackup/projects/gmao/geos_carb/embell/images/spot_check/'
    years = [2021]    
    months = ['Month'+datetime.datetime(1993,m,1).strftime('%m') for m in np.arange(1,13)]
    version = 'GRA2PESv1.0'
    reuse_weights = False
    degx, degy = 0.036, 0.036
    example_fn = f'/discover/nobackup/projects/gmao/geos_carb/embell/data/{dataset}/2021/Month08/total/weekdy/GRA2PESv1.0_total_202108_weekdy_00to11Z.nc' 

def check_file(search_string):
    if len(glob.glob(search_string)) == 0:
        print('Issue finding file')
        print(search_string)
        breakpoint()
    return 

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

def get_weights(year,month):
    mcal = calendar.monthcalendar(year,month)
    ndays = monthrange(year,month)[1]
    n_weekdays = 0
    for n in range(0,5):
        n_weekdays += sum(1 for x in mcal if x[n] != 0) 
    satw= sum(1 for x in mcal if x[5] != 0)/ndays
    sunw = sum(1 for x in mcal if x[6] != 0)/ndays
    weekw = n_weekdays/ndays
    weights = [weekw,satw,sunw]
    return weights

def check_aggs(n):                
    i = 0                         
    while i <= n:                 
        # generate random numbers for each input
        m = np.random.randint(0,high=len(inputs.months))
        s = np.random.randint(0,high=len(inputs.species))
        sec = np.random.randint(0,high=len(inputs.sectors))
        y = np.random.randint(0,high=len(inputs.years))
        #breakpoint()             
        # use those random numbers to identify which slice to validate
        year = inputs.years[y] 
        month = inputs.months[m]
        speciess = inputs.species[s]
        sector = inputs.sectors[sec] 
                                  
        # select the file your random numbers have chosen
        fn = f"{inputs.data_dir}/{year}/{month}/*{sector}*subset_regrid.nc4"
        check_file(fn)
        print(f'{sector}')       
        print('Opening aggregated file')
        print(glob.glob(fn)[0])
        data_agg = xr.open_dataset(glob.glob(fn)[0])
        agg_vals = data_agg[speciess].values    # remember this is regridded to EPSG 4326
        agg_vals = np.where(agg_vals==-9999,np.nan,agg_vals)

        #breakpoint()
        # Now we'll do our aggregation another way, to compare to the above agg_vals
        print('Opening component files')
        components = []
        for day in inputs.days:
            halves = []
            for half in inputs.halves: 
                hfn = f"{os.path.dirname(fn)}/{sector}/{day}/*{half}*.nc"
                check_file(hfn)
                
                print(glob.glob(hfn)[0])
                hdata = xr.open_dataset(glob.glob(hfn)[0])
                hdata = hdata[speciess] 
                halves.append(hdata.sum(dim='bottom_top').mean(dim='Time').values) 
            components.append(0.5*halves[0] + 0.5*halves[1])

        print('Calculating day weights')
        day_weights = get_weights(year,int(month[-2::]))
        print('Getting weighted average')
        wavg = [w*c for w,c in zip(day_weights,components)]
        wavg = np.sum(np.array(wavg),axis=0)
        # moles per km2 per hour to metric tons per km2 per hour
        wavg = moles_to_tons(wavg,speciess)
        # metric tons per km2 per hour to metric tons per km2 per month
        ndays_in_month = monthrange(year,int(month[-2::]))[1]
        wavg = wavg * ndays_in_month * 24.

        # Let's regrid our new estimate
        grid_RCM = make_grid_RCM(inputs.example_fn)
        grid_out = make_grid_out()

        regridder = xe.Regridder(grid_RCM, grid_out, method='conservative', reuse_weights=inputs.reuse_weights,unmapped_to_nan=True)
        wavg_rg = regridder(wavg)

    
        diff = agg_vals - wavg_rg
        pdiff = np.nanmean(diff)/np.nanmean(agg_vals)*100.
        if abs(pdiff) >= 0.025:
            print('Check your arithmetic!')
            breakpoint()
        else:   
            print(i,' // Great sum well done!')
            print('=============\n')

        # MAPS
        # map of diff
#        plt.pcolormesh(data_agg.lon,data_agg.lat,diff,vmin=np.nanmin(diff),vmax=np.nanmax(diff))
#        plt.colorbar()
#        plt.savefig(inputs.img_dir+'diff.png')
#        plt.close()
#        # map of original monthly agg
#        plt.pcolormesh(data_agg.lon,data_agg.lat,agg_vals,vmin=np.nanmin(agg_vals),vmax=np.nanmax(agg_vals))
#        plt.colorbar()
#        plt.savefig(inputs.img_dir+'data_agg.png')
#        plt.close()
#        # map of validation monthly agg 
#        plt.pcolormesh(data_agg.lon,data_agg.lat,wavg_rg,vmin=np.nanmin(agg_vals),vmax=np.nanmax(agg_vals))
#        plt.colorbar()
#        plt.savefig(inputs.img_dir+'wavg_rg.png')
#        plt.close()
                
        i+=1    
        #breakpoint() 

if __name__ == '__main__':
    check_aggs(5) 


