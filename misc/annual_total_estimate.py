import xarray as xr
import rioxarray as rxr
import numpy as np
import glob
import datetime
from pyproj import Transformer


def surfaceAreaGrid(**kwargs):
    # There can be different ways of specifying a rectangular lat/lon grid, assume always degrees not radians
    if 'lat_edges' in kwargs: # boundaries, do not assume uniform spacing, but assume monotonically increasing
        lat_edges = (np.pi/180.) * kwargs['lat_edges']
    elif 'lat_centers' in kwargs: # centers, assume uniform spacing and monotonically increasing
        lat_centers = kwargs['lat_centers']
        dlat = np.diff(lat_centers).mean()
        lat_edges = np.zeros(len(lat_centers)+1, dtype=np.float64)
        lat_edges[0] = lat_centers[0]-0.5*dlat
        lat_edges[1:] = lat_centers + 0.5*dlat
        lat_edges = (np.pi/180.0) * lat_edges
    else:
        lat_min = kwargs['lat_min'] if 'lat_min' in kwargs else -90.0
        lat_max = kwargs['lat_max'] if 'lat_max' in kwargs else 90.0
        try:
            nlat = kwargs['nlat']
        except:
            print('At least specify the number of divisions of latitude')
            raise
        lat_edges = (np.pi/180.) * np.linspace(lat_min, lat_max, nlat+1)

    # now the longitude grid
    if 'lon_edges' in kwargs: # boundaries, do not assume uniform spacing, but assume monotonically increasing
        lon_edges = (np.pi/180.) * kwargs['lon_edges']
    elif 'lon_centers' in kwargs: # centers, assume uniform spacing and monotonically increasing
        lon_centers = kwargs['lon_centers']
        dlon = np.diff(lon_centers).mean()
        lon_edges = np.zeros(len(lon_centers)+1, dtype=np.float64)
        lon_edges[0] = lon_centers[0]-0.5*dlon
        lon_edges[1:] = lon_centers + 0.5*dlon
        lon_edges = (np.pi/180.0) * lon_edges
    else:
        lon_min = kwargs['lon_min'] if 'lon_min' in kwargs else -180.0
        lon_max = kwargs['lon_max'] if 'lon_max' in kwargs else 180.0
        try:
            nlon = kwargs['nlon']
        except:
            print('At least specify the number of divisions of longitude')
            raise
        lon_edges = (np.pi/180.) * np.linspace(lon_min, lon_max, nlon+1)

    assert np.all(np.diff(lat_edges) > 0.0), "Latitude edges must be monotonically increasing"
    assert np.all(np.diff(lon_edges) > 0.0), "Longitude edges must be monotonically increasing"
    # Construct empty array of the right size that will later hold the grid areas
    # We'll construct an array of dLon * sin(lat) for each latitude, then take the difference between
    # successive rows later. Hence now the array will have one extra row.
    dS = np.zeros((len(lat_edges), len(lon_edges)-1), np.float64)
    dlon = np.diff(lon_edges) # a vector, could all be identical if lon spacing is uniform
    for i, lat in enumerate(lat_edges):
        dS[i] = dlon * np.sin(lat)
    dS = np.diff(dS, axis=0)
    
    ret_area = kwargs['ret_area'] if 'ret_area' in kwargs else True # whether to return surface are or solid angle
    if ret_area:
        R_e = 6371000.0 # Radius of Earth in meters, as used by TM5 and GEOS
        dS = R_e * R_e * dS

    return dS

# GRA2PES first
data_dir = '/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES/2021/'
months = ['Month'+datetime.datetime(1993,m,1).strftime('%m') for m in np.arange(1,13)]

monthly_totals = []
for month in months:
    f =glob.glob(data_dir+month+'/GRA2PESv1.0_total_*_subset_regrid.nc4')
    data = xr.open_dataset(f[0])
    co2 = data['CO2'].values
    wnn = co2 == -9999
    co2[wnn] = 0.0
    # grid cell areas in m2  
    dS = surfaceAreaGrid(lat_centers=data['lat'].values,lon_centers=data['lon'].values)
    dS = dS/1000./1000.     # in km2
    # convert co2 from tons CO2/km2/month to Tg C/km2/month
    # first, tons CO2 -> TgC
    co2 = (co2 * 1000.*1000.)   # g CO2
    co2 = co2/1e12              # Tg CO2
    co2 = co2 * (12./44.)        # Tg C
    monthly_totals.append(np.sum(dS*co2))    #TgC/km2/month * km2 for each grid cell, then sum: TgC/month!
    if month == 'Month01':
        co2_total = co2
    else:
        co2_total = co2 + co2_total
annual_total = np.sum(np.array(monthly_totals))

print('GRA2PES')
print(monthly_totals)
print(annual_total)

# Vulcan next
#vf = glob.glob('/discover/nobackup/projects/gmao/geos_carb/embell/data/Vulcan/v4.0/grid.1km.mn/TOT.CO2/*2021.tif')
#vdata = rxr.open_rasterio(vf[0]) 
#vco2 = vdata.values
#vco2 = vco2*(44./12.)
#transformer = Transformer.from_crs(vdata.rio.crs,'EPSG:4326')
#print(transformer.transform(vdata.x.values[0],vdata.y.values[0]))
## grid cell areas in km2
#vcellarea = 111.1 * 111.1 * 0.5 * 0.5 * np.cos(np.radians(vdata.y)) 
#vcellarea = np.tile(vcellarea, (len(vdata.x), 1)).T
#vannual_total = np.sum(vcellarea*vco2)
#
#print('Vulcan')
#print(vannual_total)



