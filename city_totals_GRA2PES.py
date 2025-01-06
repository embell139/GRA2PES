import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import xarray as xr
import datetime
import numpy as np
import geopandas as gpd 
import pandas as pd
import json
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import glob
import rioxarray as rxr
from shapely.geometry import mapping
from cartopy import crs as ccrs
import cartopy.feature as cf
import contextily as cx
import os 
import pathlib

"""
Input files required:
    - JSON list of cities including PLACENS code for Census Place designations
    - Census Place .gpkg file containing city shapes and other metadata
    - GRA2PES aggregated monthly sectoral emissions files

Outputs:
    - One CSV per city/year/month combo, rows are species and columns are sectors 
    - Maps of the city boundary shapes overlaid on GRA2PES data, (1) unclipped to shape, (2) clipped to shape. 
        - Only when keyword output_figures = True

Author: Emily Bell
Contact: emily.i.bell@nasa.gov
"""

class inputs:
    cities_file = '/discover/nobackup/projects/gmao/geos_carb/embell/ghgc/city_lists/urban_dashboard.json'
    shapefile = '/discover/nobackup/projects/gmao/geos_carb/embell/data/Vulcan/shapefiles/ACS_2021_5YR_PLACE_VulcPrj_Pop.gpkg'
    gra2pes_dir= '/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES'

    years = [2021]
    months = ['Month'+datetime.datetime(1993,m,1).strftime('%m') for  m in np.arange(1,13)]
    version = 'GRA2PESv1.0'
    sectors = ['total','Industrial','Onroad','EGU','Nonroad','Other','Residential_Commercial']
    sectors_long = ['Total','Industrial','Onroad Transportation','Power','Nonroad Transportation','Other','Residential + Commercial']
    species = ['CO2','CO','SO2','NOX','PM25-PRI']
    species_long = ['Carbon Dioxide','Carbon Monoxide','Sulphur Dioxide','Nitrogen Oxides', 'Particulate Matter (PM2.5)']

    image_dir = '/discover/nobackup/projects/gmao/geos_carb/embell/images/GRA2PES/city_subsets'
    out_dir = '/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES/for_ghgc/city_totals'

    with open(cities_file,'r') as infile:
        cities = json.load(infile)

    id_key = 'PLACENS'
    shape_name = 'PLACE'    # which US Census definition of the city boundaries we're using
    all_touched = False     # if True, all grid cells even TOUCHING the city boundary will be included in the total.
                            # if False, only grid cells whose center are within the city boundary will be included.
                            # See rioxarray.clip documentation for details: 
                            # https://corteva.github.io/rioxarray/html/rioxarray.html

# Apply appropriate CRS information to existing city polygons 
# ---
def get_shapes(inputs):
    gdf = gpd.read_file(inputs.shapefile)
    ids = [inputs.cities[c][inputs.id_key] for c in inputs.cities.keys()]
    wcities = [i in ids for i in gdf[inputs.id_key]]
    gdf = gdf[wcities].to_crs('epsg:4326')

    return gdf

# Create directory if it doesn't exist yet
# ---
def check_dir(fn):
    ddir = os.path.dirname(fn)
    if not os.path.exists(ddir):
        print('Creating directory ',ddir)
        pathlib.Path(ddir).mkdir(parents=True) 

    return
       
# This function from Sourish Basu, for calcaluting surface area of lat/lon grid cells.
# Returns area in meters, in dimensions matching the specifications of the original grid
# (depending on whether grid cell edges or centers were provided).
# ---
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

# Produce map with city boundary shape overlaid on top of pcolormesh of data field 
# ---
def save_map_shape(gdf,data,sn,**map_keys):
    fig = plt.figure(figsize=(8,8)) 
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    gdf.plot(ax=ax,color='black',transform=ccrs.PlateCarree(),alpha=0.3)
    cx.add_basemap(ax,crs=ccrs.PlateCarree(),source=cx.providers.CartoDB.Positron)
    ax.coastlines(lw=0.25) 
    ax.add_feature(cf.BORDERS,lw=0.25)
    ax.add_feature(cf.STATES,lw=0.25)
    #vmax = np.mean(data.read()[data.read() != -9999]) + 5*np.std(data.read()[data.read() != -9999])
    data.plot(ax=ax,alpha=0.5,transform=ccrs.PlateCarree(),**map_keys) 
    check_dir(sn)
    plt.savefig(sn)
    #print('Saving ',sn)
    plt.close('all')

    return

# Loop through sectors, years, months, cities, species 
# to calculate city totals. Return all as an xarray Dataset.
# Optional: 
# output_figure=True 
# outputs two maps, of the city boundary overlaid on
# (1) the unclipped underlying data, and
# (2) the clipped underlying data.
# ---
def get_gra2pes_city_totals(inputs,output_figures=None):
    print('Calculating city totals...')
    # first, city geodataframe
    gdf = get_shapes(inputs)
    sector_yy_mm_species_cities = []

    for sector in inputs.sectors:
        #if sector != 'total':
        #    continue
        print('====')
        print(sector)
        print('====')
        yy_mm_species_cities = []

        for year in inputs.years:
            monthly_species_cities = []

            for month in inputs.months:
                fn = f"{inputs.gra2pes_dir}/{year}/{month}/{inputs.version}_{sector}_{datetime.datetime(year,int(month[-2::]),1).strftime('%Y%m')}_subset_regrid.nc4"
                gra2pes_files = glob.glob(fn)
                data = xr.open_dataset(fn)
                # A little bit of extra formatting for rioxarray to work
                data = data.rename({'lon':'x','lat':'y'})
                data = data.rio.write_crs(data.crs.crs_wkt)
                species_cities = []

                for city in inputs.cities.keys():
                    #if city != 'Houston':
                    #    continue
                    print(city)
                    city_vector = gdf[gdf[inputs.id_key] == inputs.cities[city][inputs.id_key]].to_crs(data.crs.crs_wkt)
                    clipped = data.rio.clip(geometries=city_vector.geometry.values,all_touched=inputs.all_touched)
                    breakpoint()

                    # Area weight the data!
                    dS = np.flip(surfaceAreaGrid(lat_centers=clipped.y.values,lon_centers=clipped.x.values,ret_area=True))
                    # returns area in m2 - convert to km2
                    dS = dS/1000./1000.
                    species_totals = []

                    for s in inputs.species:
                        #if s != 'CO2':
                        #    continue
                        # Multiply our emissions - tons per km2 per month - by km2 of each cell area
                        # which gets us tons per month, and sum the cells for tons per month over the city!
                        weighted_sum = np.nansum(dS*clipped[s].values)
        
                        species_totals.append(weighted_sum)

                        # output maps of (1) the city shape with unclipped GRA2PES data underneath 
                        # (2) the clipped subset of <data> corresponding to the shape
                        if output_figures:
                            map_keys = {
                                "vmin":0,
                                "vmax":np.nanmax(clipped[s].values)
                            }
                            sn1 = f"{inputs.image_dir}/GRA2PES_{month}_{city.replace(' ','_')}_{inputs.shape_name}_{sector}_{s}_overlay.png"
                            sn2 = sn1.replace('overlay','subset')
                            if not inputs.all_touched:
                                sn1 = sn1.replace('.png','_conservative.png')
                                sn2 = sn2.replace('.png','_conservative.png')
                            save_map_shape(city_vector,data[s],sn1,**map_keys)
                            save_map_shape(city_vector,clipped[s],sn2,**map_keys)

                    species_cities.append(species_totals)
                monthly_species_cities.append(species_cities)
            yy_mm_species_cities.append(monthly_species_cities)
        sector_yy_mm_species_cities.append(yy_mm_species_cities)                        
    sector_yy_mm_species_cities = np.array(sector_yy_mm_species_cities)

    print('Returning city totals as xarray Dataset.')
    city_totals = xr.Dataset(
        # vars
        {'metric_tons_per_month':
            (['sector','year','month','city','species'],sector_yy_mm_species_cities)
        },
        # dims
        {'sector':
            ('sector',inputs.sectors_long),
        'year':
            ('year',inputs.years),
        'month':
            ('month',inputs.months),
        'city':
            ('city',list(inputs.cities.keys())),
        'species':
            ('species',inputs.species_long)
        }
    ) 

    return city_totals

# Check the sum of the different sectors against the reported Total.
# Note that as of 1 October 2024, the reported Total is not limited to CONUS,
# so for cities along country borders, the reported Total is greater than the sectoral sum.
# This is expected behavior.
# Otherwise, the max difference printed below for each city/month/year 
# should be fractions of a percent at most!
# ---
def check_totals(data):
    for i,city in enumerate(data.city.values):
        print('=======')
        print(f"{i+1}. {city}")
        print('=======')
        for year in data.year.values:
            print(year)
            for month in data.month.values:
                data_this = data.sel(city=city,year=year,month=month)
                total = data_this.sel(sector='Total').metric_tons_per_month.values
                components = []
                for s in data.sector[1::].values:
                        components.append(data_this.sel(sector=s).metric_tons_per_month)
                components = np.array(components)
                components_sum = np.sum(components,axis=0)
                diff = ((total - components_sum)/total)*100.
                print(f'{month} Maximum % diff:',np.max(diff))
            #breakpoint()
                    
# Step through xarray Dataset to output one CSV per city per month per year.
# ---
def output_CSVs(data):
    print('Converting city totals to pandas DataFrame, calculating sectors as percent of total.')
    # I don't think we need the percents in these CSVs - check with Slesa
    for city in data.city.values:
        print(city)
        for year in data.year.values:
            print(year)
            for month in data.month.values:
                print(month)
                vals = data.sel(city=city,year=year,month=month).metric_tons_per_month.T 
                df = pd.DataFrame(vals,columns=data.sector.values)
                df['Species'] = data.species 

                # Reorder columns so Species is first
                #breakpoint()
                df = df[['Species',*data.sector.values]]

                # Calculate another total to see if it's the same as the originally reported total
                df['total_check'] = df.sum(axis=1,numeric_only=True)
                # Now calculate sector emissions as percentage of total
                # And rename the original fields as emissions by mass
                for k in df.drop(['Total','Species','total_check'],axis=1).keys():
                    df[k+' Percent of Total'] = (df[k]/df['total_check'])*100.
                df = df.drop(['Total','total_check'],axis=1)

                sn = f"{inputs.out_dir}/{city.replace('/','_').replace(' ','_')}_{year}_{month}_species_sectoral_breakdown.csv"
                if not inputs.all_touched:
                    sn = sn.replace('.csv','_conservative.csv')
                print('Saving ',sn)
                df.to_csv(sn)

                
# ---
if __name__ == '__main__':
    city_totals = get_gra2pes_city_totals(inputs,output_figures=False)
    #city_totals = xr.open_dataset('city_totals_test.nc4')
    check_totals(city_totals)
    #breakpoint()
    output_CSVs(city_totals)
