import xarray as xr
import pyproj
import rioxarray
from salem import open_wrf_dataset
                                                                                                                                     
g2path = "/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES/2021/Month08/weekdy/wrfchemi_00z_d01"
                                                                                                                                     
#dat = xr.open_dataset(g2path, engine="netcdf4")
dat = open_wrf_dataset(g2path)
                                                                                                                                     
wrf_proj = pyproj.Proj(proj='lcc', # projection type: Lambert Conformal Conic
                        lat_1=dat.TRUELAT1, lat_2=dat.TRUELAT2, # Cone intersects with the sphere 33,45
                        lat_0=dat.MOAD_CEN_LAT, lon_0=dat.STAND_LON, # Center point :MOAD_CEN_LAT = 39.34594f ; :STAND_LON = -97.f ;
                        a=6370000, b=6370000) # Radius#a=6378137, b=6378137) # Radius
                                                                                                                                     
keep = ['E_CO','E_CO2','lat','lon','emissions_dim','time','west_east','south_north']
dropthese = [v for v in dat.variables if v not in keep]
dat = dat.drop_vars(dropthese)

dat = dat.mean(dim='emissions_zdim') 
dat = dat.mean(dim='time')
#dat.rio.set_spatial_dims("west_east", "south_north", inplace=True)
dat = dat.rename({'west_east':'x','south_north':'y'})
dat.rio.set_spatial_dims('y','x',inplace=True)
dat.rio.write_crs(wrf_proj.crs, inplace=True)

d2 = dat.rio.reproject("EPSG:4326")
#FIXME: "ValueError: coordinate lon has dimensions ('south_north', 'west_east'), but these are not a subset of the DataArray dimensions ('y', 'x')"
