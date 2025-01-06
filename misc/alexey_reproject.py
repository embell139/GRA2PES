import xarray as xr
import pyproj
import rioxarray
import xesmf as xe
import numpy as np

g2path = "/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES/2021/Month08/weekdy/wrfchemi_00z_d01"
                                                                                                                                     
dat = xr.open_dataset(g2path, engine="h5netcdf")

# The code below doesn't actually need these
# wrf_proj = pyproj.Proj(proj='lcc', # projection type: Lambert Conformal Conic
#                         lat_1=dat.TRUELAT1, lat_2=dat.TRUELAT2, # Cone intersects with the sphere 33,45
#                         lat_0=dat.MOAD_CEN_LAT, lon_0=dat.STAND_LON, # Center point :MOAD_CEN_LAT = 39.34594f ; :STAND_LON = -97.f ;
#                         a=6370000, b=6370000) # Radius#a=6378137, b=6378137) # Radius
                                                                                                                                     
# keep = ['E_CO','E_CO2','emissions_dim','Time','west_east','south_north', 'XLAT', 'XLONG']
# dropthese = [v for v in dat.variables if v not in keep]
# ddrop = dat.drop_vars(dropthese)

# A simpler way to do this:
dmean = dat[["E_CO", "E_CO2"]].mean(dim=('emissions_zdim', 'Time'))

# Rename coordinates so that xesmf understands them.
din = dmean.rename({"XLAT": "lat", "XLONG": "lon"})

# Create a custom grid around the CONUS bbox
# NOTE: Double check this --- I just pulled it randomly off the internet...
conus_bbox = (-124.848974, 24.396308, -66.885444, 49.384358)

# Output grid onto which you want to project the results.
# NOTE: This uses a 1.0 x 1.0 degree grid for a quick test. You'll probably 
# want a finer resolution grid here (maybe 0.025 x 0.025)
dout = xr.Dataset({
    "lat": (["lat"], np.arange(conus_bbox[1], conus_bbox[3], 1.0), {"units": "degrees_north"}),
    "lon": (["lon"], np.arange(conus_bbox[0], conus_bbox[2], 1.0), {"units": "degrees_east"})
    })

# Create the regridder object. This is a relatively time- and compute-intensive 
# step, but then the regridder object can be reused multiple times! In case you 
# want to add variables, etc. You may even be able to store it to disk somehow 
# (maybe with python pickle). Check the xESMF documentation.
regrid = xe.Regridder(din, dout, "bilinear")

# Now, actually do the regridding.
result = regrid(din)

# Assign the rio raster metadata.
result.rio.set_spatial_dims("lon", "lat", inplace=True)
result.rio.write_crs(4326 ,inplace=True)

# Write the results. Note: Need to specify NETCDF4 here because it sometimes 
# defaults to NetCDF-classic w/ 64-bit offset (because that's the scipy 
# default).
result.to_netcdf("test.nc", format="NETCDF4")

# Or, write directly to a GeoTIFF. May need to fiddle with the creation options 
# here to make it a COG.
result.rio.to_raster("test.tif")

########################################
# Now, test reading these.

dtest_nc = xr.open_dataset("test.nc", engine="h5netcdf")
# Spatial information is embedded in the spatial_ref "pseudo-variable" (per CF grid standards)
print(dtest_nc.spatial_ref.attrs["crs_wkt"])

# Dataset will also open in rioxarray, but with a more raster-y structure (note: x, y variables; bands)
# ...but rio attributes are automatically set.
dtest_rxr = xr.open_dataset("test.nc", engine="rasterio")
print(dtest_rxr.rio.crs)
