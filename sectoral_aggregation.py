import xarray as xr
import numpy as np
import datetime
import glob
import os
"""
This script adds together emissions from similar GRA2PES sectors to get a total emissions rate for an "aggregated" sector. 
<inputs.sums> contains the list of sectors being added to form the aggregates.
Inputs are hourly files 00to11Z and 12to23Z; outputs are the same.
Aggregation up to monthly emissions happens later, in aggegrate_monthly_gra2pes.py
"""

class inputs:
    sums ={
        'Nonroad':['OFFROAD','RAIL','AVIATION','SHIPPING'],
        'Residential_Commercial':['RES','COMM','COOKING','WASTE','VCP'],
        'Other':['AG','FUG']
        }
    #sums = {'Test':['Industrial','EGU','Onroad']}
    species = ['CO','CO2','NOX','SO2','PM25-PRI','PM02']
    days = ['weekdy','satdy','sundy']
    halves = ['00to11Z','12to23Z']
    data_dir = '/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES'
    years = [2021]
    months = ['Month'+datetime.datetime(1993,m,1).strftime('%m') for m in np.arange(1,13)]
    version = 'GRA2PESv1.0'

def check_data_vars(ds):
    if len(ds.data_vars) > len(inputs.species):
        remove = [not v in inputs.species for v in ds.data_vars]
        vv = [v for v in ds.data_vars] 
        ds = ds.drop_vars(np.array(vv)[remove])
    return ds

def aggregate_sectors(inputs):
    for year in inputs.years:
        for month in inputs.months:
            for day in inputs.days:
                for half in inputs.halves:
                    for s in inputs.sums.keys():
                        sector_vals = []
                        data_agg = {}
                        dims_dict = {}
                        data_coords = {}
                        attrs = {}
                        vals = {}
                        dims = {}
                        components = {}
                        dat_agg = {}
                        for i,sector in enumerate(inputs.sums[s]):
                            ss = f"{inputs.data_dir}/{year}/{month}/{sector}/{day}/{inputs.version}_{sector}_{datetime.datetime(year,int(month[-2::]),1).strftime('%Y%m')}_{day}_{half}.nc"
                            fn = glob.glob(ss)
                            print(fn)
                            print(ss)
                            if len(fn) == 0:
                                print('glob.glob did not find a match!')
                                breakpoint()
                            data = xr.open_dataset(fn[0])
                                
                            #==== 
                            #====== dictionaries -> new Dataset approach
                            for v in inputs.species:
                                if i == 0:
                                    # start with first sector values
                                    # variable-specific
                                    vals[v] = data[v].values
                                    dims[v] = data[v].dims
                                    attrs[v] = data[v].attrs
                                else:
                                    # add on the next sector values
                                    vals[v] = vals[v] + data[v].values
   
                                if i == len(inputs.sums[s])-1:
                                    data_agg[v] = (dims[v],vals[v])
   
                                if v == inputs.species[-1] and i == len(inputs.sums[s])-1 :
                                    del dims
                                    del vals
                                    #breakpoint()
                                    #diff = vals[v] - np.sum(np.array(components[v]),axis=0) 
                                    #if np.mean(diff) != 0:
                                    #    print('*** Check your arithmetic!')
                                    #    breakpoint()
                                    #else:
                                    #    print('Great sum well done!!!')
                                    # global attributes
                                    dataset_dims = data.dims
                                    data_coords = data.coords
                                    data_attrs = data.attrs
                                    
                                    #breakpoint()
                            #====== dictionaries -> new Dataset approach
                            #==== 
                            data.close()
                            
#                            if i == 0:
#                                data = xr.open_dataset(fn[0]) 
#                                print(data.CO2.values.mean())
#                                sector_vals.append(data.CO2.values)
#                            else:
#                                data_new = xr.open_dataset(fn[0]) 
#                                sector_vals.append(data_new.CO2.values)
#                                print(data_new.CO2.values.mean())
                                #if len(data.data_vars) != len(data_new.data_vars):
                                #    print('Double check file versions!')
                                #    print('file1 data_vars:',[v for v in data.data_vars])
                               #     print('file2 data_vars:',[v for v in data_new.data_vars])
                                #    breakpoint()

                        # Calculate sum another way, make sure they're the same
                        #sector_vals = np.array(sector_vals)
                        #summ = np.sum(sector_vals,axis=0)
                        #del sector_vals
                        #diff = data.CO2.values - summ
                        #del summ
                        #if np.mean(diff) != 0:
                        #    print('Double check your sums!!')
                        #    breakpoint()
                        #else:
                        #    print('\n  ()()')
                        #    print('\ (..) / Passed math check!\n')
                        #==== 
                        #====== dictionaries -> new Dataset approach
                        data_out = xr.Dataset(data_agg,data_coords)
                        for v in data_out.data_vars:
                            data_out[v].attrs = attrs[v]
                        data_out.attrs = data_attrs
                        #====== dictionaries -> new Dataset approach
                        #==== 
#                        data = data.sum(dim='sector')  #====== xarray concat approach
                        sn = fn[0].replace(sector,s)
                        save_dir = os.path.dirname(sn) 
                        if not os.path.exists(save_dir):
                            print('Creating directory ',save_dir)
                            os.makedirs(save_dir)
                        print('  ===> ',sn)
                        #breakpoint()
                        data_out.to_netcdf(sn,format='netCDF4',engine='netcdf4',mode='w')
                        print(f'Aggregate {s} saved.')
                        del data_out
                        del data_agg
        #breakpoint()

if __name__ == '__main__':
    aggregate_sectors(inputs)
