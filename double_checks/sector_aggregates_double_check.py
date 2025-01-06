import numpy as np
import xarray as xr
import glob as glob
import os
import datetime
"""
Double checking math, that the sum of individual sectors matches the outputs for each aggregated sector.
"""

class inputs:
    sums ={            
        'Nonroad':['OFFROAD','RAIL','AVIATION','SHIPPING'],
        'Residential_Commercial':['RES','COMM','COOKING','WASTE','VCP'],
        'Other':['AG','FUG'],
        }              
    #sums = {'Test':['Industrial','EGU','Onroad']}
    species = ['CO','CO2','NOX','SO2','PM25-PRI','PM02']
    days = ['weekdy','satdy','sundy']
    halves = ['00to11Z','12to23Z']
    data_dir = '/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES'
    years = [2021]     
    months = ['Month'+datetime.datetime(1993,m,1).strftime('%m') for m in np.arange(1,13)]
    version = 'GRA2PESv1.0'

def check_file(search_string):
    if len(glob.glob(search_string)) == 0:
        print('Issue finding file')
        print(search_string)
        breakpoint()
    return

def check_aggs(n):
    i = 0
    while i <= n:
        m = np.random.randint(0,high=len(inputs.months))
        s = np.random.randint(0,high=len(inputs.species))
        sec = np.random.randint(0,high=len(inputs.sums.keys()))
        d = np.random.randint(0,high=len(inputs.days))
        h = np.random.randint(0,high=len(inputs.halves))
        y = np.random.randint(0,high=len(inputs.years)) 
        #breakpoint()
        year = inputs.years[y]
        month = inputs.months[m]
        day = inputs.days[d]
        half = inputs.halves[h]
        speciess = inputs.species[s]
        sector = [k for k in inputs.sums.keys()][sec]

        fn = f"{inputs.data_dir}/{year}/{month}/{sector}/{day}/*{half}*.nc"
        check_file(fn)
        
        print('sector')
        print('Opening aggregated file')
        print(glob.glob(fn)[0])
        data_agg = xr.open_dataset(glob.glob(fn)[0])
        agg_vals = data_agg[speciess].values

        print('Opening component files')
        component_vals =[]
        component_sum = []
        for i,s in enumerate(inputs.sums[sector]):
            sfn = fn.replace(sector,s)
            check_file(sfn)

            print(glob.glob(sfn)[0])
            compdata  = xr.open_dataset(glob.glob(sfn)[0])
            component_vals.append(compdata[speciess].values)
            if i == 0:
                component_sum = compdata[speciess].values 
            else:
                component_sum = np.add(component_sum,compdata[speciess].values)
        component_vals = np.array(component_vals)
        sum_components = np.sum(component_vals,axis=0)

        diff = agg_vals - sum_components
        if np.mean(diff) != 0:
            print('Check your arithmetic!')
            breakpoint()
        else:
            print('Great sum well done!')

        i+=1

if __name__ == '__main__':
    check_aggs(5)

    
