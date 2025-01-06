import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import datetime
import glob
"""
Author: Emily Bell
Contact: emily.i.bell@nasa.gov

Outputs bar plot showing GRA2PES city TOTAL CO2 emissions vs. Vulcan Scope 1 and Scope 2 for cities in the urban dashboard, for 2021.
"""

# Urban dashboard cities
cities_file = '/discover/nobackup/projects/gmao/geos_carb/embell/ghgc/city_lists/urban_dashboard.json'
id_key = 'PLACENS'
with open(cities_file,'r') as infile:
    cities = json.load(infile) 

# Info for GRA2PES 2021 monthly city totals, and where to output the figure
version = 'GRA2PESv1.0'
years = [2021]
months = ['Month'+datetime.datetime(1993,m,1).strftime('%m') for  m in np.arange(1,13)]
data_dir = '/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES/for_ghgc/city_totals'
conservative = True     # using only GRA2PES grid cells where the centers are inside the city boundary
image_dir = '/discover/nobackup/projects/gmao/geos_carb/embell/images/GRA2PES/'
# We don't need the percentages that are also included in the CSV
use_keys = ['Species', 'Industrial', 'Onroad Transportation', 'Power', 'Nonroad Transportation', 'Other', 'Residential + Commercial']

# Vulcan 2021 city sector totals (provided by Vulcan team)
vfile = '/discover/nobackup/projects/gmao/geos_carb/embell/data/Vulcan/v4.0/city_totals/PLACE/AllSectors.CO2.PLACE.2021.allyrs.csv'
vdata = pd.read_csv(vfile)



# Calculate annual CO2 total from GRA2PES monthly data
# and grab Vulcan totals for the same cities
gra2pes_annual_totals = [] 
vulcan_scope1 = []
vulcan_scope2 = []
for city in cities.keys():
    monthly_totals = []
    for month in months:
        ss = f"{data_dir}/*{city.replace('/','_').replace(' ','_')}*{month}*breakdown*.csv"
        if conservative:
            ss = ss.replace('.csv','_conservative.csv')
        fn = glob.glob(ss)[0]
        data = pd.read_csv(fn)
        monthly_totals.append(data[data['Species'] == 'Carbon Dioxide'][use_keys].sum(axis=1,numeric_only=True).values)
    gra2pes_annual_totals.append(np.array(monthly_totals).sum())
    vulcan_scope1.append(vdata[vdata[id_key] == int(cities[city][id_key])]['Total_Scope1only_FFCO2_tC'].values)
    vulcan_scope2.append(vdata[vdata[id_key] == int(cities[city][id_key])]['Total_withScope2_FFCO2_tC'].values)
    
# Vulcan native values are tons of Carbon - scale by 44./12. to get tons of CO2.
vulcan_scope1 = np.array(vulcan_scope1).squeeze()*(44./12.)
vulcan_scope2 = np.array(vulcan_scope2).squeeze()*(44./12.)

#breakpoint()
# Put data into pandas to easily create the multi-bar plot
df = pd.DataFrame(
    {'City':[k for k in cities.keys()],
    'GRA2PES':gra2pes_annual_totals,
    'Vulcan Scope 1':vulcan_scope1,
    'Vulcan Scope 1+2':vulcan_scope2}
)
fig,ax = plt.subplots(figsize=(15,8))
p = df.plot(ax=ax,kind='bar',x='City',y=['GRA2PES','Vulcan Scope 1','Vulcan Scope 1+2'],color=['#E8781C','#22BEE6','#3E7582'])
p.set_xlabel('City')
p.set_ylabel('Metric Tons CO2')
p.set_title('City Annual Total CO2 Emissions for 2021')
sn = f'{image_dir}/city_annual_CO2_totals.png'
if conservative:
    p.set_title('City Annual Total CO2 Emissions for 2021\nMore conservative grid cell selection')
    sn = sn.replace('.png','_conservative.png')
print('Saving ',sn)
plt.tight_layout()
plt.savefig(sn)

        
