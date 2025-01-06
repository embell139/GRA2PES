import pandas as pd
import numpy as np
import json
import xarray as xr


cityfile = '/discover/nobackup/projects/gmao/geos_carb/embell/ghgc/city_lists/r3_urban_dashboard.json'

with open(cityfile,'r') as infile:
    cities = json.load(infile)

city_names =[k for k in cities.keys()]
sectors = ['Airports','Residential Buildings','Commercial Buildings','Industrial Buildings','Power Plants','Onroad Gas','Onroad Diesel','Electricity']
species = ['CO2','CO','NOX','SOX','PM2.5']
mass_labels = [s+' Mass' for s in sectors]
percent_labels = [s+' Percent of Total' for s in sectors]

city_totals = []
city_fractions = []
for city in city_names:
    print(city)
    species_totals = []
    species_fractions = []
    for s in species:
        print(s)
        print(len(sectors))
        species_total = np.random.rand(1)*1e6
        sector_rand = np.random.rand(len(sectors))
        sector_fractions_this = sector_rand/np.sum(sector_rand)
        species_fractions.append(sector_fractions_this) 
        species_totals.append(sector_fractions_this * species_total)

    city_totals.append(species_totals)
    city_fractions.append(species_fractions)

city_totals = np.array(city_totals)
city_fractions = np.array(city_fractions)
         

#totals = np.random.rand(len(city_names),len(species))
#sector_fractions = sector_rand/np.sum(sector_rand)
#sector_totals = []
#for i,s in enumerate(sectors):
#    fraction_this = sector_fractions[:,:,i]
#    totals_this = fraction_this * totals
#    sector_totals.append(totals_this)
#    if np.array_equal(totals_this,totals) 
     
ds = xr.Dataset(
    data_vars={'mass':(['city','species','sector'],city_totals),'percent':(['city','species','sector'],city_fractions*100.)},
    coords={'city':(['city'],city_names),'species':(['species'],species),'sector':(['sector'],sectors)})

for city in city_names:
    dd = ds.sel(city=city)
    df_mass = pd.DataFrame(data=dd['mass'].values,columns=mass_labels)
    df_percent = pd.DataFrame(data=dd['percent'].values,columns=percent_labels)
    df = pd.concat([df_mass,df_percent],axis=1)
    df['Species'] = species
    neworder = ['Species',*mass_labels,*percent_labels]
    df = df[neworder]

    sn = f'/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES/for_ghgc/city_totals/mockup_csv/2021_{city}_species_sector_totals.csv'
    df.to_csv(sn)
    print('Saved to ',sn)

print('Done.')

