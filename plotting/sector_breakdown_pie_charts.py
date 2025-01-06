import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import datetime
import numpy as np
import json
import glob
"""
Outputs imitation pie charts for visual spot checking against Urban Dashboard visualizations
"""

class inputs:
    version = 'GRA2PESv1.0'
    years = [2021]
    months = ['Month'+datetime.datetime(1993,m,1).strftime('%m') for m in np.arange(7,8)]
    data_dir = '/discover/nobackup/projects/gmao/geos_carb/embell/data/GRA2PES/for_ghgc/city_totals' 
    city_file = '/discover/nobackup/projects/gmao/geos_carb/embell/ghgc/city_lists/urban_dashboard.json'
    image_dir = '/discover/nobackup/projects/gmao/geos_carb/embell/images/GRA2PES/city_pie_charts'
    slices = ['Industrial','Onroad Transportation','Nonroad Transportation','Residential + Commercial','Power','Other']
    colors = ['#BCF4F5','#7E6B8F','#FF715B','#D3FAC7','#FFB7C3','#FF9B71']
    
    with open(city_file,'r') as infile:
        cities = json.load(infile)

def set_subplot_dims(nn):
    ncol = 2
    nrow = int(np.ceil(nn/ncol))
    return nrow,ncol

def pie_chart_figure(data,sn=None):
    nrow,ncol = set_subplot_dims(len(data['Species']))
    fig,axs = plt.subplots(nrow,ncol,figsize=(8,12)) 
    for i,ax in enumerate(axs.flatten()):
        if i >= len(data['Species']):
            plt.delaxes(ax)
        else:
            p = data.iloc[i][inputs.slices].plot(
                kind='pie',
                ax=ax,
                colors=inputs.colors, 
                wedgeprops={"linewidth": 1, "edgecolor": "#252323"},
                labeldistance=None
            )
            ax.set_title(data.iloc[i]['Species'])
            if i == len(data['Species'])-1:
                ax.legend(loc='best')
    
    plt.tight_layout()
    if sn:
        plt.savefig(sn)
    plt.close('all')

    return

def gen_pie_charts(inputs):
    for year in inputs.years:
        for city in inputs.cities.keys():
            for month in inputs.months:
                f = glob.glob(f"{inputs.data_dir}/{city.replace('/','_').replace(' ','_')}_{year}_{month}_species_sectoral_breakdown.csv")[0]
                data = pd.read_csv(f)
                pie_chart_figure(data,sn=f"{inputs.image_dir}/{city.replace('/','_').replace(' ','_')}_{month}_{year}_pies.png")

    return
 

if __name__ == '__main__':
    gen_pie_charts(inputs)
