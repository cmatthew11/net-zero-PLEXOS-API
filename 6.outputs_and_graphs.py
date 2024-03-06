# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 08:56:35 2023

@author: Chris
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from collections import OrderedDict
import seaborn as sns
from scipy import stats
import pathlib
from matplotlib.container import BarContainer#
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

from plexosapi import *
 
#%% triggers plots

PLOTMAIN = True # mainland 
PLOTISLSUM = False # grouped island results
P2XANALYSIS = False # looking at p2x operatoin
#%% 
def rename_main_gen(df, group=True):
    """ converts mainland generation names into graph names"""
    G = GENNAMES
    if group:
        G = GROUPGENNAMES
    if type(df) is pd.core.frame.DataFrame:
        df = df[[i for i in df.columns if 'MA-0' in i]]
        df.columns = [G[i.split('_')[0]] for i in df.columns]
        if group:
            df = df.groupby(df.columns,axis=1).sum()
    else:
        df = df[[i for i in df.index if 'MA-0' in i]]
        df.index = [G[i.split('_')[0]] for i in df.index]
        if group:
            df = df.groupby(df.index).sum()

    return df

def plot_with_secondary_axis(dataframe, secondary_columns, title, 
                             firstkwargs={}, secondkwargs={}):
    """creates a plot with a secondary axis for that category of data in the 
    same dataframe"""
    fig, ax1 = plt.subplots()
    primary_columns = [i for i in dataframe if i not in secondary_columns]
    # Plot primary columns on the left y-axis
    dataframe[primary_columns].plot(ax=ax1,linewidth=3,**firstkwargs)
    ax1.legend(primary_columns, loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_xlabel(None)
    # Plot secondary columns on the right y-axis
    color = 'k'
    for column in secondary_columns:
        ax2.plot(dataframe[column], color=color, **secondkwargs)
    fig.tight_layout()  # otherwise, the right y-label is slightly clipped
    ax1.set_title(title)
    plt.show()
    

def corrdot(*args, **kwargs):
    """https://stackoverflow.com/questions/48139899/correlation-matrix-plot-with-coefficients-on-one-side-scatterplots-on-another"""
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = round(corr_r, 2)
    ax = plt.gca()
    font_size = abs(corr_r) * 80 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)
    

def corrfunc(x, y, **kws):
    """https://stackoverflow.com/questions/48139899/correlation-matrix-plot-with-coefficients-on-one-side-scatterplots-on-another"""
    r, p = stats.pearsonr(x, y)
    p_stars = ''
    if p <= 0.05:
        p_stars = '*'
    if p <= 0.01:
        p_stars = '**'
    if p <= 0.001:
        p_stars = '***'
    ax = plt.gca()
    ax.annotate(p_stars, xy=(0.65, 0.6), xycoords=ax.transAxes,
                color='red', fontsize=70)
    
def corrmatrix(data):
    """ see the above links"""
    sns.set(style='white', font_scale=1.6)
    g = sns.PairGrid(data, aspect=1.5, diag_sharey=False, despine=False)
    g.map_lower(sns.regplot, lowess=True, ci=False,
                line_kws={'color': 'red', 'lw': 1},
                scatter_kws={'color': 'black', 's': 20})
    g.map_diag(sns.distplot, color='black',
               kde_kws={'color': 'red', 'cut': 0.7, 'lw': 1},
               hist_kws={'histtype': 'bar', 'lw': 2,
                         'edgecolor': 'k', 'facecolor':'grey'})
    g.map_diag(sns.rugplot, color='black')
    g.map_upper(corrdot)
    g.map_upper(corrfunc)
    g.fig.subplots_adjust(wspace=0, hspace=0)
    
    # Remove axis labels
    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')
    
    # Add titles to the diagonal axes/subplots
    for ax, col in zip(np.diag(g.axes), data.columns):
        ax.set_title(col, y=0.82, fontsize=26)
        
    g.fig.subplots_adjust(top=0.9) # adjust the Figure in rp
    g.fig.suptitle(s)
    
def neat_scen(scen,cat):
    """ returns nicer scenario names"""
    if cat == 's':
        return scen.split('-')[0].capitalize()
    elif cat == 'w':
        return scen.split('-')[1].replace('_',' ').capitalize()
    else:
        raise Exception('must be weather or scen')
        
def get_node(index):
    return [i.split('_')[1] for i in index]

def get_zone(index):
    return [i.split('_')[1].split('-')[0] for i in index]

def hist_density_plot(val, bin_, density, c, ax):
    d = stats.gaussian_kde(val)
    weights = np.ones_like(val) / len(val)
    n, b, _ = ax.hist(val, bins=bin_,
                           histtype=u'step', 
                           density=True, 
                           linewidth=1,
                           color=c, 
                           label='_nolegend_',
                           weights=weights)

    ax.plot(b, d(b),
             color=c,
             linewidth=3)

def align_yaxis(ax1, v1, ax2, v2):
    """https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin/10482477#10482477
    ONLY WORKS ONE WAY AROUND
    adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

#%%

OUTPUTS = r'C:\Users\Chris\Desktop\Coding\Python\PLEXOS\PLEXOS API\Output analysis'
# use the below if not defined
FOLDER = max(pathlib.Path(OUTPUTS).glob('*/'), key=os.path.getmtime)

DEM = r'C:\Users\Chris\Desktop\Coding\Python\PLEXOS\Input data\Demand\total_PLEXOS_demand.pickle'
dem = pd.read_pickle(DEM)

FOLDER = os.path.join(OUTPUTS, FOLDER)

print(f'THIS FOLDER BEING USED: {FOLDER}')

RES = os.path.join(FOLDER, 'DA_ID_Diff_outputs.pickle')
res = pd.read_pickle(RES)

SS = list(res.keys())
scens = sorted(list(set([i.split('-')[0] for i in SS])))
ws = list(set([i.split('-')[1] for i in SS]))

GENS = ['GCC_MA-0', 'BEC_MA-0', 'PST_MA-0', 'NUC_MA-0', 'REN_MA-0', 'HYD_MA-0',
       'ONS_AB-23', 'ONS_HI-20', 'ONS_LH-12', 'ONS_LH-13', 'ONS_LH-14',
       'ONS_LH-17', 'ONS_MA-0', 'ONS_OR-1', 'ONS_OR-2', 'ONS_OR-3', 'ONS_OR-4',
       'ONS_OR-5', 'ONS_SH-10', 'ONS_SH-6', 'ONS_SH-7', 'ONS_SH-8', 'ONS_SH-9',
       'OFF_MA-0', 'SOL_MA-0', 'MAR_MA-0', 'MAR_OR-1', 'HYG_MA-0']

GENNAMES = {'GCC': 'Gas CCUS',
            'BEC': 'Biomass CCUS',
            'PST': 'Pumped storage hydro',
            'HYG': 'Hydrogen',
            'NUC': 'Nuclear',
            'REN': 'Other renewables',
            'HYD': 'Hydro',
            'ONS': 'Onshore wind',
            'OFF': 'Offshore wind',
            'SOL': 'Solar PV',
            'MAR': 'Tidal current'}

ZONEMAP = {'AB':'Argyll\nand Bute', 
           'OR':'Orkney',
           'SH':'Shetland',
           'HI':'Highland',
           'MA':'Mainland',
           'LH':'Na h-Eileanan\nSiar',
           'NA':'North\nAyrshire' 
           }

""" more specific groups"""


GROUPGENNAMES = {'GCC': 'Gas CCUS',
                'BEC': 'Biomass CCUS',
                'PST': 'Storage',
                'HYG': 'Hydrogen',
                'NUC': 'Nuclear',
                'REN': 'Other renewables',
                'HYD': 'Other renewables',
                'ONS': 'Wind',
                'OFF': 'Wind',
                'SOL': 'Solar PV',
                'MAR': 'Other renewables',
                'BAT': 'Storage'
                }

GENAREACOLS = OrderedDict({'Nuclear': 'firebrick',
                           'Biomass CCUS': 'orangered',
                           'Other renewables': 'darkorange',
                           'Solar PV': 'mediumorchid',
                           'Wind': 'slateblue',
                           'Storage': 'aqua',
                           # 'DSR':'navy',
                           'Gas CCUS': 'lime',
                           'Hydrogen': 'forestgreen'
                           })


WRENAME = {'winter_low_wind_high_demand': 'Low wind (winter)',
            'summer_high_wind': 'High wind (summer)',
            'summer_low_wind_high_demand': 'Low wind (summer)'}

NODEGROUPS = {
             'AB-22': 'AB-4',
             'AB-23': 'AB-4',
             'AB-24': 'AB-4',
             'AB-25': 'AB-3',
             'AB-26': 'AB-3',
             'AB-27': 'AB-3',
             'HI-18': 'HI-5',
             'HI-19': 'HI-5',
             'HI-20': 'HI-5',
             'HI-21': 'HI-5',
             'LH-12': 'LH-7',
             'LH-13': 'LH-7',
             'LH-14': 'LH-7',
             'LH-15': 'LH-7',
             'LH-16': 'LH-6',
             'LH-17': 'LH-6',
             'MA-0': 'MA-0',
             'NA-28': 'NA-1',
             'NA-29': 'NA-1',
             'NA-30': 'NA-2',
             'NA-31': 'NA-2',
             'OR-1': 'OR-8',
             'OR-2': 'OR-8',
             'OR-3': 'OR-8',
             'OR-4': 'OR-8',
             'OR-5': 'OR-8',
             'SH-10': 'SH-9',
             'SH-11': 'SH-9',
             'SH-6': 'SH-9',
             'SH-7': 'SH-9',
             'SH-8': 'SH-9',
             'SH-9': 'SH-9'
             }


RENGEN = [i for i in GENS if not any([j in i for j in ['GCC', 'BEC', 'NUC',
                                                        'PST']])] # these are thermal since can be dispatched
THERMGEN = [i for i in GENS if i not in RENGEN]
ISLGEN = [i for i in GENS if 'MA-0' not in i ]
MAGEN = [i for i in GENS if 'MA-0' in i ]

GENMAP = dict(zip(GENS,['REN' if i in RENGEN else 'THERM' for i in GENS ]))

allscens = sorted(allscens)

NEATSCENS = [[j.replace('summer_','').replace('winter_','').replace('_',' ').capitalize() for j in i.split('-')] for i in allscens]
SUMMERSCENS = [i for i in allscens if 'summer' in i]
WINTERSCENS = [i for i in allscens if 'winter' in i]

LOWGENZONES = ['HI','AB','NA']

fig_out = {}
#%% mainland data comparisons

main = {}
#price
main['price'] = dict(zip(SS,[res[i]['price'] for i in SS]))

m = len(set([i.split('-')[0] for i in SS]))
n = len(set([i.split('-')[1] for i in SS]))
main['pricemean'] = pd.concat([main['price'][i] for i in SS],axis=1)
main['pricemean'] = pd.DataFrame(main['pricemean'].mean().values.reshape(m, n ),
                              index=set([i.split('-')[0] for i in SS]),
                              columns=set([i.split('-')[1] for i in SS])
                              )

main['maxcap'] = rename_main_gen(res[SS[0]]['maxcap'].mean())
# this is fudged as the average hour capacity (value is in MWH not MW)
main['maxcap']['Storage'] += res[SS[0]]['battcap'].mean()['BAT_MA-0']
main['maxcap']['Electrolysis'] = res[SS[0]]['p2xcap'].mean()['power2x_MA-0']

for i in ['gen', 'allgen', 'p2xload','battgen','battload','dem','dsr']:
    main[i] = {}

for scen in res:
    main['p2xload'][scen] = res[scen]['p2xload']['power2x_MA-0']
    main['battgen'][scen] = res[scen]['battgen']['BAT_MA-0']
    main['battload'][scen] = res[scen]['battload']['BAT_MA-0']
    # main['dsr'][scen] = res[scen]['dsrcharging']['charging-station_MA-0']
    
    main['dem'][scen] = res[scen]['dem']['MA-0']

    new = res[scen]['gen'].copy()
    new['BAT_MA-0'] = main['battgen'][scen]
    # new['DSR_MA-0'] = main['dsr'][scen]
    new = rename_main_gen(new)
    main['allgen'][scen] = new

    
plot_range = {'winter_low_wind_high_demand':list(range(312,480)),
              'summer_high_wind':list(range(312,480)),
              'summer_low_wind_high_demand':list(range(120,288))}

def gw(df):
    return df/1000

if PLOTMAIN:
    #mean price
    plt.figure()
    main['pricemean'].plot.bar()
    # gen cap
    plt.figure()
    main['maxcap'].plot.bar()
    
    #hourly price
    q = pd.concat([pd.DataFrame(main['price'][i].values) for i in SS],axis=1)
    q.columns = SS
    q = q.iloc[:,:3]
    q.columns = [WRENAME[i.split('-')[1]] for i in q]
    ax = q.plot(linewidth=3)
    ax.set_yscale('log')
    # plt.ylim(0,1500)
    
    #generation
    for scen in SS[:2]:
        plt.rc('font', size=20) 
        w = scen.split('-')[1]
        R = plot_range[w]
        # hourly average looks wierd probably ignore
        #add this to get hourly average- .groupby([i%24 for i in range(672)]).mean()
        # look at representative week instead
        
        ax = gw(main['allgen'][scen][GENAREACOLS.keys()].iloc[R,:]).plot.area(color=GENAREACOLS)

        #fill between for p2x
        y1 = gw(main['allgen'][scen].sum(axis=1))
        y2 = gw(main['allgen'][scen].sum(axis=1)-main['p2xload'][scen])
        x = y1.index
        # https://stackoverflow.com/questions/29549530/how-to-change-the-linewidth-of-hatch-in-matplotlib
        plt.rcParams['hatch.linewidth'] = 2
        ax.fill_between(x, y2, y1, facecolor='none', hatch='/',
                        edgecolor='k',zorder=1)
        h,l = ax.get_legend_handles_labels()
        
        LEG = list(GENAREACOLS)+['Demand (less electrolysis)']
        ax.legend(h,LEG, loc=3, fontsize=12)
        # plt.legend()
        plt.ylabel('Demand (GW)')

        # plt.title(scen.split('-')[1].replace('_',' ').capitalize())
        
        fig_out[f'mainland_gen_{w}'] = plt.gcf()


#%% DONE island generation curtailed compare

isl_gen = {}
cols = ['Generation','Demand','Curtailed','Imports','Exports']

fig,ax = plt.subplots(2,1,
                      layout='constrained')
ss = ['import-summer_high_wind',
          'export-summer_high_wind',
          'independence-summer_high_wind',
          'middle-summer_high_wind']
scens = [i.split('-')[0] for i in ss]

for i,w in enumerate(ws):
    isl_gen[w] = pd.DataFrame(np.nan, index=scens,
                              columns=cols)
    ss = [f'{s}-{w}' for s in scens]
    
    isl_gen[w].loc[scens,'Generation'] = [res[s]['gen'][[i for i in ISLGEN if i in res[s]['gen']]].sum().sum() for s in ss]
    isl_gen[w].loc[scens,'Curtailed'] = [res[s]['curtailed'][[i for i in ISLGEN if i in res[s]['curtailed']]].sum().sum() for s in ss]
    # isl_gen[w].loc[scens,'Undispatched'] = [res[s]['undispatched'][[i for i in ISLGEN if i in res[s]['curtailed']]].sum().sum() for s in ss]
    isl_gen[w].loc[scens,'Imports'] = [res[s]['zoneimports'][ISLZONES].sum().sum() for s in ss]
    isl_gen[w].loc[scens,'Exports'] = [res[s]['zoneexports'][ISLZONES].sum().sum() for s in ss]
    # isl_gen[w].loc[scens,'Max cap (MW)'] = [res[s]['maxcap'][[i for i in ISLGEN if i in res[s]['maxcap']]].mean().sum() for s in ss]
    
    isl_gen[w][[i for i in isl_gen[w] if 'Max' not in i]] /= 1000
    isl_gen[w].loc[scens,'Demand'] = [dem[SCENMAP.loc[s.split('-')[0],'demand']]['elec'][w][ISLNODES].sum().sum() for s in ss]
    isl_gen[w]['Demand'] /= 10**9
    isl_gen[w].loc[scens,'Electrolysis'] = [res[s]['p2xload'][[i for i in res[s]['p2xload'] if 'MA-0' not in i]].sum().sum()/1000 for s in ss]

    isl_gen[w].index = ['BAU' if i=='Import' or i == 'import' else i.capitalize() for i in isl_gen[w].index ]
    
    isl_gen[w][cols].plot.bar(ax=ax[i],
                        rot=0,
                        legend=False,
                        width=0.85)
    
    ax[i].set_ylim(0,800)
    fig.legend(cols,
               ncol=len(cols),
               loc='center',
               fontsize=15)
    # ax2 = ax[i].twinx()
    # ax2.set_ylabel(w.replace('_',' ').capitalize())
    # ax2.set_yticks([])
    ax[i].text(3.3,700,WRENAME[w],
           ha='right')
    if i==1:
        c='k'
    else:
        c='w'
    ax[i].set_xticklabels(isl_gen[w].index, color=c)
    
    bars = [i for i in ax[i].containers if isinstance(i, BarContainer)]
    #add extra color for electrolysis
    ax[i].bar(
              x=[i.get_x()+0.085 for i in bars[1]],
              bottom=[i.get_height() for i in bars[1]],
              width=0.17, 
               height=isl_gen[w]['Electrolysis'].values,
               color=['orangered']*len(ss),
               zorder=1
               )
    
    bars = [i for i in ax[i].containers if isinstance(i, BarContainer)]
    for bar in bars:
        lab = bar.get_label()
        if lab != 'Generation' and lab!='Demand':
            if '_' in lab:
                q = (((isl_gen[w]['Demand']+isl_gen[w]['Electrolysis'])/isl_gen[w]['Generation'])*100).astype(int)
            else:
                q = ((isl_gen[w][lab]/isl_gen[w]['Generation'])*100).astype(int)
                # q = ((isl_gen[w][lab]/isl_gen[w]['Generation'])*100).round(1)
            q = [f'{i}%' for i in q]    
            ax[i].bar_label(bar,
                            labels=q,
                            fontsize=15
                            )
            
    fig.supylabel('Monthly electricity (GWh)')

fig_out['island_gen_by_scen'] = plt.gcf()

#%% DONE island power2x curtailed compare

total_p2x = {}
cols = ['Demand','Electrolysis','Curtailed','Net exports']

fig,ax = plt.subplots(2,1,
                      layout='constrained')

for i,w in enumerate(ws):
    ss = [f'{s}-{w}' for s in scens]
    total_p2x[w] = pd.DataFrame(np.nan, index=scens,
                              columns=cols)
    q = total_p2x[w]
    q.loc[scens,'Demand'] = [res[s]['gasdemand'][[i for i in res[s]['gasdemand'] if 'MA-0' not in i]].sum().sum() for s in ss]
    q.loc[scens,'Electrolysis'] = [res[s]['p2xload'][[i for i in res[s]['p2xload'] if 'MA-0' not in i]].sum().sum() for s in ss]
    q.loc[scens,'Electrolysis'] *= 3.6 
    q.loc[scens,'Curtailed'] = [res[s]['curtailed'][[i for i in ISLGEN if i in res[s]['curtailed']]].sum().sum() for s in ss]
    q.loc[scens,'Imports'] = [res[s]['gasshortage'][[i for i in res[s]['gasshortage'] if 'MA-0' not in i]].sum().sum() for s in ss]
    q /= 1000
    q.loc[scens,'Exports'] = [res[s]['p2xmarketsales'][[i for i in res[s]['p2xmarketsales'] if 'MA-0' not in i]].sum().sum() for s in ss]
    q.loc[scens,'Net exports'] = q.loc[scens,'Exports'] - q.loc[scens,'Imports']
    #This just checks that exports is the right units
    # q.loc[scens,'dem+exp'] = (q.loc[scens,'Demand'] + q.loc[scens,'Exports'] - q.loc[scens,'Imports'])/0.82
    
    q.index = ['BAU' if i=='Import' or i == 'import' else i.capitalize() for i in q.index ]

    q[cols].plot.bar(ax=ax[i],
                     legend=False,
                     rot=0,
                     width=0.85)
    
    ax[i].set_ylim(-0,1000)
    ax[i].set_yticks([-250,0,250,500,750,1000])

    
    # ax2 = ax[i].twinx()
    # ax2.set_ylabel(w.replace('_',' ').capitalize())
    # ax2.set_yticks([])
    ax[i].text(3,850,WRENAME[w],
               ha='right')
    
    if i==1:
        c='k'
    else:
        c='w'
    ax[i].set_xticklabels(q.index, color=c)

fig.legend(cols,
           ncol=len(cols),
           loc='center right',
           fontsize=15)
fig.supylabel('Energy (TJ)')
fig_out['island_electrolysis_bar_chart'] = plt.gcf()


    
        
#%% DONE island power2x curtailed compare- but for regions

isl_p2x = {}
cols = ['Demand','Electrolysis','Curtailed','Net exports']

nscens = 3
fig,ax = plt.subplots(nscens,1,
                      layout='constrained')

Z =['AB', 'NA', 'HI', 'SH', 'OR', 'LH']

allscens2 = ['import-summer_high_wind',
  'import-winter_low_wind_high_demand',
  'export-summer_high_wind',
  'export-winter_low_wind_high_demand',
  'independence-summer_high_wind',
  'independence-winter_low_wind_high_demand',
  'middle-summer_high_wind',
  'middle-winter_low_wind_high_demand']

ss = [i for i in allscens2 if 'summer' in i]

def group_island(df,zone=True,exc_main=True):
    if zone:
        group = get_zone(df)
    else:
        group = get_node(df)
    out = df.groupby(group,axis=1).sum()
    if exc_main:
        out = out[[i for i in out if 'MA' not in i]]
    out = out.sum()
    return out
    

for i,s in enumerate(ss[:nscens]):
    isl_p2x[s] = pd.DataFrame(np.nan, index=Z,
                              columns=cols)
    q = isl_p2x[s]
    
    r = res[s]
    #gasdemand
    q.loc[Z,'Demand'] = group_island(r['gasdemand'])
    #electrolysis
    q.loc[Z,'Electrolysis'] = group_island(r['p2xload'])
    q.loc[Z,'Electrolysis'] *= 3.6 
    #curtailed
    q.loc[Z,'Curtailed'] = group_island(r['curtailed'])
    #Imports
    q.loc[Z,'Imports'] = group_island(r['gasshortage'])
    q /= 1000 #TJ
    #Exports
    q.loc[Z,'Exports'] = group_island(r['p2xmarketsales']) 
    
    q.loc[Z,'Net exports'] = q.loc[Z,'Exports'] - q.loc[Z,'Imports']
     

    if i == nscens-1:
        LABEL = [ZONEMAP[i] for i in q.index]
    else:
        LABEL = []
    q[cols].plot.bar(ax=ax[i],
                     legend=False,
                     width=0.85,
                     rot=0)
    ax[i].set_xticklabels(LABEL,
                          fontsize=12)
    
    fig.legend(cols,
               ncol=len(cols),
               loc='upper center',
               fontsize=12)
    fig.supylabel('Energy (TJ)')
    ax[i].axvline(2.5, c='k', linestyle='--',
                  linewidth=0.5, label="_nolegend_")
    ax2 = ax[i].twinx()
    if 'import' in s:
        s = 'BAU'
    else:
        s = neat_scen(s,'s')
    ax2.set_ylabel(s,
                     size=20
                     )
    ax2.set_yticks([])
    
    if i==nscens-1:      
        height=0.75
        ylim = q.max().max()
        ax[i].annotate('Low\ngeneration',
            [-0.4, ylim*height],
            size=15,
            ha='left',style='italic')
        ax[i].annotate('High\ngeneration',
            [2.6,ylim*height],
            size=15,
            ha='left',style='italic')

    fig_out['island_electrolysis_by_region'] = plt.gcf()
    
    
#%% p2x storage sizing

# LOOK AT SIZE OF HYDROGEN STORAGE BY SCENARIO- HOW MUCH IS NEEDED TO MEET DEMAND- 
# ALSO CAN OIL AND GAS INFRSTRUCUTRE BE USED TO EXPORT FROM ORKENY OR SHETLAND?
# also what is the required size to export if export is once a month say
# how do the sizes compare


#storage sizing is the net of the market sales (export) and shortages (import)
#sizing for demand is the min of this (shortage negative)
#sizing for export is the max (sales positive)

p2x_store = {}
cols = ['Import sized storage','Export sized storage','Demand']

fig, ax = plt.subplots(1,1,
                   constrained_layout=True)

p2x_store = pd.DataFrame(np.nan, index=scens,
                              columns=cols)
for s in scens:
    for w in ws:
        scen = f'{s}-{w}'
        r = res[scen]
        #SALES IS EXCESS HYDROGEN
        sales = r['p2xmarketsales'].copy()
        sales.columns = [i.split('_')[1].split('gas')[0] for i in sales]
        #SHORT IS WHEN STORAGE IS NEEDED
        short = r['gasshortage'].copy()
        short.columns = [i.split('_')[1] for i in short]
        short = short / 1000
        for i in sales:
            if i not in short:
                short[i] = 0
        net = sales - short
        net = net[ISLNODES]
        
        comp = 0
        if 'winter' in w:
            p2x_store.loc[s, 'Import sized storage'] = abs(net.cumsum().min().sum())
            # this would do the storage sizing by weekly demand but ignore
            # (net.cumsum().groupby([i//168 for i in range(672)]).max()-net.cumsum().groupby([i//168 for i in range(672)]).min()).max().sum()
            
            q = net.diff().abs()
            q = q.max().sum()
            if q>comp:
                comp = q
        if 'summer' in w:
            p2x_store.loc[s, 'Export sized storage'] = abs(net.cumsum().max().sum())
            # weekly storage sizing
            # (net.cumsum().groupby([i//168 for i in range(672)]).max()-net.cumsum().groupby([i//168 for i in range(672)]).min()).min().sum()
            
            q = net.diff().abs()
            q = q.max().sum()
            if q>comp:
                comp = q
            p2x_store.loc[s, 'Compressor size (MW)'] = q * 0.277
        
        p2x_store.loc[s, 'Demand'] = r['gasdemand'][[i for i in r['gasdemand'] if 'MA-0' not in i]].sum().sum() / 1000

p2x_store.index = ['BAU' if i=='Import' or i == 'import' else i.capitalize() for i in scens]
p2x_store[['Import sized storage', 'Export sized storage', 'Demand']].plot.bar(rot=0,
                   ax=ax)
ax.set_ylabel('Energy (TJ)')

fig_out['hyrdogen_storage_sizing'] = plt.gcf()
pd.to_pickle(p2x_store,
             os.path.abspath(r'C:\Users\Chris\Desktop\Coding\Python\PLEXOS\Costing\hydrogen_storage.pickle'))

#%% ROUGH CALC OF WATER DEMAND PER REGION

water_act = pd.read_excel(r'C:/Users/Chris/Desktop/Coding/Python/Demand Modelling/Water Treatment/Scottish Islands Water Treatment Energy Data.xlsx',
                      sheet_name='Sheet1')
water_act = water_act[~water_act.gsp_group.isna()]
water_act = water_act.groupby('Local_Authority_Name')['Annual Ml treated water 2018-19'].sum() 
water_act *= 10**6 # ML to L
water_act.index = [i.replace(' Islands','') for i in water_act.index]

ZONEMAP2 = {'AB':'Argyll and Bute', 
           'OR':'Orkney',
           'SH':'Shetland',
           'HI':'Highland',
           'LH':'Na h-Eileanan Siar',
           'NA':'North Ayrshire' 
           }

hyd_dem = pd.DataFrame(0,index=ZONEMAP2.values(),
                       columns=[neat_scen(s,'s') for s in scens])

for s in scens:
    ss = [i for i in allscens if s in i]
    # 6 to get annual values
    r = (res[ss[0]]['p2xload'] + res[ss[1]]['p2xload'].values).sum() * 6 #index is different hence values
    r = r.groupby([i.split('_')[1].split('-')[0] for i in r.index]).sum() #MWh
    r = r.loc[[i for i in ISLZONES if i in r]]
    r.index = [ZONEMAP2[i] for i in r.index]
    hyd_dem.loc[:,neat_scen(s,'s')] = r


# 11-14.5 L/kg H2
# 33.3 kwh/kg H2
# 0.33-0.44 L/kWh H2
# 330-440L/MWh
hyd_dem *= 0.82 # this gives hydrogen not the elec required

water = {}
water['low'] = hyd_dem * 330
water['high'] = hyd_dem * 440
water['mean'] = (water['low'] + water['high'])/2

q = water['mean']
q = q[['Export', 'Independence', 'Middle']]
err = water['high']-water['mean']
for i in q:
    q[i] = q[i]/water_act
    err[i] = err[i]/water_act


fig,ax = plt.subplots(1,1,layout='constrained')
q *= 100
err *= 100
q.index = [i.replace(' ','\n') for i in q.index]
err.index = [i.replace(' ','\n') for i in err.index]
ax = q.plot.bar(yerr=err,capsize=6,rot=0,
                ax=ax)
ax.set_ylabel('Percentage of current\nwater demand (%)')


fig_out['water_stress'] = plt.gcf()

#%% # REGIONAL HYDROGEN HEAT AND BIOGAS
# what is the scale of biogas potential vs hydrogen vs heat demand
# maybe bar chart but with highlighted (or outlined?) section of Imports
# or callout

# bar chart of different demands and net import for elec and hydrogen

nscens = 4


combal = {}
cols = ['Electricity demand', 'Hydrogen demand', 'Heat demand', 'Biogas supply', 'Net electricity exports', 'Net hydrogen exports']

for w in ws:
    nw = w.split('_')[0].capitalize()
    combal[nw] = {}
    for i,s in enumerate(scens[:nscens]):
    
        lab = s.capitalize()
        if s=='import':
            lab = 'BAU'
        combal[nw][lab] = pd.DataFrame(np.nan, index=ZONES,
                                   columns=cols)
        q = combal[nw][lab]
        #DEMANDS
        q.loc[ISLZONES,'Electricity demand'] = (dem[SCENMAP.loc[s,'demand']]['elec_wo_bio'][w].sum().groupby([i.split('-')[0] for i in dem[SCENMAP.loc[s,'demand']]['elec_wo_bio'][w]]).sum() / 10**9).loc[ISLZONES]
        
        q.loc[ISLZONES,'Hydrogen demand'] = (dem[SCENMAP.loc[s,'demand']]['hyd'][w].sum().groupby([i.split('-')[0] for i in dem[SCENMAP.loc[s,'demand']]['hyd'][w]]).sum() / 10**9).loc[ISLZONES]
        q.loc[ISLZONES,'Heat demand'] = (dem[SCENMAP.loc[s,'demand']]['heat_wo_bio'][w].sum().groupby([i.split('-')[0] for i in dem[SCENMAP.loc[s,'demand']]['heat_wo_bio'][w]]).sum() / 10**9).loc[ISLZONES]
        #SUPPLY
        
        q.loc[ISLZONES,'Biogas supply'] = (dem[SCENMAP.loc[s,'demand']]['bio'][w].sum().groupby([i.split('-')[0] for i in dem[SCENMAP.loc[s,'demand']]['bio'][w]]).sum() / 10**9).loc[ISLZONES]
        #elec
        sw = s+'-'+w
        
        q.loc[ISLZONES,'Electricity supply'] = res[sw]['zonegen'][ISLZONES].sum()/1000
        q.loc[ISLZONES,'Net electricity exports'] = (res[sw]['zoneexports'][ISLZONES].sum() - res[sw]['zoneimports'][ISLZONES].sum())/1000
        q.loc[ISLZONES,'Hydrogen imports'] = res[sw]['gasshortage'].sum().groupby([i.split('_')[1].split('-')[0] for i in res[sw]['gasshortage']]).sum().loc[ISLZONES]/10**3/3.6 # fro m TJ to GWH
        q.loc[ISLZONES,'Hydrogen exports'] = res[sw]['p2xmarketsales'].sum().groupby([i.split('_')[1].split('-')[0] for i in res[sw]['p2xmarketsales']]).sum().loc[ISLZONES]/3.6 # fro m GJ to GWH
        if lab!='BAU':
            q.loc[ISLZONES,'Hydrogen supply'] = res[sw]['p2xload'].sum().groupby([i.split('_')[1].split('-')[0] for i in res[sw]['p2xload']]).sum().loc[ISLZONES]*0.82/10**3
            q.loc[ISLZONES,'Electrolysis demand'] = res[sw]['p2xload'].sum().groupby([i.split('_')[1].split('-')[0] for i in res[sw]['p2xload']]).sum().loc[ISLZONES]/10**3
        else:
            q.loc[ISLZONES,['Hydrogen supply','Electrolysis demand']] = 0
        q.loc[ISLZONES,'Net hydrogen exports'] = q.loc[ISLZONES,'Hydrogen exports'] - q.loc[ISLZONES,'Hydrogen imports']


#%%  DONE ELEC, HYDROGEN AND BIOGAS ENERGY BY SCEN 

cols = ['Electricity demand', 'Electrolysis demand', 'Electricity supply', 'Net electricity exports',
     'Hydrogen demand','Hydrogen supply','Biogas supply','Net hydrogen exports']

elec = ['Electricity demand', 'Electrolysis demand', 'Electricity supply', 'Net electricity exports']
gas = ['Hydrogen demand','Hydrogen supply','Biogas supply','Net hydrogen exports']

neg_cols = [i for i in cols if 'demand' in i]
neg_cols += [i for i in cols if 'Net' in i]

nscens = 4
# do the same in pyplot
fig, axs = plt.subplots(nscens,2,
                        layout='constrained',
                        sharey='col'
                        )

for i, ax in enumerate(axs.flat):
    ax.axhline(0,
               c='k',zorder=0,
               linewidth=0.5,
               label='_nolegend_')
    scen = allscens2[i]
    s = neat_scen(scen,'s')
    if s == 'Import':
        s = 'BAU'
    w = neat_scen(scen,'w').split(' ')[0]
    val = combal[w][s].sum()[cols]
    val[neg_cols] *= -1
    #first do elec values
    elecval = val[elec]
    height = elecval[elec].cumsum()
    height = pd.concat([pd.Series(0,[0]),height])
    for j in range(len(elecval)):
        h = height.iloc[j]
        v = elecval.iloc[j]
        
        z =  elecval.index[j].lower()
        if 'supply' in z:
            c = 'tab:green'
        elif 'demand' in z:
            c = 'tab:red'
        else:
            c = 'tab:blue'
        
        
        bar = ax.bar(x=j,
               height=v,
               bottom=h,
                color=c
               )
        if abs(v)>0:
            ax.bar_label(bar,
                     labels=[abs(round(v,1))],
                     fontsize=10)
    #then a gap
    j+=1
    ax.bar(x=j,height=0,
           width=0.2)
    j+=1
    
    #make second axes for hydrogen
    ax2 = ax.twinx()
    ax2.set_yticklabels([])
    
    # then do hydrogen
    hydval = val[gas]
    height = hydval[gas].cumsum()
    height = pd.concat([pd.Series(0,[0]),height])
    for k in range(len(hydval)):
        l=k+j
        v = hydval.iloc[k]
        h = height.iloc[k]
        
        z =  hydval.index[k].lower()
        labels = [abs(round(v,1))]
        if 'supply' in z:
            c = 'tab:green'
        elif 'demand' in z:
            c = 'tab:red'
        elif s == 'BAU':
            c = 'tab:orange'
            labels=[-round(v,1)]
        else:
            c = 'tab:blue'
            
        bar = ax.bar(x=l,
               height=v,
               bottom=h,
                color=c
               )
        if abs(v)>0:
            ax.bar_label(bar,
                     labels=labels,
                     fontsize=10)
    

    #format graph
    labels = elec + [''] + gas
    labels = [i.replace(' ', '\n') for i in labels]
    labels = [i.replace('Electricity', 'Elec.') for i in labels]
    labels = [i.replace('electricity', 'elec.') for i in labels]
    labels = [i.replace('Hydrogen', 'Hyd.') for i in labels]
    labels = [i.replace('hydrogen', 'hyd.') for i in labels]
    labels = [i.replace('demand', 'dem.') for i in labels]
    labels = [i.replace('supply', 'sup.') for i in labels]
    labels = [i.replace('Electrolysis', 'Elect-\nrolysis') for i in labels]
    if i >= len(axs.flat)-2:
        ax.xaxis.set_ticks(range(len(labels)))
        ax.xaxis.set_ticklabels(labels,
                                fontsize=10)
    else:
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
    
    if i == 0:
        ax.set_title(w)
    if i == 1:
        ax.set_title(w)
    
    if i%2 == 0:
        ax.set_ylim(-350,750)
    else:
        ax.set_ylim(-250,550)
        ax2 = ax.twinx()
        ax2.set_ylabel(s)
        ax2.set_yticklabels([])

fig.supylabel('Monthly total energy (GWh)')

        
fig_out['total_energy_by_scen_waterfall'] = plt.gcf()


#%% HEAT AND BIOGAS SUPPLY

cols = ['Hydrogen demand', 'Heat demand', 'Biogas supply',
     'Hydrogen supply'
      # 'Net hydrogen exports'
     ]
LOWTOHIGHGEN = ['HI', 'AB', 'NA','OR','SH','LH']

nscens = 2
fig, axs = plt.subplots(nscens,1,
                        layout='constrained',
                        # sharey='col'
                        )

w = 'Winter'
for i,s in enumerate(scens[1:]):
    if s!='middle':
        ax = axs[i]

        ax.axvline(2.5,
                   linewidth=1,
                   c='k',
                   linestyle='--')
        lab = neat_scen(s,'s')
        if lab=='Import':
            lab = 'BAU'

        combal[w][lab].loc[LOWTOHIGHGEN,cols].plot.bar(ax=ax,legend=False, 
                                      width=0.9
                                    )
        if i != nscens:
            ax.set_xticklabels([])
        
        ax2 = ax.twinx()
        if 'import' in s:
            s = 'BAU'
        else:
            s = neat_scen(s,'s')
        ax2.set_ylabel(s,
                          size=20
                          )
        ax2.set_yticks([])

axs[0].text(1,40,'Low generation',ha='center',fontsize=15,style='italic')
axs[0].text(4,40,'High generation',ha='center',fontsize=15,style='italic')

handles = []
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
                'tab:brown']
for i,j in zip(cols,
                colors[:len(cols)]):
    if '(less heat)'in i:
        i = i.replace(' (less heat)','')
    handles.append(mpatch.Patch(label=i,color=j))
    
fig.legend(handles=handles,
            ncol=combal[w]['BAU'].shape[1],
            loc='center',
            bbox_to_anchor=(0.5,0.5),
            fontsize=15)

axs[-1].set_xticklabels([ZONEMAP[i] for i in LOWTOHIGHGEN],
                        rotation=0,
                        size=15)
    
fig.supylabel('Energy (GWh)')

fig_out['heat_biogas_and_hydrogen_compare'] = plt.gcf()


#%% 


# have a think about what to say about flexibikity and storage
# what is happening when the flexibility is operating? 
# it reduces line stress by keeping electricity in the same node- so maybe for 
# the peak stress periods look at whats happening across all scenarios (line load in MW)
# DEFINE PEAK STRESS AS WHENEVER THE HYDROGEN GEN IS RUNNING (28% OF WINTER PERIOD)

cols = ['total line flow','zone line flow','mainland line flow']

flex = pd.DataFrame(np.nan, 
                    index=scens,
                    columns=cols
                    )
w = 'winter_low_wind_high_demand'
ss = [i+'-'+w for i in scens]
cs = ['tab:blue','tab:orange','tab:green','tab:red']

fig, axs = plt.subplots(4,1,
                        layout='constrained')

for i, s in enumerate(ss):
    
    ## NODES WITH NO GEN
    NOGEN = [i for i in NODES if not any([i in j for j in res[s]['maxcap']])]
    
    ##NODES WITH NO GEN OR P2X
    # NOP2X = [i for i in NODES if not any([i in j for j in res[s]['p2xcap']])]
    # NOGEN = [i for i in NOGEN if i in NOP2X]
    # NOGEN = ['AB-24', 'HI-18', 'LH-15', 'NA-28', 'NA-29', 'NA-31']
    NOGENLINES = [i for i in res[s]['lineflow'].columns if any([j in i for j in NOGEN])]
    # # times of mainland stress - KEEP IT SIMPLE
    q = res[s]['gen']['HYG_MA-0']>0    
    ax = axs[i]
    r = res[s]['lineflowdirection'][q][NOGENLINES].mean()
    p = res[s]['linemaxcap'][NOGENLINES].mean()
    z = res[s]['lineflowdirection'][q][NOGENLINES]/res[s]['linemaxcap'][NOGENLINES]
    z = pd.Series(np.ravel(z)).dropna()*100
    hist_density_plot(z,np.linspace(-100,100,51),True,
                      cs[i],ax)
    
    #average line
    height = ax.get_ylim()[1]
    ax.axvline(x=z.mean(), lw=3, color='k', linestyle='--',
                        zorder=0,alpha=0.5)
    t = ax.annotate(str(int(z.mean()))+'%',
                [z.mean()+3,height*0.7],
                c='k',
                ha='left')
    #zero line
    ax.axvline(x=0, lw=1, color='k',
                        zorder=0,
                        alpha=0.5,linestyle='--')
    ax.set_xlim(-100,100)
    
    # ylabel
    ax2= ax.twinx()
    if 'import' in s:
        lab = 'BAU'
    else:
        lab = neat_scen(s,'s')
    ax2.set_ylabel(lab, size=18)
    ax2.set_yticklabels([])
    
    #import export value
    v = 75
    val = round((z<-v).sum()*100/len(z),1)
    ax.annotate(#f'<-{v}%:\n{val}%',
                f'{val}%',
                    [-v, height*0.5],
                    size=18,
                    style='italic',
                    c='r',
                    ha='right')
    
    ax.fill_between([-100,-v],height,
                     color='r',
                     alpha=0.1,
                     hatch='x'
                     )
    
    val = round((z>v).sum()*100/len(z),1)
    ax.annotate(#f'>{v}%:\n{val}%',
                f'{val}%',
                    [v, height*0.5],
                    size=18,
                    style='italic',
                    c='r',
                    ha='left')
    ax.fill_between([v,100],height,
                     color='r',
                     alpha=0.1,
                     hatch='x'
                     )
    ax.set_ylim(0,height)
    
    
    
fig.supylabel('Proportion of the time')

fig_out['flexibility-non_gen_peak_stress_line_load'] = plt.gcf()

#%% COST OF ELECTRICITY???

# SEEMS LIKE THE ADDITIONAL WIND HAS A LARGER EFFFECT ON THE PRICE THAN
#THE STORAGE OR FLEXIBILITY DUE TO THE SCALE 1GW VS 100MW ISH


########## ONLY WITHOUT ELECTROLYSIS NOW
cols = ['dem', # both in GWh
        'average price',
        'total cost'
        ]

cost = pd.DataFrame(np.nan,
                    index=allscens2,
                    columns=cols
                    )

# seems like price is 6000 in some cases where there is 0 unserved energy-
# pretty sure this is from the DSR which wasnt reporting demand properly

for s in res:
    P2XNODES = [i for i in res[s]['p2xload'] if 'MA' not in i]
    r = res[s]['dem'][ISLNODES].sum(axis=1) - res[s]['p2xload'].loc[:,P2XNODES].sum(axis=1) 
    # r = r[r>0]
    p = res[s]['price']
    p[p>500] = 0
    p = p.loc[r.index].values.squeeze()
    cost.loc[s,'dem'] = r.sum()

    q = (r * p).sum()
    cost.loc[s,'average price'] = q/cost.loc[s,'dem']
    cost.loc[s,'total cost'] = q / 10**6
    cost.loc[s,'dem'] /= 1000 # GWH

cost = cost.round(1)


q = [i  for i in allscens2 if 'winter' in i]+[i  for i in allscens2 if 'summer' in i]
cost.loc[q,['average price','total cost']].to_clipboard(sep=',',excel=True)


#%% LINES now by capacity not load- with nicer hist plots


# using the ST.Line Flow.csv look at the flow in terms of flow towards the 
# mainland (make dict of line pairs to get the polarity of the direction right)
# proportion of time exporting vs importing and load

fig,axs = plt.subplots(4,2,
                        layout='constrained')


limit = {'max':90,
         'min':40}

line_loading_periods = pd.DataFrame(np.nan, index=allscens2,
                                    columns=[f'>{limit["max"]}%',f'<{limit["min"]}%'])
imp_exp = pd.DataFrame(np.nan, index=allscens2, 
                       columns=['Import','Export'])
xs = []

for i,s in enumerate(allscens2):
    j = 1
    if 'summer' in s:
        j = 0
    i = (i - j ) // 2
    ax = axs[i,j]
    
    if i == 0:
        w = s.split('-')[1]
        ax.set_title(WRENAME[w],size=18)
    if j == 1:
        ax2= ax.twinx()
        if 'import' in s:
            lab = 'BAU'
        else:
            lab = neat_scen(s,'s')
        ax2.set_ylabel(lab, size=18)
        ax2.set_yticklabels([])
    # ax.set_xticklabels([-100,-50,0,50,100])
    if j>0:
        ax.set_yticklabels([])
    if i!=3:
        ax.set_xticklabels([])
    q = pd.Series(np.ravel(res[s]['lineload']))
    # FIND OUT WHAT UNDER/OVERUTILISED IS FOR TRANMISSION, THEN MAKE TABLE
    line_loading_periods.loc[s,f'>{limit["max"]}%'] = round((q>limit['max']).sum()/len(q)*100,1)
    line_loading_periods.loc[s,f'<{limit["min"]}%'] = round((q<limit['max']).sum()/len(q)*100,1)

    nbins = 51
    # # this is net flow grouped by zone
    # bins = np.linspace(-200,800, nbins)
    # q = res[s]['lineflowdirection'].groupby([i.split('_')[0].split('-')[0] for i in res[s]['lineflowdirection']],axis=1).sum()
    # q = pd.Series(np.ravel(q))
    
    # this is net flow grouped by groups of islands (ie islay/jura, orkney , etc)
    bins = np.linspace(-100,100, nbins)
    q = res[s]['lineflowdirection'].groupby([NODEGROUPS[i.split('_')[0]] for i in res[s]['lineflowdirection']],axis=1).sum()
    q /= res[s]['linemaxcap'].groupby([NODEGROUPS[i.split('_')[0]] for i in res[s]['linemaxcap']],axis=1).sum()
    q *= 100
    q = pd.Series(np.ravel(q))
        
    hist_density_plot(q, bins, False, cs[i], ax)
    
    height = ax.get_ylim()[1]
    ax.axvline(x=q.mean(), lw=3, color='k', linestyle='--',
                        zorder=0,alpha=0.5)
    t = ax.annotate(str(int(q.mean()))+'%',
                [q.mean()+10,height*0.7],
                c='k',
                ha='left')
    xs.append(ax.get_xlim())
    ax.axvline(x=0, lw=1, color='k',
                        zorder=0,
                        alpha=0.5,linestyle='--')

    
xmin = 0
xmax = 0
for x in xs:
    if x[0] < xmin:
        xmin = x[0]
    if x[1] > xmax:
        xmax = x[1]

for i,ax in enumerate(axs):
    # ax[0].set_xlim(-200,800)
    # ax[1].set_xlim(-100,500)
    
    ax[0].set_xlim(-100,100)
    ax[1].set_xlim(-100,100)
    ylim1 = ax[0].get_ylim()[1]
    ylim2 = ax[1].get_ylim()[1]
    if ylim2 > ylim1:
        ax[0].set_ylim(0,ylim2)
    else:
        ax[1].set_ylim(0,ylim1)
    height = 0.9
    
    
    for j,a in enumerate(ax):
        ylim = a.get_ylim()[1]
        xlim = a.get_xlim()
        if i==0:      
            
            a.annotate('<-Import',
                [-10, ylim*height],
                size=15,
                ha='right')
            a.annotate('Export->',
                [10,ylim*height],
                size=15,
                ha='left')


        ind = i*2
        if j==1:
            i += 1

fig.supylabel('Proportion of time periods')
        
fig_out['line_load_hist_by_scen_directional'] = plt.gcf()
 

#%% line loading periods plot

fig,axs = plt.subplots(2,1,
                       layout='constrained')
 
for i, w in enumerate(ws):
    ss = [i for i in allscens2 if w in i]
    new = line_loading_periods.loc[ss]
    new.index = [i.split('-')[0].capitalize() for i in new.index]
    new.index = [i if i!='Import' else 'BAU' for i in new.index ]
    new.plot.bar(ax=axs[i],
                 rot=0,
                 legend=False)
    if i==0:
        axs[i].set_xticklabels([])
    axs[i].annotate(WRENAME[w],
                    [-0.25,axs[i].get_ylim()[1]*0.9])

axs[0].legend(line_loading_periods.columns)
fig.supylabel('Proportion of time (%)')
    
fig_out['line_utilisation_proportions'] = plt.gcf()



#%% # look at electrolysis ramp rate and LOAD FACTOR- briefly

fig, axs = plt.subplots(2,2, layout='constrained',
                        sharey=True)
p2x_ramp = {'High generation':[],
            'Low generation':[]}



ramp_prop=50
ramp_lab = f'>{ramp_prop}% ramp proportion of time'
for j in p2x_ramp:
    p2x_ramp[j] = pd.DataFrame(np.nan,index=[i for i in allscens2 if 'import' not in i],
                            columns=['Load factor','Mean ramp (when >0)'])
    for s in p2x_ramp[j].index:
        
    
        ISLP2X = [i for i in res[s]['p2xload'] if 'MA-0' not in i]
        
        if j == 'Low generation':
            ISLP2X = [i for i in ISLP2X if  any([j in i for j in LOWGENZONES])]
        else:
            ISLP2X = [i for i in ISLP2X if  not any([j in i for j in LOWGENZONES])]
        p2x_ramp[j].loc[s,'Load factor'] = res[s]['p2xcapfactor'][ISLP2X].mean().mean()
        # p2x_ramp[j].loc[s,'Ramp'] = res[s]['p2xcapfactor'][ISLP2X].diff().abs().mean().mean()
        
        q = res[s]['p2xcapfactor'][ISLP2X].diff().abs()
        q = q[q!=0].mean()
        q = (q * res[s]['p2xcap'][ISLP2X].mean()).sum() / res[s]['p2xcap'][ISLP2X].mean().sum()
        p2x_ramp[j].loc[s,'Mean ramp (when >0)'] = q
        
        p2x_ramp[j].loc[s, ramp_lab] = (res[s]['p2xcapfactor'][ISLP2X].diff().abs()>ramp_prop).sum().sum()/len(np.ravel(res[s]['p2xcapfactor'][ISLP2X]))
        p2x_ramp[j].loc[s, ramp_lab] *= 100

for i, ax in enumerate(axs.flat):
    k = 'Low generation'
    if i%2 == 0:
        k = 'High generation'
    
    q = p2x_ramp[k]
    
    w = 'winter'
    if i//2 == 0:
        w = 'summer'
    j = [i for i in q.index if w in i]
        
    q.loc[j,:].plot.bar(ax=ax,rot=0,legend=False,
                        secondary_y=ramp_lab)
    ax.right_ax.set_ylim(0,8)    
    if i%2 > 0:
        ax.right_ax.set_yticklabels([0,2,4,6,8],
                                color='tab:green')
    else:
        ax.right_ax.set_yticklabels([])
    if i>=2:
        ax.set_xticklabels([neat_scen(i,'s') for i in j])
    else:
        ax.set_xticklabels([])
    
    
    if i==0:
        ax.set_ylabel('Summer (high wind)')
        ax.set_title('High generation capacity')
    if i==2:
        ax.set_ylabel('Winter (low wind)')
    if i==1:
        ax.set_title('Low generation capacity')
    
    bars = [i for i in ax.containers if isinstance(i, BarContainer)]
    bars += [i for i in ax.right_ax.containers if isinstance(i, BarContainer)]
    for bar in bars:
        lab = bar.get_label()
        if '(right)' in lab:
            lab = lab.replace(' (right)','')
            new = q.loc[j,lab].round(1)
            new = [f'{i}%' for i in new]    
            ax.right_ax.bar_label(bar,
                        labels=new,
                        fontsize=15,
                        color='tab:green'
                        )
        else:
            new = q.loc[j,lab].astype(int)
            new = [f'{i}%' for i in new]    
            ax.bar_label(bar,
                        labels=new,
                        fontsize=15
                        )



handles = []
for i,j in zip(p2x_ramp[k].columns,['tab:blue','tab:orange','tab:green']):
    if 'proportion' in i:
        i = f'{i} (right axis)'
    handles.append(mpatch.Patch(label=i,color=j))
    
fig.legend(handles=handles,
           ncol=p2x_ramp[k].shape[1],
           loc='center',
           fontsize=12)

fig_out['electrolysis_ramp_rates'] = plt.gcf()

#%% HEATMAP OF NETWORK LOADING

# https://stackoverflow.com/questions/10388462/matplotlib-different-size-subplots
ss = [i for i in allscens2 if 'winter' in i]

# mainfig = plt.figure()
mainfig = plt.figure(layout='constrained')
subfigs = mainfig.subfigures(2,2)

for j, (s,fig) in enumerate(zip(ss,subfigs.flat)):
    axs = fig.subplots(6,1,
                        gridspec_kw={'height_ratios': [sum([j in i for i in NODES])/len(ISLNODES) for j in ISLZONES]}
                        )
    
    # plt.subplots_adjust(wspace=0.0001)
    if j//2 < 1:
        fig.suptitle(neat_scen(s,'s'))
    else:
        fig.supxlabel(neat_scen(s,'s'))

    for i, (z,ax) in enumerate(zip(ISLZONES,axs)):
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ax.tick_params(axis='y', which='both', bottom=False, top=False)
        q = res[s]['lineload']
        q = q[[i for i in q if z in i]]
        q = q.groupby(pd.Grouper(freq='D')).mean()
        sns.heatmap(q.T, ax=ax,
                    vmin=0, vmax=100,
                    cbar=False)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        if j%2 == 0:
            ax.set_ylabel(z)
            
            
#%% look at peak demand with flexibility

z = [i for i in SS if 'winter' in i]
d = pd.DataFrame(0,index=z,columns=['peak'])

for s in z:
    ISLP2X = [i for i in res[s]['p2xload'] if 'MA' not in i]
    q = res[s]['dem'][ISLNODES].sum(axis=1) - res[s]['p2xload'][ISLP2X].sum(axis=1)
    d.loc[s,'peak'] = q.max()
    d.loc[s,'mean'] = q.mean()



#%% write graphs to files
portrait = ['island_electrolysis_by_region',
            'line_load_hist_by_scen_directional']

taller = ['island_gen_by_scen','total_energy_by_scen_waterfall',
          'flexibility-non_gen_peak_stress_line_load']

IMAGEFOLDER = r'C:\Users\Chris\Desktop\THESIS\Images and graphs\net zero model images'
for i in fig_out:
    OUT = os.path.join(IMAGEFOLDER ,i+'.jpg')
    fig_out[i].set_size_inches(1920/160, 1080/160)
    if i in portrait:
        fig_out[i].set_size_inches(1420/160, 1580/160)
    if i in taller:
        fig_out[i].set_size_inches(1920/160, 1580/160)
    fig_out[i].savefig(OUT)