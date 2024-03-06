# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:02:30 2023
@author: Chris

1	GAS DEMAND SHORTAGE PRICE NEEDS TO BE FOR WIND ONLY- NOT MORE EXPENSIVE GENERATION
    a.	Currently set to £27/GJ, but should be for wind- £50/MWh @82% = £17/GJ
2	GAS DEMAND IS CURRENTLY ENTIRELY SHORT- WHY?? MAYBE UNITS IN DEMAND ARE WRONG AGAIN
    a.	Was caused by shortage price being the same as the market price
3	Number of units seems to make the model go crazy- adjusting so there are more units increases the curtailed generation (including thermal??), unserved energy, etc 
    a.	Have checked that there is still the same amount of generation in the system and all generator properties are the same but doesn’t seem to be any difference.
    b.	Also much slower with more units
4	Fuels need to be defined or else thermal generation will report capacity curtailed
5	Use highest resolution in MT model- 
    a.	Number of steps in LDC cant be more than number of intervals (hours in days- duh)
    b.	Don’t think there is any problem with this run, but not sure why fixed costs aren’t being passed on
6	Baseload generation has been set as fixed load so it always runs- BEC, NUC, HYD
    a.	This represents the long term contracts which dictate dispatch for these generations, makes sure that wind is the generation which is curtailed 
7	All generation LCOE is just set at the VOM charge, markup did not seem to behave as expected AT ALL
8	To induce some more price variation (needed for storage and P2X to run), VOM costs for wind have been set to decrease when there is a lot of wind on the system and increase vice versa (price cannibalisation, can find ref if needed). The inverse has been set for GCC- when there is more demand, they will bid higher.
9	Mainland P2X is set without a VOM charge, so the only loss of price is in the efficiency losses
    a.	It is subject to a storage target of the same as it started, which makes it so that the P2X tries to replenish the storage (otherwise there is no incentive and all the demand comes out of the storage)
    b.	The dispatch of the P2X is dictated by the gas demand shortage cost, withdrawal charge and recycle penalty (applied as a penalty to not achieving the recycle rate)
    c.	To force the P2X to run, the withdrawal charge must be less than the shortage or recycle penalty (otherwise the penalty is cheaper than meeting demand)
    d.	If the penalty is too high, it will force the P2X to generate from all generation, including GCC, which doesn’t make sense
    e.	To get around this, the withdrawal charge is set artificially low (just below the shortage penalty)
    i.	Although this is cheaper than the assumed £64/GJ price of hydrogen, it is the only way to force the P2X to behave as expected
    f.	This has no impact on the island systems (which although they might run while GCC is generating on the mainland, this should be small) due to the mainland gas demand/nodes having no connection.


"""

import os
import pandas as pd
import subprocess
from shutil import copytree
from datetime import datetime
import matplotlib.pyplot as plt
import chime
from plexosapi import *

#try except statements so the master file runs with those settings

# set to true to use local file paths for datafiles
TROUBLESHOOT = False
# whether or not mainland scenarios are included for generatoin and demand
MAINLAND_SCEN = False
#whether to include dayahead/intraday model
DAYAHEAD = False

#%% FUNCTIONS

def run_model(plexospath, filename, foldername, modelname):
    
    """launch the model on the local desktop
    The \n argument is very important because it allows the PLEXOS
    engine to terminate after completing the simulation"""

    subprocess.call([os.path.join(plexospath, 'PLEXOS64.exe'), filename, 
                     r'\n', r'\o', foldername, r'\m', modelname])
    
def check_if_open(folder):
    """ running model in PLEXOS will fail without explanation if an output
    file is still open in excel, so this checks for that and informs
    which file is still open to close"""
    count = 1
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(subdir, file)
            if 'csv' in file:
                try:
                    new = os.path.join(subdir,f'asdsa{count}.csv')
                    os.rename(path, new)
                    os.rename(new, path)
                    
                except:
                    print(f'FILE {path} IS STILL OPEN DUMMY')
                    chime.error()
                    input('press enter when it is closed')
                count +=1
    print('Check for open files completed uwu')
    
def change_scen_run(file, scens):
    
    """ runs the PLEXOS fiel for every scenario in the list scens, with output
    data copied from each run into an output folder, before it is deleted by
    the next model run 
    
    Need to make sure scenarios are properly defined so that the model runs
    properly, if there is an error in the file it will not show up here, 
    only in the results folder error logx"""
    current_time = datetime.now().strftime("%Y.%m.%d, %H.%M.%S")
    current_time += ' ' + input('enter a quick description of the run and changes')
    OUTFOLDER = os.path.join(OUT, current_time)
    os.makedirs(OUTFOLDER, exist_ok=True)
    
    for scen in scens:
        #setup connection
        db = DatabaseCore()
        db.Connection(file ) 
        
        #get old scenarios
        old_scens  = db.GetMemberships(CollectionEnum.ModelScenarios)
        models = [i.split('( ')[1].split(' )')[0].replace(' ','') for i in old_scens]
        old_scens = [i.split('( ')[2].split(' )')[0].replace(' ','') for i in old_scens]
        
        #remove old ones and add new ones- only if the old models are relevant
        for old_scen in old_scens:
            if old_scen != 'ID' and old_scen != 'DA':
                db.RemoveMembership(CollectionEnum.ModelScenarios, MODELNAME, old_scen)
            if DAYAHEAD:
                db.RemoveMembership(CollectionEnum.ModelScenarios, 'Intraday', 'ID_'+old_scen)
                db.RemoveMembership(CollectionEnum.ModelScenarios, MODELNAME, 'DA_'+old_scen)

        #add back new ones
        if DAYAHEAD:
            for model, suff in zip(['Intraday','DayAhead','DayAhead','Intraday'],
                                    ['', '', 'DA_', 'ID_']):
                S = suff+scen
                # print(S)
                db.AddMembership(CollectionEnum.ModelScenarios,model,S)
        else:
            db.AddMembership(CollectionEnum.ModelScenarios, MODELNAME, scen)
            
        db.Close()
        print(f'EXECUTING MODEL: {MODELNAME} WITH SCENARIO: {scen} uwu\n{current_time}')
        
        for i in SCENFOLDERS:
            check_if_open(SCENFOLDERS[i])

        run_model(PLEXOSPATH, file, '.', MODELNAME)
        LOG = os.path.join(FOLDERMAIN, FOLDER, f'Model {MODELNAME} Solution', 
                    f'Model ( {MODELNAME} ) Log.txt' )
        FAILEDLOG = LOG = os.path.join(FOLDERMAIN, FOLDER, 
                    f'~Model ( {MODELNAME} ) Log.txt' )
        try:
            if os.path.exists(FAILEDLOG):#
                with open(FAILEDLOG) as L:
                    L = L.read()
                    print(L)
                chime.error()
                raise Exception('THIS PLEXOS MODEL HAS FAILED')
        except:
            with open(LOG) as L:
                L = L.read()
                MESSAGE = 'Error and Warning Messages for Model'
                if MESSAGE in L:
                    print(L[L.index(MESSAGE):])
        print('PLEXOS MODEL COMPLETED\n\n')  
        
        #copy outputs to new folder
        OUTFOLDERSCEN = os.path.join(OUTFOLDER, scen)
        for i in SCENFOLDERS:
            copytree(SCENFOLDERS[i],
                  os.path.join(OUTFOLDERSCEN,i))
    return OUTFOLDER


def make_diff(name, scen):
    return out['Intraday'][scen][name] - out['DayAhead'][scen][name]

def get_all_scens(name, model=None, op=False,axis=False):
    """group PLEXOS outputs into dataframe"""
    if model:
        q = out[model]
    else:
        q = out
    if not axis:
        axis=1
    if not op:
        x = pd.concat([q[i][name] for i in q],axis=1)
    elif op == 'sum':
        x = pd.concat([q[i][name].sum(axis=1) for i in q],axis=1)
    elif op == 'mean':
        x = pd.concat([q[i][name].mean(axis=1) for i in q],axis=1)
    elif op == 'min':
        x = pd.concat([q[i][name].min(axis=1) for i in q],axis=1)
    elif op == 'max':
        x = pd.concat([q[i][name].max(axis=1) for i in q],axis=1)
        
    if x.shape[1] == len(q.keys()):
        x.columns = [i for i in q]
    else:
        ind1 = []
        ind2 = []
        for S in MASTER_SCENS:
            for i in q[S][name].columns:
                ind1.append(S)
                ind2.append(i)
        x.columns = pd.MultiIndex.from_arrays([ind1, ind2],
                                               names=['scen','col'])
    return x

    
    
#%% INPUT DATA

FOLDERMAIN = os.getcwd() 
        
FILES = {
         1: '1.elec_model',
         2: '2.power2x_model',
         3: '3.storage_model',
         4: '4.heat_model'
         }

year = 2045
start_end = {'winter_low_wind_high_demand': [pd.to_datetime('01/12/'+str(year), format='%d/%m/%Y'),
                                             pd.to_datetime('29/12/'+str(year),format='%d/%m/%Y')],
             
              'summer_low_wind_high_demand': [pd.to_datetime('17/07/'+str(year), format='%d/%m/%Y'),
                                              pd.to_datetime('14/08/'+str(year), format='%d/%m/%Y')]#,
             
              # 'summer_high_wind': [pd.to_datetime('01/04/'+str(year), format='%d/%m/%Y'),
              #                      pd.to_datetime('29/04/'+str(year), format='%d/%m/%Y')]
                  }

date_range = dict(zip(ws,
                      [pd.date_range(start_end[i][0], start_end[i][1] ,
                                     inclusive="left",
                                     freq="H") for i in start_end]))
        
#%%  Set up which folder to run and post-process

# this is used in the files below as the base file to build on
BASE = os.path.abspath(r'Set up and blank file\blank_file_v3 - 10.0 - 1.11.23.xml')

NUM = 4
"""IT DOESNT WORK RUNNING MISSING ONE OUT DUE TO BEING BASED OFF THE OLD
VERSION OF THE NUMBER BEFORE- COULD UPDATE BUT CBA"""
NUM = [i+1 for i in range(NUM)]
if 3 not in NUM:
    input('NOTE THAT MAINLAND DEMAND PROFILES HAVE BEEN CHANGED TO REMOVE DSR NOW, press ener if this is OK')
    # raise Exception('NOTE THAT MAINLAND DEMAND PROFILES HAVE BEEN CHANGED TO REMOVE DSR NOW')

FOLDER = FILES[max(NUM)]

for i in NUM:
    MODEL = FILES[i] + '.py'
    print(f'ADDING THE {MODEL} DATA TO PLEXOS XML FILE')
    # this way runs the scripts within this folder
    with open(MODEL) as f:
        exec(compile(f.read(), MODEL, 'exec'))
        
print('PLEXOS UPDATE FILES FINISHED RUNNING\n\n')
        
# %% Execute the PLEXOS model

FILE = os.path.join(FOLDERMAIN, FOLDER, FOLDER+'.xml')
if DAYAHEAD:
    MODELNAME = 'DayAhead'
    SCENFOLDERS = ['DayAhead', 
               'Intraday',
               'Intraday ONLY'
               ]
else:
    MODELNAME = 'Intraday ONLY'
    SCENFOLDERS = ['Intraday ONLY']

SCENFOLDERS = dict(zip(SCENFOLDERS, 
                       [os.path.join(FOLDERMAIN, FOLDER, f'Model {i} Solution') for i in SCENFOLDERS]))

OUT = r'C:\Users\Chris\Desktop\Coding\Python\PLEXOS\PLEXOS API\Output analysis'

MASTER_SCENS = allscens # ONLY 2 WEATHER SCENS NOW
DOWNLOAD_OLD = False

#THIS RUNS ALL THE SCNEARIOS FOR THE GIVEN FILE
OUTFOLDER = change_scen_run(FILE, MASTER_SCENS)


# %% DO SOME DA ID COMPARISON

# ONLY IF LOOING AT OLD RESULTS
# define OUTFOLDER if the section above has not run
# DOWNLOAD_OLD = True
# DATEFOLDER = '2023.12.20, 12.14.52first run nodal p2x_v2'
# OUTFOLDER = os.path.join(OUT, DATEFOLDER)

OUTRES = os.path.join(OUTFOLDER, 'DA_ID_Diff_outputs.pickle')

GENS = ['GCC_MA-0', 'BEC_MA-0', 'PST_MA-0', 'NUC_MA-0', 'REN_MA-0', 'HYD_MA-0',
       'ONS_AB-23', 'ONS_AB-25', 'ONS_HI-20', 'ONS_LH-12', 'ONS_LH-13',
       'ONS_LH-14', 'ONS_LH-17', 'ONS_MA-0', 'ONS_OR-1', 'ONS_OR-2',
       'ONS_OR-3', 'ONS_OR-4', 'ONS_OR-5', 'ONS_SH-10', 'ONS_SH-6', 'ONS_SH-8',
       'ONS_SH-9', 'OFF_MA-0', 'SOL_MA-0', 'MAR_MA-0', 'MAR_OR-1']

if 3 in NUM:
    GENS.append('HYG_MA-0')

RENGEN = [i for i in GENS if not any([j in i for j in ['GCC', 'BEC', 'NUC',
                                                        'PST']])] # these are thermal since can be dispatched
THERMGEN = [i for i in GENS if i not in RENGEN]
ISLGEN = [i for i in GENS if 'MA-0' not in i ]
MAGEN = [i for i in GENS if 'MA-0' in i ]

GENMAP = dict(zip(GENS,['REN' if i in RENGEN else 'THERM' for i in GENS ]))

if DOWNLOAD_OLD:
    out = pd.read_pickle(OUTRES)
else:
    if DAYAHEAD:
        out= {'DayAhead':dict(zip(MASTER_SCENS,[{} for i in range(len(MASTER_SCENS))])),
          'Intraday':dict(zip(MASTER_SCENS,[{} for i in range(len(MASTER_SCENS))])),
          'Diff':dict(zip(MASTER_SCENS,[{} for i in range(len(MASTER_SCENS))]))
          } 
    else:
        out = dict(zip(MASTER_SCENS,[{} for i in range(len(MASTER_SCENS))]))

DIFF_FILES = {#GEN
              'gen': 'ST Generator.Generation.csv',
              'nodegen': 'ST Node.Generation.csv',
              'fixedload': 'ST Generator.Fixed Load Generation.csv',
              'fixedloadviolation': 'ST Generator.Fixed Load Violation.csv',
              'gencurtailed': 'ST Generator.Capacity Curtailed.csv',
              'maxcap': 'ST Generator.Installed Capacity.csv',
              'plexoscapfactor': 'ST Generator.Capacity Factor.csv',
              'cap': 'ST Generator.Installed Capacity.csv',
              'curtailed': 'ST Generator.Curtailment Factor.csv',
              'undispatched': 'ST Generator.Undispatched Capacity.csv',
              'forcedout': 'ST Generator.Forced Outage.csv',
              'outage': 'ST Generator.Outage.csv',
              'zonegencurtailed':'ST Zone.Generation Capacity Curtailed.csv',
              
              #LINE
              'lineflow':'ST Line.Flow.csv',
              'linecongestion':'ST Line.Hours Congested.csv',
              'lineload':'ST Line.Loading.csv',
              'linemaxcap':'ST Line.Export Limit.csv',
              
              #DEM
              'dem': 'ST Node.Load.csv',
              'export':'ST Node.Exports.csv',
              'unserved': 'ST Node.Unserved Energy.csv',
              'zoneunserved':'ST Zone.Unserved Energy.csv',
              'zoneload':'ST Zone.Load.csv',
              'zoneexports': 'ST Zone.Exports.csv',
              'zoneimports': 'ST Zone.Imports.csv',
              'nodeexports': 'ST Node.Exports.csv',
              'nodeimports': 'ST Node.Imports.csv',
              
              #PRICES
              'price': 'ST Region.Price.csv',
              'zoneprice':'ST Zone.Price.csv',
              'srmc':'ST Generator.SRMC.csv',
              'offerbase': 'ST Generator.Offer Base.csv',
              
              }

if 1 in NUM:
    NEW = {
            'gasshortage':'ST Gas Demand.Shortage.csv',
            'gasserved':'ST Gas Demand.Served Demand.csv', 
            'gasprice': 'ST Gas Node.Delivered Price.csv',
            'gasdemand':'ST Gas Demand.Demand.csv', 
            
            #STORAGE
            'gasstorageutilisation':'ST Gas Storage.Utilization.csv',
            'gasstorageinjection':'ST Gas Storage.Injection.csv',
            'gasstoragewithdrawal':'ST Gas Storage.Withdrawal.csv',
            'gasstoragevolume':'ST Gas Storage.End Volume.csv',
            'gasstorage2demand':'ST Gas Demand.Source Gas Storages.Delivered Quantity.csv',
            
            
            #POWER2X
            'p2xload':'ST Power2X.Load.csv',
            'p2xprod':'ST Power2X.Production Rate.csv',
            'p2xtodemand':'ST Gas Demand.Source Power2X.Delivered Quantity.csv',
            'p2xcapfactor':'ST Power2X.Capacity Factor.csv',
            'p2xcap': 'ST Power2X.Installed Capacity.csv',
            
            #MARKETS (for excess hydrogen)
            'p2xmarketsales':'ST Gas Node.Markets.Sales.csv',
            'p2xmarketpurchases':'ST Gas Node.Markets.Purchases.csv'
            
            }
    
    DIFF_FILES.update(NEW)
    
if 2 in NUM:
    NEW = {
            'battenergy':'ST Battery.Energy.csv',
            'battload':'ST Battery.Load.csv',
            'battgen':'ST Battery.Generation.csv',
            'battcap': 'ST Battery.Generation Capacity.csv',
            'battcharge':'ST Battery.Charging.csv',
            'battdischarge':'ST Battery.Discharging.csv',
            'dsrdeferred':'ST Charging Station.Deferred Load.csv',
            'dsrcharging':'ST Charging Station.Charging.csv'}
    DIFF_FILES.update(NEW)
    
if 3 in NUM:
    NEW = {
            'heatdemand':'ST Heat Node.Heat Demand.csv',
            'heatwithdraw':'ST Heat Node.Heat Withdrawal.csv',
            'heatprod':'ST Heat Plant.Heat Production.csv'
            }
    DIFF_FILES.update(NEW)

for S in MASTER_SCENS:
    NODECURT = True
    for C, F in DIFF_FILES.items():
        for D in SCENFOLDERS:
            PATH = os.path.join(OUTFOLDER,S,D,'Interval',F)
            if DAYAHEAD:
                q = out[D][S]
            else:
                q = out[S]
        
            if os.path.exists(PATH):
                q[C] = pd.read_csv(PATH,    
                                        index_col=0) 
            else:
                print(f'{C} is missing for {S}')
                # q[C] = np.nan
                
            if DAYAHEAD:
                out['Diff'][S][C] = make_diff(C,S)
            #THIS WONT WORK FOR DAY AHEAD
            # if NODECURT and 'gencurtailed' in out[S]:
            if C == 'srmc':
                out[S]['nodegencurtailed'] = out[S]['gencurtailed'].groupby([i.split('_')[1] for i in out[S]['gencurtailed'].columns],axis=1).sum()
                out[S]['zonecurtialed'] = out[S]['nodegencurtailed'].groupby([i.split('-')[0] for i in out[S]['nodegencurtailed'].columns],axis=1).sum()
                out[S]['zonegen'] = out[S]['nodegen'].groupby([i.split('-')[0] for i in out[S]['nodegen'].columns],axis=1).sum()
                out[S]['nodemaxcap'] = out[S]['maxcap'].groupby([i.split('_')[1] for i in out[S]['maxcap'].columns],axis=1).sum()
                out[S]['zonemaxcap'] = out[S]['nodemaxcap'].groupby([i.split('-')[0] for i in out[S]['nodemaxcap'].columns],axis=1).sum()
                #[j for i in out[S]['lineload'].columns for j in ISLZONES if j in i]
                # out[S]['zonelineload'] = out[S]['lineflow'].abs().groupby()
                NODECURT = False

#%% add in the updated dates (not just 1st Jan) 

for S in out:
    for N in out[S]:
        W = S.split('-')[1]
        out[S][N].index = date_range[W]

#%%

if DAYAHEAD:
    
    for S in MASTER_SCENS:
        for D in SCENFOLDERS:
            out[D][S]['maxcaptype'] = out[D][S]['maxcap'].groupby([GENMAP[i] for i in out['Diff'][S]['maxcap']],axis=1).sum()
            out[D][S]['gentype'] = out[D][S]['gen'].groupby([GENMAP[i] for i in out['Diff'][S]['gen']],axis=1).sum()
            out[D][S]['capfactor'] = out[D][S]['gentype'] / out[D][S]['maxcaptype']
        #grouped generation
        out['Diff'][S]['gentype'] = make_diff('gentype', S)

    # make df to compare key data
    comp = {}
    for S in MASTER_SCENS:
        comp[S] = out['Diff'][S]['gentype'].copy()
        x = comp[S]
        # x['id_therm_capfactor'] = out['Intraday'][S]['capfactor']['THERM']
        x.columns = ['diff_'+i for i in x.columns]
        x['dem_diff'] = out['Diff'][S]['dem'].sum(axis=1)
        x[['id_'+i for i in out['Intraday'][S]['gentype']]] = out['Intraday'][S]['gentype']
        
        x['id_curtailed'] = out['Intraday'][S]['gencurtailed'].sum(axis=1)
        x['id_unserved'] = out['Intraday'][S]['unserved'].sum(axis=1)
        x['id_dem'] = out['Intraday'][S]['dem'].sum(axis=1)
        x['id_price'] = out['Intraday'][S]['price']
        x['id_therm_undispatched'] = out['Intraday'][S]['undispatched'][THERMGEN].sum(axis=1)
        x = x.round(2)
    
else:
    
    for S in MASTER_SCENS:
        out[S]['maxcaptype'] = out[S]['maxcap'].groupby([GENMAP[i] for i in out[S]['maxcap']],axis=1).sum()
        out[S]['gentype'] = out[S]['gen'].groupby([GENMAP[i] for i in out[S]['gen']],axis=1).sum()
        out[S]['capfactor'] = out[S]['gentype'] / out[S]['maxcaptype']

    comp = {}
    for S in MASTER_SCENS:
        comp[S] = out[S]['gentype'].copy()
        x = comp[S]
        x['therm_capfactor'] = out[S]['capfactor']['THERM']
                
        x['curtailed'] = out[S]['gencurtailed'].sum(axis=1)
        x['unserved'] = out[S]['unserved'].sum(axis=1)
        x['dem'] = out[S]['dem'].sum(axis=1)
        x['price'] = out[S]['price']
        x['therm_undispatched'] = out[S]['undispatched'][THERMGEN].sum(axis=1)
        if 1 in NUM:
            try:
                x['power2x'] = out[S]['p2xload'].sum(axis=1)
            except:
                print(f'THERE IS NO P2X FOR {S}')
        if 2 in NUM:
            try:
                x['batteryload'] = out[S]['battload'].sum(axis=1)
                x['dsrdeferred'] = out[S]['dsrdeferred'].sum(axis=1)
            except:
                print(f'THERE IS NO BATTERY FOR {S}')
        x = x.round(2)
        


#%% 

pd.to_pickle(out, OUTRES)



