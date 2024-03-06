# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:45:23 2023

@author: Chris
"""

import os
import sys
import pandas as pd
from shutil import copyfile
import clr #https://github.com/spyder-ide/spyder/issues/21269#issuecomment-1684439990

#determines if these settings run
TROUBLESHOOT = False
MAINLAND_SCEN = False
DAYAHEAD = False

version = '10.0'
print('PLEXOS VERISON {} BEING USED\n\n'.format(version))

# load PLEXOS assemblies
PLEXOSPATH = f'C:\Program Files\Energy Exemplar\PLEXOS {version}'
PLEXOSAPIPATH = PLEXOSPATH +' API'

sys.path.append(PLEXOSAPIPATH)
clr.AddReference('PLEXOS_NET.Core')
clr.AddReference('EEUTILITY')
clr.AddReference('EnergyExemplar.PLEXOS.Utility')

# .NET related imports
from PLEXOS_NET.Core import DatabaseCore
from EEUTILITY.Enums import *
from EnergyExemplar.PLEXOS.Utility.Enums import *
# from EnergyExemplar.PLEXOS.Utility.Enums import ClassEnum, CollectionEnum
from System import Enum
from plexosapi import *


#%% 

BASEHEAT = os.path.abspath(r'3.storage_model\3.storage_model.xml')

# the input is the finished elec model xml, ouput is to storage folder
NEW = os.path.abspath(r'4.heat_model\4.heat_model.xml')
db, DB_METHODS = db_setup(BASEHEAT , NEW)

# not interested in mainland heat demand
NODES = [i for i in NODES if i!='MA-0']

CLASS_NAMES = cache_classes(db)
SYSTEM_COLLECTION_NAMES = cache_system_collections(db)

GENTECHNO = os.path.join(INPUT, 'PLEXOS mainland generation data.xlsx')


#%% Input data
  
dems = os.listdir(os.path.join(INPUT,'Demand','high','summer_high_wind'))
dems = [i for i in dems if 'heat' in i.lower() and 'MA' not in i ]
NODES = [i for i in NODES if any([i in j for j in dems])]

techno = pd.read_excel(GENTECHNO, sheet_name='heat_technoeconomic',
                       index_col=0)['value'].dropna()

if not TROUBLESHOOT:
    ZERODF = os.path.join(INPUT,'Demand','zero_demand_file.csv')                       
else:
    ZERODF = os.path.join('Demand','zero_demand_file.csv')

#%% 4. Add heat node and demand
print('ADDING HEAT NODES AND DEMAND ...')

# Add heat demand
for REGION in REGIONS:
    #heat
    if REGION!='MA':
        add_category(db, ClassEnum.HeatNode, REGION)

for NODE in NODES:
    NAME = 'heat-node_'+NODE
    
    REGION = NODE.split('-')[0]
    add_object(db, ClassEnum.HeatNode, NAME, REGION)

    #add demand
    for scen in allscens:
        w = scen.split('-')[-1]
        s_no_w = scen.split('-')[0]
        s_dem = SCENMAP.at[s_no_w,'supply']
        if not TROUBLESHOOT:
            PATH_SUP = os.path.join(INPUT,'Demand', s_dem, w)
        else:
            PATH_SUP = os.path.join('Demand', s_dem, w)
        OUTPATH = os.path.join(PATH_SUP,f'DEM-HEAT_{NODE}.csv')
        if not os.path.exists(OUTPATH):
            OUTPATH = ZERODF
        add_prop(db, CollectionEnum.SystemHeatNodes,
                            None, NAME,
                            'Heat Demand', 1,
                            1, scen, OUTPATH, 'heatnode')
        
# ADD AN EXTRA GAS NODE AND STORAGE SHARED BETWEEN ALL HEAT PLANTS 
# WITH A WITHDRAWAL COST EQUAL TO THE AMMONIA COST (£107/GJ)
NODENAME = 'gas-node_heatbackup'
STORNAME = 'gas-storage_heatbackup'
add_object(db, ClassEnum.GasNode, NODENAME, None)
add_object(db, ClassEnum.GasStorage, STORNAME,None)
db.AddMembership(CollectionEnum.GasStorageGasNodes, STORNAME, NODENAME)
add_prop(db, CollectionEnum.SystemGasStorages,
                  None, STORNAME, 
                  'Max Volume', float(10**3), # this is in TJ
                  1, type_='gasstorage')
add_prop(db, CollectionEnum.SystemGasStorages,
                  None, STORNAME, 
                  'Initial Volume', float(10**3), # this is in TJ
                  1, type_='gasstorage')
add_prop(db, CollectionEnum.SystemGasStorages,
                  None, STORNAME, 
                  'Withdrawal Charge', 107,
                  1, type_='gasstorage')
            
#%% Add heat plants 

print('ADDING HEAT PLANTS ...')

for REGION in REGIONS:
    #heat
    if REGION!='MA':
        add_category(db, ClassEnum.HeatPlant, REGION)

for NODE in NODES:
    NAME = 'heat-plant_'+NODE
    REGION = NODE.split('-')[0]
    add_object(db, ClassEnum.HeatPlant, NAME, REGION)
    db.AddMembership(CollectionEnum.HeatPlantHeatOutputNodes, NAME, 'heat-node_'+NODE)
    db.AddMembership(CollectionEnum.HeatPlantGasNodes, NAME, 'gas-node_'+REGION)
    # this is for the backup heat node/storage
    db.AddMembership(CollectionEnum.HeatPlantGasNodes, NAME, NODENAME)
    for PROP in techno.index:
        VAL = techno.at[PROP]
        add_prop(db, CollectionEnum.SystemHeatPlants,
                            None, NAME,
                            PROP, VAL,
                            1, scen=None, type_='heatplant')
    

#%% 

db.Close()
