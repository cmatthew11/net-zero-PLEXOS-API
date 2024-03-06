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
from System import Enum
from plexosapi import *

#%% 

BASESTOR = os.path.abspath(r'2.power2x_model\2.power2x_model.xml')

# the input is the finished elec model xml, ouput is to storage folder
NEW = os.path.abspath(r'3.storage_model\3.storage_model.xml')
db, DB_METHODS = db_setup(BASESTOR , NEW)

CLASS_NAMES = cache_classes(db)
SYSTEM_COLLECTION_NAMES = cache_system_collections(db)

#%% Input data
    
BAT = os.path.join(INPUT, 'Hydrogen and storage', 
                       'Hydrogen_and_storage_capacity_all_scens.csv')
BAT = pd.read_csv(BAT, index_col=0)
batmw = BAT[[i for i in BAT.columns if 'storage_MW_' in i]]
batmw.columns = [i.replace('storage_MW_','') for i in batmw]

batmwh = BAT[[i for i in BAT.columns if 'storage_MWh_' in i]]
batmwh.columns = [i.replace('storage_MWh_','') for i in batmw]

TECHNO = os.path.join(INPUT, 'PLEXOS mainland generation data.xlsx')
techno = pd.read_excel(TECHNO, sheet_name='PLEXOS_technoeconomic',
                       usecols='A:S')
techno = techno.loc[~techno.value.isna()]
techno = techno[techno['use']]
techno = techno[[i for i in techno.columns if i not in 
                 ['units', 'category','use']]]
techno.loc[techno.scenario.isna(),'scenario'] = None
techno = techno[~techno.BAT.isna()]
techno = dict(zip(techno['value'],techno['BAT']))

dsr_techno = pd.read_excel(TECHNO, sheet_name='dsr_technoeconomic',
                           index_col=0)
dsr_techno = dsr_techno['value']

#%% 3. Add  batteries

print('ADDING BATTERIES TO MODEL')

for NODE in BAT.index:
    NAME = f'BAT_{NODE}'
    add_object(db, ClassEnum.Battery, NAME, None)
    db.AddMembership(CollectionEnum.BatteryNodes, NAME, NODE)
    
    # units for islands are 1
    VAL = round(batmw.at['MA-0','medium']/1000)
    if NODE != 'MA-0':
        VAL = 1
    add_prop(db,  CollectionEnum.SystemBatteries, 
                     None, NAME, 
                     'Units', VAL, 
                     1, type_='battery')
    # add properties (capacity and power in the table)
    for PROP, VAL in techno.items():
        if PROP not in ['Capacity', 'Max Power']:
            add_prop(db,  CollectionEnum.SystemBatteries, 
                     None, NAME, 
                     PROP, VAL, 
                     1, type_='battery')
        elif NODE == 'MA-0':
            add_prop(db,  CollectionEnum.SystemBatteries, 
                     None, NAME, 
                     PROP, VAL, 
                     1, type_='battery')
    
    #add island capacities by scenario
    if NODE !='MA-0':
        for scen in allscens:
            s_no_w = scen.split('-')[0]
            s_stor = SCENMAP.at[s_no_w,'storage']
            add_prop(db,  CollectionEnum.SystemBatteries, 
                         None, NAME, 
                         'Max Power', float(batmw.at[NODE,s_stor]), 
                         1, scen=scen, type_='battery')
            add_prop(db,  CollectionEnum.SystemBatteries, 
                         None, NAME, 
                         'Capacity', float(batmwh.at[NODE,s_stor]), 
                         1, scen=scen, type_='battery')


#%% DEMAND SIDE RESPONSE - AS VEHICLE CLASS

print('ADDING DSR TO MODEL')

#ADD CHARGING STATIONS- need charging settings and max power
for REGION in REGIONS:
    NAMEREGCHARGE = f'charging-station_{REGION}'
    add_category(db, ClassEnum.ChargingStation, NAMEREGCHARGE)
    NAMEREGVEH = f'dsr_{REGION}'
    add_category(db, ClassEnum.Vehicle, NAMEREGVEH)


    for NODE in NODES:
        if REGION in NODE:
            #charging stations
            NAMENODECHARGE = f'charging-station_{NODE}'
            add_object(db, ClassEnum.ChargingStation, NAMENODECHARGE, NAMEREGCHARGE)
            db.AddMembership(CollectionEnum.ChargingStationNode, NAMENODECHARGE, NODE)
            
            for VALNAME in dsr_techno.index:
                VAL = int(dsr_techno.at[VALNAME])
                add_prop(db,  CollectionEnum.SystemChargingStations, 
                          None, NAMENODECHARGE, 
                          VALNAME, VAL, 
                          1, type_='chargingstation')
            
            
            #vehicles (dsr load)            
            NAMENODEVEH = f'dsr_{NODE}'
            add_object(db, ClassEnum.Vehicle, NAMENODEVEH, NAMEREGVEH)
            db.AddMembership(CollectionEnum.VehicleChargingStations,NAMENODEVEH, NAMENODECHARGE )
            add_prop(db,  CollectionEnum.SystemVehicles, 
                          None, NAMENODEVEH, 
                          'Units', 1, 
                          1, type_='vehicle')
            
            #dsr profiles
            for scen in allscens:
                s_no_w = scen.split('-')[0]
                s_stor = SCENMAP.at[s_no_w,'storage']
                if not TROUBLESHOOT:
                    PATH_DEM = os.path.join(INPUT,'Demand', s_stor, w)
                else:
                    PATH_DEM = os.path.join('Demand', s_stor, w)
                
                DATAFILE = os.path.join(PATH_DEM,f'DEM-DSR_{NODE}.csv')                
                if os.path.exists(DATAFILE):
                    add_prop(db,  CollectionEnum.SystemVehicles, 
                              None, NAMENODEVEH, 
                              'Fixed Load', 1, 
                              1, scen, DATAFILE, 
                              type_='vehicle')
                    DATAFILE = os.path.join(PATH_DEM,f'MAX-CHARGE_{NODE}.csv')
                    add_prop(db,  CollectionEnum.SystemChargingStations, 
                              None, NAMENODECHARGE, 
                              'Max Charge Rate', 1, 
                              1, scen, DATAFILE, 
                              type_='chargingstation')
                
                
#%% CLOSE

db.Close()
