# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:19:33 2023

@author: Chris
"""

""" first section from
C:/Users/Chris/Desktop/Coding/Python/PLEXOS/PLEXOS API/Github API docs/Python-PLEXOS-API/Input Files/create_inputs.py


COPY FIRST SECTION FOR ALL API SCRIPTS
"""

import os
import sys
import pandas as pd
from shutil import copyfile
import numpy as np
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

#%% FUNCTIONS

def sin_eq(val, factor):
    return  factor * ( (np.sin( (val+0.5) / (np.pi / 10)))) + 1

def wind_variable_bidding(df,factor):
    MAX = df.max()
    MIN = df.min()
    df = (df  - MIN) / MAX
    df = df.apply(sin_eq,factor=factor)
    return df
    
#%% IMPORT BLANK FILE AND DB

BASE = os.path.abspath(r'Set up and blank file\blank_file_v3 - 10.0 - 1.11.23.xml')
NEW = os.path.abspath(r'1.elec_model\1.elec_model.xml')

db, DB_METHODS = db_setup(BASE, NEW)
    
CLASS_NAMES = cache_classes(db)
SYSTEM_COLLECTION_NAMES = cache_system_collections(db)
    
#%% SET UP FILE NAMES FOR DATA OUTPUTS and define misc data 

GENTECHNO = os.path.join(INPUT, 'PLEXOS mainland generation data.xlsx')
TRANS = os.path.join(INPUT, 'Transmission', 'transmission_plexos - BY SCENARIO.xlsx')
ISLCAP = os.path.join(INPUT, 'Supply', 'island_max_capacities.pickle')

# generator data and capacities
SHEET = 'PLEXOS_technoeconomic'
if DAYAHEAD:
    SHEET += '_dayahead'
techno = pd.read_excel(GENTECHNO, sheet_name=SHEET,
                       usecols='A:S')
techno = techno.loc[~techno.value.isna()]
techno = techno[techno['use']]
techno = techno[[i for i in techno.columns if i not in 
                 ['units', 'category','use']]]
techno.loc[techno.scenario.isna(),'scenario'] = None
GENS = [i for i in techno.columns if i not in ['use', 'scenario', 'band',
                                               'units','category','gen_type',
                                                'BAT','value']]

DA_GEN = [i for i in GENS if i not in ['ONS','OFF','SOL','MAR','PST','REN',
                                       'HYD']]

gencap = pd.read_csv(os.path.join(INPUT, 'Supply', 
                                  'mainland_units_PLEXOS_input.csv'),
                     index_col=0)
gencap.index = [i.replace('GEN_','') for i in gencap.index]
gencap = gencap.loc[[i for i in GENS if 'FIT' not in i]]

cfs = os.listdir(os.path.join(INPUT,'Capacity factors','cent','summer_high_wind'))
cfs = [i for i in cfs if 'DA_' not in i and 'vom' not in i.lower() and 'markup' not in i.lower()]
islcap = pd.read_pickle(ISLCAP)

trans = pd.read_excel(TRANS, sheet_name = 'transmission_plexos - BY SCENAR')

factor_prop = ['Max Capacity', 'Load Point', 'Start Cost'] # some depend on the max capacity
skip_props = ['Max Capacity', 'Fuel Price']

#%% 1. GENERATORS
print('ADDING GENERATORS...')

# 1A. Add categories and gen objects- list of generator types
for GEN in GENS:
    if GEN not in ['GCC','BEC']:
        add_category(db, ClassEnum.Generator, GEN)
        add_object(db, ClassEnum.Generator, 
               GEN+'_MA-0', GEN)

for GEN in cfs:
    add_object(db, ClassEnum.Generator,
               GEN.split('.')[0], GEN.split('_')[0])
    
    
#%%  2. Add scenarios and datafiles
print('ADDING SCENARIOS AND DATAFILES...')

# 2A. Add scenario objects
WADDED = []
for scen in allscens:
    w = scen.split('-')[-1]
    
    if w not in WADDED:
        add_category(db, ClassEnum.Scenario, w)
        WADDED.append(w)
    s_no_w = scen.split('-')[0]    
    s_dem = SCENMAP.at[s_no_w,'demand']
    s_sup = SCENMAP.at[s_no_w,'supply']
    add_object(db, ClassEnum.Scenario, scen, w)
    if DAYAHEAD:
        add_object(db, ClassEnum.Scenario, 'DA_'+scen, w)
        add_object(db, ClassEnum.Scenario, 'ID_'+scen, w)
    # 2B. Add capacity factors
    if not TROUBLESHOOT:
        PATH_SUP = os.path.join(INPUT,'Capacity factors', s_sup, w)
    else:
        PATH_SUP = os.path.join('Capacity factors', s_sup, w)
    for CF in cfs: #HIS IS USED TO FILTER WHICH GEN TO INCLUDE
        DATAFILE = os.path.join(PATH_SUP, CF)
        GEN = CF.split('_')[0]
        NODE = CF.split('_')[1].replace('.csv','')
        CF_NAME = CF.replace('.csv','')
        if NODE == 'MA-0' or islcap[GEN].loc[NODE,s_sup] > 0:
            
            # not sure why but appaz fixed load is for renewbales
            PROP = 'Rating'
            if GEN == 'MAR':
                PROP = 'Fixed Load'
            # PROP = 'Fixed Load'
            
            if DAYAHEAD:
                # rating is for renewable generation, max capacity is the installed cap
                add_prop(db,  CollectionEnum.SystemGenerators, 
                         GEN, NODE, 
                        PROP, 1, # Anything with datafile has value "1"
                        1, 'ID_'+scen, DATAFILE, 'gen')
                # marine generation has the same cf day ahead
                if 'MAR' not in GEN:
                    DATAFILE = os.path.join(PATH_SUP, 'DA_'+CF)
                add_prop(db,  CollectionEnum.SystemGenerators, 
                         GEN, NODE, 
                        PROP, 1, # Anything with datafile has value "1"
                        1, 'DA_'+scen, DATAFILE, 'gen')   
            else:
                add_prop(db,  CollectionEnum.SystemGenerators, 
                         GEN, NODE, 
                        PROP, 1, # Anything with datafile has value "1"
                        1, scen, DATAFILE, 'gen')
            
        if 'MA-0' not in CF_NAME:
            CAP = float(islcap[GEN].loc[NODE,s_sup])
            add_prop(db,  CollectionEnum.SystemGenerators, 
                 GEN, NODE, 
                'Max Capacity', round(CAP/unit_factor,2), 
                1, scen, None, 'gen')

          
    if MAINLAND_SCEN :
        # 2C. Add mainland capacity units
        for GEN in GENS:
            if 'FIT' not in GEN and GEN!='HYG': #avoid adding hydrogen yet
                units = int(gencap.loc[GEN,s_sup])
                NODE = 'MA-0'
                add_prop(db,  CollectionEnum.SystemGenerators, 
                         GEN, NODE, 
                        'Units', units * unit_factor,
                        1, scen, None, 'gen')
                if GEN in ['BEC','HYD','NUC']:
                    add_prop(db,  CollectionEnum.SystemGenerators, 
                         GEN, NODE, 
                        'Fixed Load', units * unit_factor * 1000,
                        1, scen, None, 'gen')
                

if not MAINLAND_SCEN:
    for GEN in GENS:
        if 'FIT' not in GEN and GEN!='HYG': #avoid adding hydrogen yet
            units = int(gencap.loc[GEN,'mid'])
            NODE = 'MA-0'
            add_prop(db,  CollectionEnum.SystemGenerators, 
                     GEN, NODE, 
                    'Units', units * unit_factor,
                    1, type_='gen')
            if GEN in ['BEC','HYD','NUC']:
                    add_prop(db,  CollectionEnum.SystemGenerators, 
                         GEN, NODE, 
                        'Fixed Load', units * unit_factor * 1000,
                        1, type_='gen')


#%% add wind variable bidding behaviour- for markup and variable for mainland
# this alters the bid price of wind and gas depending on how much of either
# there is on the system- more wind==lower bids, and vice versa

if not DAYAHEAD:  

    for scen in allscens:
        w = scen.split('-')[-1]
        s_no_w = scen.split('-')[0]
        s_dem = SCENMAP.at[s_no_w,'demand']
        s_sup = SCENMAP.at[s_no_w,'supply']
        if not TROUBLESHOOT:
            PATH_SUP = os.path.join(INPUT,'Capacity factors', s_sup, w)
        else:
            PATH_SUP = os.path.join('Capacity factors', s_sup, w)
        
        ONS = os.path.join(PATH_SUP, 'ONS_MA-0.csv')
        OFF = os.path.join(PATH_SUP, 'OFF_MA-0.csv')
        ONS = pd.read_csv(ONS)
        OFF = pd.read_csv(OFF)
        WIND = ONS.copy()
        WIND.loc[:,'VALUE'] += OFF.VALUE
        WINDVARBID = WIND.copy()
        WINDVARBID.loc[:,'VALUE'] = wind_variable_bidding(WIND.loc[:,'VALUE'],-0.25)
        
        #do the same (but opposite- negative factor to invert) for gas
        GASVARBID = WIND.copy()
        GASVARBID.loc[:,'VALUE'] = wind_variable_bidding(WIND.loc[:,'VALUE'],0.25)
        
        # then write this to a file to use as the varibale costs
        #this is for mainland
        for GEN in ['ONS','OFF','GCC']:
            for PROP in ['VOM Charge','Markup']:
                    
                VAR = techno[(techno['value']==PROP)][GEN].values[0]
                if GEN != 'GCC':
                    OUT = WINDVARBID.copy()
                else:
                    OUT = GASVARBID.copy()
                OUT.loc[:,'VALUE'] *= VAR
                OUT.loc[:,'VALUE'] = OUT.loc[:,'VALUE'].round(2)
                OUTPATH = os.path.join(PATH_SUP,f'{GEN}_MA-0_{PROP}.csv')
                OUT[['DATETIME', 'VALUE']].to_csv(OUTPATH,index=False)
                if GEN == 'ONS':
                    OUT.loc[:,'VALUE'] *= 0.8
                    OUT.loc[:,'VALUE'] = OUT.loc[:,'VALUE'].round(2)
                    OUTPATH = os.path.join(PATH_SUP,f'{GEN}_islands_{PROP}.csv')
                    OUT[['DATETIME', 'VALUE']].to_csv(OUTPATH,index=False)
                    
                    
                add_prop(db, CollectionEnum.SystemGenerators,
                        GEN, 'MA-0',
                        PROP, 1,
                        1, scen, OUTPATH, 'gen')
        #for island generation
        for CF in cfs:
            if 'ONS' in CF and 'MA-0' not in CF:
                GEN = 'ONS'
                NODE = CF.split('_')[1].replace('.csv','')
                for PROP in ['VOM Charge','Markup']:
                        
                    
                    OUTPATH = os.path.join(PATH_SUP,f'{GEN}_islands_{PROP}.csv')
                    add_prop(db, CollectionEnum.SystemGenerators,
                            GEN, NODE,
                            PROP, 1,
                            1, scen, OUTPATH, 'gen')
            
else:
    print('********VARIABLE WIND BIDDING NOT WORKING FOR DAY AHEAD*******')
    
#%% 1C. Add generators for the mainland 


# Mainland capacity all has capacity of 1000MW, just the units that changes with scenarios
for GEN in GENS:
    for i in techno.index:
        PROP = techno.at[i,'value']
        if PROP not in ['Fuel Price']:
            val = techno.at[i, GEN]
            if not pd.isna(val) and type(val) is not str:
                band = int(techno.at[i,'band'])
                scen = techno.at[i,'scenario']

                if PROP in factor_prop:
                    val /= unit_factor
                    val = float(val)
                #this is for the variable vom charge
                if not all([GEN in ['ONS','OFF','GCC'] , PROP in ['VOM Charge','Markup']]):
                    add_prop(db,  CollectionEnum.SystemGenerators, 
                            GEN, 'MA-0', 
                            PROP, val, 
                            band, scen=scen)
                
# ADD DA GEN BEHAVIOUR
if DAYAHEAD:
    for GEN in DA_GEN:
        for NAME, VALUE_NAME in zip(['Generation','Generation','Undispatched Capacity'],
                                    ['Offer Quantity','Offer Base','Offer Quantity']):
            FILE = 'Model Day Ahead Solution\Interval\ST Generator(*).{}.csv'.format(NAME)
            band = 1
            if NAME == 'Undispatched Capacity':
                band = 2
            scen = 'ID'
            add_prop(db,  CollectionEnum.SystemGenerators, 
                                     GEN, 'MA-0', 
                                    VALUE_NAME, 0, 
                                    band, scen,
                                    FILE)

#   1D. Add island capacities- units for mainland, 1 for island
for GEN_TECH in cfs:
    NODE = GEN_TECH.split('_')[1]
    GEN = GEN_TECH.split('_')[0]
    if 'MA-0' not in NODE: # otherwise being added twice
        add_prop(db,  CollectionEnum.SystemGenerators, 
                         GEN, NODE, 
                        'Units', unit_factor, 1)

# Add other properties 
    for i in techno[GEN].dropna().index:
        PLEXOS_PROP = techno.at[i,'value']
        if PLEXOS_PROP not in skip_props:
            val = techno.at[i, GEN]

            if not pd.isna(val) and type(val) is not str:

                band = int(techno.at[i,'band'])
                if 'MA-0' not in NODE and not all([GEN == 'ONS' , PROP in ['VOM Charge','Markup']]):
                    add_prop(db,  CollectionEnum.SystemGenerators, 
                                         GEN, NODE, 
                                        PLEXOS_PROP, val, 
                                        band)
                 

#%% 4. Add elec nodes and regions
print('ADDING NODES AND REGIONS...')

# 4A. Add nodes and regions - node naming 
REGION = 'UK'
SETTLE_OPTIONS = {'lmp':0, #has basically 0 variation, dont use
                  'reference node pricing':1, #wtf- 100% always the same
                  'regional load weighted':2, #still no variation
                  'pay as bid':3, #least worst variation
                  'uniform':4, #always infeasible
                  'none':5, #NA
                  'custom':6, #NA
                  'most expensive':7} #makes no sense

SETTLE = SETTLE_OPTIONS['uniform']#

add_object(db, ClassEnum.Region, REGION, None)
add_prop(db,  CollectionEnum.SystemRegions,
                       None, REGION, 
                      'Generator Settlement Model', SETTLE, 
                      1, None, None, type_='region')
add_prop(db,  CollectionEnum.SystemRegions,
                       None, REGION, 
                      'Load Settlement Model', SETTLE, 
                      1, None, None, type_='region')
add_prop(db,  CollectionEnum.SystemRegions,
                       None, REGION, 
                      'Uplift Enabled', -1, 
                      1, None, None, type_='region')

ZONES = list(set([i.split('-')[0] for i in NODES]))
for ZONE in ZONES:
    # add_category(db, ClassEnum.Region, REGION)
    add_object(db, ClassEnum.Zone, ZONE, None)
    add_category(db, ClassEnum.Node, ZONE)
    db.AddMembership(CollectionEnum.ZoneRegion, ZONE, REGION)
    
    for NODE in NODES:
        if ZONE in NODE:
        # elec nodes
            add_object(db, ClassEnum.Node, NODE, ZONE)
            db.AddMembership(CollectionEnum.NodeRegion, NODE, REGION)
            db.AddMembership(CollectionEnum.NodeZone, NODE, ZONE)
            add_prop(db,  CollectionEnum.SystemNodes,
                           GEN, NODE, 
                          'Load Participation Factor', 1, 
                          1, None, None, type_='node')
            for scen in allscens:
                s_no_w = scen.split('-')[0]
                s_dem = SCENMAP.at[s_no_w,'demand']
                s_sup = SCENMAP.at[s_no_w,'supply']
                w = scen.split('-')[-1]
                
                NAME = ''.join([s_sup.upper(), '-DEM-ELEC_', NODE, '.csv'])

                if 'MA' in NODE and not MAINLAND_SCEN:
                    s_dem = 'medium'
                if not TROUBLESHOOT:
                    DATAFILE = os.path.join(INPUT,'Demand', s_dem,
                                            w, NAME)
                    DADATAFILE = os.path.join(INPUT,'Demand', s_dem,
                                            w, 'DA_'+NAME)
                else:
                    DATAFILE = os.path.join('Demand', s_dem,
                                            w, NAME)
                    DADATAFILE = os.path.join('Demand', s_dem,
                                            w, 'DA_'+NAME)

                if DAYAHEAD:
                    add_prop(db,  CollectionEnum.SystemNodes,
                               None, NODE, 
                              'Load', 0, 
                              1, 'DA_'+scen, DADATAFILE, type_='node')
                    scen = 'ID_' + scen
                add_prop(db,  CollectionEnum.SystemNodes,
                           None, NODE, 
                          'Load', 0, 
                          1, scen, DATAFILE, type_='node')

#%% 5. Add interconnections
print('ADDING LINES...')

for i in trans.index:
    FROM = trans.at[i,'g1']
    TO = trans.at[i,'g2']
    NAME = FROM+'_'+TO
    add_object(db, ClassEnum.Line, NAME, None)
    for scen in allscens:
        s_tran = scen.split('-')[0]
        VAL = float(round(trans.at[i,s_tran],1))            
        add_prop(db,  CollectionEnum.SystemLines,
                 REGION, NAME, 
                'Max Flow', VAL,
                1, scen=scen, type_='line')
    db.AddMembership(CollectionEnum.LineNodeFrom, NAME, FROM)
    db.AddMembership(CollectionEnum.LineNodeTo, NAME, TO)
    
# #%% Add fuels
for GEN in ['GCC','BEC','NUC']:
    FUELNAME = 'fuel-'+GEN
    add_object(db, ClassEnum.Fuel, FUELNAME, None)
    GENNAME = GEN+'_MA-0'
    PRICE = float(round(techno.iloc[list(techno.value).index('Fuel Price'),:][GEN],1))
    db.AddMembership(CollectionEnum.GeneratorFuels, GENNAME, FUELNAME)
    add_prop(db,  CollectionEnum.SystemFuels,
             None, FUELNAME, 
            'Price', PRICE,
            1, type_='fuel')


#%% 8. Add node/region membership for everything
print('ADDING MEMBERSHIPS...')

for OBJ in db.GetObjects(ClassEnum.Generator):
    NODE = OBJ.split('_')[1]
    GEN = OBJ.split('_')[0]
    db.AddMembership(CollectionEnum.GeneratorNodes, OBJ, NODE)
    

#%% CLOSE AND SAVE

db.Close() 