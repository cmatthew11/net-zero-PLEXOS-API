# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:45:52 2023

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

BASE2X = os.path.abspath(r'1.elec_model\1.elec_model.xml')

# the input is the finished elec model xml, ouput is to storage folder
NEW = os.path.abspath(r'2.power2x_model\2.power2x_model.xml')
db, DB_METHODS = db_setup(BASE2X , NEW)

CLASS_NAMES = cache_classes(db)
SYSTEM_COLLECTION_NAMES = cache_system_collections(db)

#%% Input data
    
P2XTECHNO = os.path.join(INPUT, 'PLEXOS mainland generation data.xlsx')
p2xtechno = pd.read_excel(P2XTECHNO,sheet_name='PLEXOS_power2x_technoeconomic',
                          index_col=0)['value'].dropna()

power2xcap = pd.read_excel(P2XTECHNO,sheet_name='PLEXOS_power2x',
                          index_col=0,keep_default_na=False)

storagecap = pd.read_excel(P2XTECHNO,sheet_name='PLEXOS_hydrogen_storage',
                          index_col=0,keep_default_na=False)

#only need this for mainland generation
gencap = pd.read_csv(os.path.join(INPUT, 'Supply', 
                                  'mainland_units_PLEXOS_input.csv'),
                     index_col=0)
gencap.index = [i.replace('GEN_','') for i in gencap.index]
gencap = gencap.loc['HYG']

DEMNODES = os.listdir(r'C:\Users\Chris\Desktop\Coding\Python\PLEXOS\Input data\Demand\low\winter_low_wind_high_demand')
DEMNODES = [i.split('_')[1].split('.')[0] for i in DEMNODES if 'HYD' in i]

#%%

#     gas shortage price NEEDS TO BE LESS THAN COST OF GCC 
#     GENERATION OTHERWISE GAS WILL RUN TO PRODUCE HYDROGEN (GAS IN MODEL)
    
#     £460/MWh = £127.77/GJ
#     but GCC gen is £99/MWh, which is £27.5/GJ
     
#       Gas Demand Shortage Price page says:
# In case P2X element is converting electricityto gas, the Gas Demand Shortage 
# Price should be smaller than the VoLL on the electricity node. In case gas is 
# being supplied to a generator that is generating electricity, the shortage
#   price of Gas Demand should be higherthan the VoLL on electric side.
  

#  the withdrawal charge here is based on the mainland price of hydrogen
# 63.89 $/GJ- times the efficiency for ammonia of 60%- 
# so 106.48$/GJ

# THIS MEANS THE GAS SHORTAGE PRICE NEEDS TO BE HIGHER THAN THIS

# - if the recycle price is greater than the shortage cost, it will just 
# cause the gas demand to be short, so it must be less than this but
# greater than the withdrawal cost

# with only the market and no storage, only need one price-
# the price at which p2x will run (the generation type up to, in this
# case wind @£50/MWh (with price variability)- @ 82% = 60.1£/MWh
# = 17£/GJ)
gas_prices = {
    'gen_fuel':64,
    'wind_cap_cost': 17
              }


# NEED TO CONVER STORAGE VOLUME INTO TERJOULES TJ
# INPUT FILE IS IN MWH

# For the non-dist scenarios, the islands will have an additional gas 
# storage object with an intial volume very high (effectively infinite as
# gas demand on islands is very small vs mainland) and a "Withdrawal Charge" 
# which will set the price of the gas used to be balanced against the cost
# of local generation (for the mid scenario only). This will mean that when 
# local hydrogen is cheaper it will be used or else imported.

mwh2tj = 0.0036
    
#%% add fuel - one for each region

print('ADDING HYDROGEN FUELS...')

    
#needs to be a seperate fuel for hyrdogen generation to the gas network
add_object(db, ClassEnum.Fuel, 'fuel-HYG', None)

# ASSUME THAT GAS PRICE IS SO HIGH IT WILL DISCOURAGE GENERATION 
# FROM RUNNING UNLESS TO AVOID UNSERVED ENERGY-
# NOTE THAT GAS SHORTAGE PRICE NEEDS TO BE LESS THAN THE HYDROGEN SRMC OF 
# ~460£/MWh OTHERWISE IT WILL RUN HYDROGEN TO PROCUDE THE GAS

add_prop(db, CollectionEnum.SystemFuels,
         None, 'fuel-HYG',
         'Price', gas_prices['gen_fuel'], #cost of 23p/kWh from THESIS 
         1, None, None, type_='fuel')


#%% Add gas nodes- with demand

print('ADDING HYDROGEN NODES AND DEMAND')


for REGION in REGIONS:
    NAMEREGNODE = 'gas-node_'+REGION
    add_category(db, ClassEnum.GasNode, NAMEREGNODE)
    NAMEREGDEM = 'gas-demand_'+REGION
    add_category(db, ClassEnum.GasDemand, NAMEREGDEM)


"""HYDROGEN DEMAND IS IN TERAJOULES TJ"""

for NODE in NODES:#GASNODES:
    #gas
    REGION = NODE.split('-')[0]
    # gas nodes
    NAMENODENODE = 'gas-node_'+NODE
    NAMEREGNODE = 'gas-node_'+REGION
    add_object(db, ClassEnum.GasNode, NAMENODENODE, NAMEREGNODE)

    # add demand
    NAMENODEDEM = 'gas-demand_'+NODE
    NAMEREGDEM = 'gas-demand_'+REGION
    
    # V1 where only demand nodes have demand- not p2xonly nodes
    if NODE in DEMNODES:
        add_object(db, ClassEnum.GasDemand, NAMENODEDEM, NAMEREGDEM)
        db.AddMembership(CollectionEnum.GasDemandGasNodes, NAMENODEDEM, NAMENODENODE)
    
        add_prop(db, CollectionEnum.SystemGasDemands, 
                      None, NAMENODEDEM, 
                      # 'Shortage Price', gas_shortage_price[NODE],
                      'Shortage Price', gas_prices['wind_cap_cost'],
                      1, None, None, type_='gasdemand')
        
        for scen in allscens:
            s_no_w = scen.split('-')[0]    
            s_dem = SCENMAP.at[s_no_w,'demand']
        
            w = scen.split('-')[-1]
            FILENAME = ''.join(['DEM-HYD_', NODE, '.csv'])
            if NODE == 'MA-0' and not MAINLAND_SCEN:
                s_dem = 'medium'
            
            if not TROUBLESHOOT:
                DATAFILE = os.path.join(INPUT,'Demand', s_dem,
                                        w, FILENAME)
            else:
                DATAFILE = os.path.join('Demand', s_dem,
                                        w, FILENAME)
            
            add_prop(db, CollectionEnum.SystemGasDemands, 
                      None, NAMENODEDEM, 
                      'Demand',0,
                      1, scen, DATAFILE, type_='gasdemand')
        
        
            
#%% mainland generation- units and fuel

print('ADDING HYDROGEN GENERATION...')

if not MAINLAND_SCEN:
    UNITS = int(gencap['mid'] * unit_factor)
    add_prop(db,  CollectionEnum.SystemGenerators, 
                             'HYG', 'MA-0', 
                            'Units', UNITS * unit_factor,
                            1, None, None, 'gen')

else:
    for scen in allscens:
        s_no_w = scen.split('-')[0]    
        s_sup = SCENMAP.at[s_no_w,'supply']
    
        UNITS = int(gencap[s_sup] * unit_factor)
        add_prop(db,  CollectionEnum.SystemGenerators, 
                             'HYG', 'MA-0', 
                            'Units', UNITS ,
                            1, scen, None, 'gen')
        
db.AddMembership(CollectionEnum.GeneratorFuels, 'HYG_MA-0','fuel-HYG')

#%% power2x units
"""POWER2X FOR GAS MODEL AND ELEC MODEL NEED TO BE SEPERATE, 
THEREFORE THE ISLANDS AND MAINLAND HAVE ONE GAS DEMAND POWER2X WITH ONE 
ADDITIONAL MAINLAND ONE FOR GENERATOIN"""

print('ADDING POWER2X UNITS...')

""" NEED TO ADJUST THIS LATER, SAME FOR STORAGE"""
# gas_elec_prop = 0.2


for REGION in REGIONS:
    CAT = 'power2x_'+REGION
    add_category(db, ClassEnum.Power2X, CAT)
    

for NODE in NODES:#GASNODES:
    REGION = NODE.split('-')[0]
    CAT = 'power2x_'+REGION
    
    NAME = 'power2x_'+NODE
    add_object(db, ClassEnum.Power2X, NAME, CAT)
    db.AddMembership(CollectionEnum.Power2XGasNodes, NAME, 'gas-node_'+NODE)
    db.AddMembership(CollectionEnum.Power2XNodes, NAME, NODE)
        
    
    
    for VALNAME in p2xtechno.index:
        # if NODE == 'MA-0' and VALNAME == 'VOM Charge':
        #     continue
        if VALNAME == 'VOM Charge':
            continue
        VAL = p2xtechno.at[VALNAME]
        add_prop(db, CollectionEnum.SystemPower2X,
                  None, NAME, 
                  VALNAME, VAL,
                  1, type_='power2x')
    
    for scen in allscens:
        s_no_w = scen.split('-')[0]    
        s_p2x = SCENMAP.at[s_no_w,'p2x']
    
        w = scen.split('-')[-1]
        # s_sup = ssup_map_mw[s_sup]
        
        VAL = float(power2xcap.loc[NODE, s_p2x].round(2))
        if VAL == 0:
            UNIT = 0
        else:
            UNIT = 1
        
        add_prop(db, CollectionEnum.SystemPower2X,
              None, NAME, 
              'Units', UNIT * unit_factor,
              1, scen, type_='power2x')

        add_prop(db, CollectionEnum.SystemPower2X,
                  None, NAME, 
                  'Max Load', round(VAL,1),
                  1, scen=scen, type_='power2x')
        
        
            
#%% ADD GAS MARKET TO CAUSE P2X TO RUN EXTRA DEMAND- MAINLY FOR EXPORT SCEN

# need to define units, price and max sales (market is buying from node) 
# or purchase (market is selling to demand = 0)

#PRICE NEEDS TO BE LESS THAN SHORTAGE OR ELSE ALL SHORTAGE
MARKET_PROPS = {'Max Purchases':0,
                # 'Price':gas_prices['market_buy_from_island'],
                'Price': gas_prices['wind_cap_cost']-0.5,
                'Units':1
                }

for REGION in REGIONS:
    NAME = f'gas-market_{REGION}'
    add_object(db, ClassEnum.Market, NAME, None)
    
    for VALNAME, VAL in MARKET_PROPS.items():
        if REGION == 'MA' and VALNAME == 'Price':
        #     VAL = 27
            VAL -= 1 # this means that island will run ahead of mainland
        add_prop(db, CollectionEnum.SystemMarkets,
                 None, NAME,
                 VALNAME, VAL,
                 1, type_='market')
            
for NODE in power2xcap.index:
    REGION = NODE.split('-')[0]
    NAME = f'gas-market_{REGION}'

    db.AddMembership(CollectionEnum.GasNodeMarkets, 'gas-node_'+NODE, NAME)       

#%%

db.Close()
