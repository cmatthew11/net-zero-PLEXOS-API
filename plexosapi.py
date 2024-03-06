# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:15:26 2023

@author: Chris
"""

import os
import sys
import pandas as pd
from shutil import copyfile

# load PLEXOS assemblies
import clr #https://github.com/spyder-ide/spyder/issues/21269#issuecomment-1684439990

# UPDATE FILE PATH HERE WITH API FILES FROM ENERGY EXMPLAR ACADEMIC SUPPORT
sys.path.append(r'C:\Program Files\Energy Exemplar\PLEXOS 10.0 API')
clr.AddReference('PLEXOS_NET.Core')
clr.AddReference('EEUTILITY')
clr.AddReference('EnergyExemplar.PLEXOS.Utility')

# .NET related imports
from PLEXOS_NET.Core import DatabaseCore
from EEUTILITY.Enums import *
from EnergyExemplar.PLEXOS.Utility.Enums import *
from System import Enum

#%% THIS SECTION IS FOR SCOTTISH ISLANDS MODELLING, CAN DELETE OTHERWISE

unit_factor = 1 #MAKE SURE THIS MATCHES CFS
print('Unit factor of (check matches cfs):',unit_factor,'\n')

ss = ['high', 'medium', 'low']
ws = ['winter_low_wind_high_demand',
                 'summer_high_wind']

ssup_map = {'high':'cent',
            'medium':'mid',
            'low':'dist'}

scens = ['export',
         'import',
         'middle',
         'independence']


allscens = []

for scen in scens:
    for w in ws:
        allscens.append(scen+'-'+w)

NODES = pd.read_excel(r'C:/Users/Chris/Desktop/Coding/Python/PLEXOS/Node names and groupings.xlsx',
                  sheet_name='gsp_group',
                  header=0)
NODES = list(NODES['GSP areas'].unique())

REGIONS = list(set([i.split('-')[0] for i in NODES]))

#This is where input data folder is defined
INPUT = os.path.abspath(r'C:\Users\Chris\Desktop\Coding\Python\PLEXOS\Input data')

SCENMAP = os.path.join(INPUT, 'NEW_SCENARIO_MAP.xlsx')
SCENMAP = pd.read_excel(SCENMAP, index_col=0)

ZONES = REGIONS
ISLZONES = [i for i in REGIONS if 'MA' not in i]
ISLNODES = [i for i in NODES if i!='MA-0']

#%% FUNCTIONS

def cache_classes(db):
    CLASS_NAMES = {}
    rs = db.GetData("t_class", None)[0]
    while rs.EOF == False:
        CLASS_NAMES[rs["class_id"]] = rs["name"]
        rs.MoveNext()
    rs.Close()
    return CLASS_NAMES
    
def cache_system_collections(db):
    SYSTEM_COLLECTION_NAMES = {}
    rs = db.GetData("t_collection", None)[0]
    while rs.EOF == False:
        if rs["parent_class_id"] == 1:
            SYSTEM_COLLECTION_NAMES[rs["child_class_id"]] = rs["name"]
        rs.MoveNext()
    rs.Close()
    return SYSTEM_COLLECTION_NAMES

"""FOR SOME REASON THE PROP2ENUMID FUNCTION IS WHACK
    ADD ADDITIONAL ENUMS FOR OTHER SECTORS AS REQUIRED"""

    
def real_prop_enumid():
    """ returns a dict of the system enums which are used to look up the 
    integer value for each property name which is an input to the add_prop
    function
    
    Note that these enums will only work for those listed below, add
    more in if they are required. The simplest way is to use the dir() 
    function to list all the related enums, find the one you need, then add
    it to the below dict
    """
    
    enums = {
         'gen':SystemGeneratorsEnum,
         'datafile':SystemDataFilesEnum,
         'node':SystemNodesEnum,
         'line':SystemLinesEnum,
         'region':SystemRegionsEnum,
         'fuel':SystemFuelsEnum,
         'contract':SystemFinancialContractsEnum,
         'gasdemand':SystemGasDemandsEnum,
         'power2x':SystemPower2XEnum,
         'gasstorage':SystemGasStoragesEnum,
         'gasnode':SystemGasNodesEnum,
         'reserve':SystemReservesEnum,
         'battery':SystemBatteriesEnum,
         'heatnode':SystemHeatNodesEnum,
         'heatplant':SystemHeatPlantsEnum,
         'chargingstation':SystemChargingStationsEnum,
         'vehicle':SystemVehiclesEnum,
         'market':SystemMarketsEnum
         }
    
    out = {}    
    for name, enum in enums.items():
        # print(name)
        enumstr = str(enum).split('.')[-1].replace("'>","")
        out[name] = {}
        for meth in dir(enum):
            try:
                out[name][str(meth)] = int(eval(enumstr+'.'+meth))
            except:
                #this just ignores functions that are not related to PLEXOS
                pass
    return out

prop_enum = real_prop_enumid()

def db_setup(BASE, OUT):
    """ This sets up the database instance db in python, to which objects and 
    properties can be added"""
    # delete the modified file if it already exists
    if os.path.exists(OUT):
        os.remove(OUT)
    copyfile(BASE, OUT)
    
    # Create an object to store the input data
    db = DatabaseCore()
    db.DisplayAlerts = False
    DB_METHODS = dir(db)
    db.Connection(OUT)
    return db, DB_METHODS

def prop_type_check(params):
        """Checks that the input parameters for add_prop have the correct 
        datatypes, otherwise it will not work. 
        
        Note that some values that in the PLEXOS help are text based actually 
        have integer mappings, so make sure to use the integer input and not 
        the string/text"""
        
        check = {'MembershipId':[int],
              'EnumId':[int],
              'BandId':[int],
              'Value':[int,str],
              'DateFrom':[None],
              'DateTo':[None],
              'Variable':[None],
              'DataFile':[str, None],
              'Pattern':[None],
              'Scenario':[str,None],
              'Action':[int],
              'PeriodTypeId':[PeriodEnum]
              }
        
        out = {}
        
        for i in params:
            if type(params[i]) not in check[i] and (params[i]==None and None not in check[i]):
                raise Exception(i, params[i], check[i])
                out[i] = 'Type {} does not match required {}'.format(params[i],check[i])


db = DatabaseCore()
db.DisplayAlerts = False
db.Connection(r'C:/Users/Chris/Desktop/Coding/Python/PLEXOS/PLEXOS API/API docs and troubleshooting/API test/test_from_new (10.000 (beta 2023-09-22)).xml')
    
CLASS_NAMES = cache_classes(db)
SYSTEM_COLLECTION_NAMES = cache_system_collections(db)
db.Close()
    
def add_prop(db, collection_id, 
             cat_tech, node,  prop_name, prop_value, 
             band=1, scen=None, data_file=None, type_='gen'):
    '''
    Adds properties to an exiting PLEXOS object defined by the collection_id
    and cat_tech- which is combined below to form the child_name. Alter the 
    inputs to this to just include the child_name if the naming system is 
    different.
    
    Note that adding properties twice will result in an error when trying to 
    open the model, so make sure this isnt occurring.
    
    Optional inputs will be ignored if not included in the function call.
    
    type_ must belong to the 
    
    '''
    parent_name='System'
    
    # This is dependent on the nomenclature used for modelling the Scottish 
    # islands, where the object name depended on the cat_tech and node, 
    # otherwise this can be removed
    if type_ in ['gen','datafile']:
        child_name = cat_tech+'_'+node
        child_name = child_name.replace('.csv','')

    elif type_ in ['line','node','region','fuel',
                   'contract','gasdemand','power2x',
                   'gasstorage','gasnode','reserve','battery',
                   'heatnode','heatplant','chargingstation',
                   'vehicle','market']:
        child_name = node
    else: 
        raise Exception('Input type must be gen, line or node')

    '''
    Int32 GetMembershipID(
    	CollectionEnum nCollectionId,
    	String strParent,
    	String strChild
    	)
    '''
    # print(collection_id, parent_name, child_name)
    mem_id = db.GetMembershipID(collection_id, parent_name, child_name)
    '''
    Int32 PropertyName2EnumId(
    	String strParentClassName,
    	String strChildClassName,
    	String strCollectionName,
    	String strPropertyName
    	)
    '''
    # print(prop_name)
    if type_ not in prop_enum:
        raise Exception(f'The object type {type_} is not in the enum list')
    enum_id = prop_enum[type_][prop_name.replace(' ','')]
    
    '''
    Int32 AddProperty(
        Int32 MembershipId,
        Int32 EnumId,
        Int32 BandId,
        Double Value,
        Object DateFrom,
        Object DateTo,
        Object Variable,
        Object DataFile,
        Object Pattern,
        Object Scenario,
        Object Action,
        PeriodEnum PeriodTypeId
        )
    '''
    params = {'MembershipId':mem_id,
              'EnumId':enum_id,
              'BandId':band,
              'Value':prop_value,
              'DateFrom':None,
              'DateTo':None,
              'Variable':None,
              'DataFile':data_file,
              'Pattern':None,
              'Scenario':scen,
              'Action':0,
              'PeriodTypeId':PeriodEnum.Interval                     
              }
    # this makes sure the data input types are correct, otherwise will fail
    prop_type_check(params)
    db.AddProperty(mem_id, enum_id, band, 
                    prop_value, None, None, 
                    None, data_file, None, 
                    scen, 0, PeriodEnum.Interval)
    

    
    
def add_category(db, child_class_id, category):
    '''
    Add a category to a specified group of objects child_class_id, 
    where this is in the form ClassEnum.Fuels or ClassEnum.Generators
    
    To check what ClassEnums are called, use dir(ClassEnums)
    
    Int32 AddCategory(
        ClassEnum nClassId,
        String strCategory
        )
    '''
    cats = db.GetCategories(child_class_id)
    if len(category) > 0:
        if cats is None or category not in cats:
            db.AddCategory(child_class_id, category)
        else:
            print('Category {} already added to database'.format(category))
            
            
def add_object(db, child_class_id, child_name, category):
    '''
    
    Add a object to a specified group of objects child_class_id, 
    where this is in the form ClassEnum.Fuels or ClassEnum.Generators    
    To check what ClassEnums are called, use dir(ClassEnums)
    The category must exist or else it will fail
    
    Int32 AddObject(
        String strName,
        ClassEnum nClassId,
        Boolean bAddSystemMembership,
        String strCategory[ = None],
        String strDescription[ = None]
        )
    '''
    objs = db.GetObjects(child_class_id)
    if objs is None or child_name not in objs:
        db.AddObject(child_name, 
                     child_class_id, 
                     True,
                     category, 
                     'Added from Python')
    else:
        print('Object {} already added to database'.format(child_name))

