import pandas as pd
import numpy as np

def manual_downcast(frame):
    # int8
    int8_cols = ['garagecarcnt', 'storytypeid', 'typeconstructiontypeid',
                 ]
    for col in int8_cols:
        frame[col] = frame[col].fillna(-1).astype(np.int8)
    # int16
    int16_cols = ['garagetotalsqft', 'yardbuildingsqft17',
                  'yardbuildingsqft26', ]
    for col in int16_cols:
        frame[col] = frame[col].fillna(-1).astype(np.int16)
    # uint8
    uint8_cols = ['fireplacecnt', 'architecturalstyletypeid',
                  'airconditioningtypeid', 'bedroomcnt',
                  'roomcnt', 'threequarterbathnbr',
                  'fullbathcnt', 'buildingqualitytypeid',
                  'buildingclasstypeid', 'decktypeid',
                  'heatingorsystemtypeid', 'numberofstories',
                  'propertylandusetypeid', 'unitcnt',
                  'yearbuilt', 'assessmentyear',
                  'taxdelinquencyyear',
                  ]
    for col in uint8_cols:
        frame[col] = frame[col].fillna(0).astype(np.uint8)

    # uint16
    uint16_cols = ['basementsqft', 'calculatedbathnbr',
                   'finishedfloor1squarefeet', 'fips',
                   'regionidcounty',
                   ]
    for col in uint16_cols:
        frame[col] = frame[col].fillna(0).astype(np.uint16)

    # uint32
    uint32_cols = ['calculatedfinishedsquarefeet', 'finishedsquarefeet12',
                   'finishedsquarefeet6', 'finishedsquarefeet13',
                   'finishedsquarefeet15', 'finishedsquarefeet50',
                   'lotsizesquarefeet', 'parcelid',
                   'poolsizesum', 'regionidcity',
                   'regionidzip', 'regionidneighborhood',
                   'taxvaluedollarcnt', 'structuretaxvaluedollarcnt',
                   'landtaxvaluedollarcnt']
    for col in uint32_cols:
        frame[col] = frame[col].fillna(0).astype(np.uint32)

    # float16
    float16_cols = ['bathroomcnt']
    for col in float16_cols:
        frame[col] = frame[col].fillna(0).astype(np.float16)
    # float32
    frame['taxamount'] = frame['taxamount'].fillna(0).astype(np.float32)

    # Bools
    bool_cols = ['fireplaceflag', 'hashottuborspa',
                 'poolcnt', 'pooltypeid10',
                 'pooltypeid2', 'pooltypeid7',
                 'taxdelinquencyflag',
                 ]
    for col in bool_cols:
        frame[col] = frame[col].fillna(0).astype(np.bool)

    # Geo
    geo_cols = ['latitude', 'longitude',
                ]
    for col in geo_cols:
        frame[col] = (frame[col].fillna(0)/10000).astype(np.float32)

    # Part
    part_cols = ['rawcensustractandblock', 'censustractandblock']
    for col in part_cols:
        frame[col].fillna(0, inplace=True)
    frame['rawcensustractandblock-1'] = (frame['rawcensustractandblock'] / 10000).astype(np.int16)
    frame['rawcensustractandblock-2'] = (frame['rawcensustractandblock'] % 10000).astype(np.int16)
    frame['censustractandblock-1'] = (frame['censustractandblock'] / 100000000).astype(np.int16)
    frame['censustractandblock-2'] = (frame['censustractandblock'] % 10000 / 10000).astype(np.int16)
    frame['censustractandblock-3'] = (frame['censustractandblock'] % 10000).astype(np.int16)
    frame.drop(part_cols, axis=1, inplace=True)

    # Drop
    drop_cols = ['finishedfloor1squarefeet']
    for col in drop_cols:
        frame.drop(col, axis=1, inplace=True)

    return frame