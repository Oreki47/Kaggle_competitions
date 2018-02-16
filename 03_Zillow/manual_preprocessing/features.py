import pandas as pd
import numpy as np


def add_features(frame):
    # ===============================================================
    # Property related
    # ===============================================================
    # Rooms
    frame['N-TotalRoomsProd'] = frame['bathroomcnt'] * frame['bedroomcnt'] # Room product
    frame['N-TotalRoomsSum'] = frame['bathroomcnt'] + frame['bedroomcnt'] #Room sum
    frame['N-TotalRoomsNotPriRes'] = frame['bathroomcnt'] + frame['bedroomcnt'] - frame['roomcnt']
    frame['N-ExtraRooms'] = frame['roomcnt'] - frame['N-TotalRoomsSum']
    frame['N-BathRoomCntError'] = frame['threequarterbathnbr'] + frame['fullbathcnt'] - frame['bathroomcnt']
    frame['N-AvRoomSize'] = frame['calculatedfinishedsquarefeet'] / frame['roomcnt'] # Average room size
    # fix
    cols = ['N-AvRoomSize']
    for col in cols:
        frame[col].fillna(0, inplace=True)
    # Area
    frame['N-BaseProp'] = frame['basementsqft'] / frame['finishedsquarefeet15'] # Percentage of baseroom
    frame['N-GarageProp'] = frame['garagetotalsqft'] / frame['finishedsquarefeet15']  # Percentage of garage
    frame['N-LoftProp'] = frame['lotsizesquarefeet'] / frame['finishedsquarefeet15']  # Percentage of Loft
    frame['N-YardProp-1'] = frame['yardbuildingsqft17'] / frame['finishedsquarefeet15']  # Percentage of Yard1
    frame['N-YardProp-2'] = frame['yardbuildingsqft26'] / frame['finishedsquarefeet15']  # Percentage of Yard2
    frame['N-PoolProp'] = frame['poolsizesum'] / frame['finishedsquarefeet15'] # Percentage of Pool
    frame['N-ExternalSpaceSum'] = frame['basementsqft'] + frame['garagetotalsqft'] + \
                               frame['lotsizesquarefeet'] + frame['yardbuildingsqft17'] +\
                               frame['yardbuildingsqft26'] + frame['poolsizesum'] # Sum of External space
    frame['N-ExternalSpaceProp'] = frame['N-ExternalSpaceSum'] / frame['finishedsquarefeet15'] # Prop of External space

    frame['N-LivingAreaDiff'] = frame['calculatedfinishedsquarefeet'] - frame['finishedsquarefeet12']
    frame['N-LivingAreaError'] = frame['N-LivingAreaDiff'] / frame['finishedsquarefeet12'] # Percentage difference
    frame['N-FirstFloorProp-1'] = frame['finishedsquarefeet50'] / frame['finishedsquarefeet15'] # Percentage of 1st / all
    frame['N-FirstFloorProp-2'] = frame['finishedsquarefeet50'] / frame['finishedsquarefeet12'] # Percentage of 1st / living
    frame['N-LivingAreaProp-1'] = frame['finishedsquarefeet12'] / frame['finishedsquarefeet15'] # Percentage of living /all
    frame['N-ExtraSpace-1'] = frame['finishedsquarefeet15'] - frame['finishedsquarefeet12'] # Extra space

    cols = ['N-BaseProp', 'N-GarageProp', 'N-LoftProp', 'N-YardProp-1', 'N-YardProp-2',
            'N-PoolProp', 'N-ExternalSpaceProp', 'N-LivingAreaError', 'N-FirstFloorProp-1',
            'N-FirstFloorProp-2', 'N-LivingAreaProp-1', ]
    for col in cols:
        frame[col].fillna(-1, inplace=True)

    # Facilities
    frame['N-GarPoolAC'] = ((frame['garagecarcnt'] > 0) & (frame['pooltypeid10'] > 0) &
                            (frame['airconditioningtypeid'] != 5)) * 1
    frame['N-ACInd'] = (frame['airconditioningtypeid']!=5)*1 # Whether it has AC or not
    frame['N-HeatInd'] = (frame['heatingorsystemtypeid']!=13)*1 # Whether it has Heating or not

    # location
    frame["N-location"] = frame["latitude"] + frame["longitude"]
    frame["N-location-2"] = frame["latitude"] * frame["longitude"]
    frame["N-location-2round"] = frame["N-location-2"].round(-4)
    frame["N-latitude-round"] = frame["latitude"].round(-4)
    frame["N-longitude-round"] = frame["longitude"].round(-4)

    # Life
    frame['N-life'] = 2018 - frame['yearbuilt']

    # # Ratio of the built structure value to land area
    frame['N-ValueProp'] = frame['structuretaxvaluedollarcnt'] / frame['landtaxvaluedollarcnt']

    # ===============================================================
    # tax related
    # ===============================================================
    # Ratio of tax of property over parcel
    frame['N-ValueRatio'] = frame['taxvaluedollarcnt'] / frame['taxamount']

    # TotalTaxScore
    frame['N-TaxScore'] = frame['taxvaluedollarcnt'] * frame['taxamount']

    # # polnomials of tax delinquency year
    frame["N-taxdelinquencyyear-2"] = frame["taxdelinquencyyear"] ** 2
    frame["N-taxdelinquencyyear-3"] = frame["taxdelinquencyyear"] ** 3
    #
    # # Length of time since unpaid taxes
    frame['N-unpaid'] = 2018 - frame['taxdelinquencyyear']
    # # ===============================================================
    # # location related
    # # ===============================================================
    # # Number of properties in the zip
    zip_count = frame['regionidzip'].value_counts().to_dict()
    frame['N-zip_count'] = frame['regionidzip'].map(zip_count)
    #
    # # Number of properties in the city
    city_count = frame['regionidcity'].value_counts().to_dict()
    frame['N-city_count'] = frame['regionidcity'].map(city_count)
    #
    # # Number of properties in the city
    region_count = frame['regionidcounty'].value_counts().to_dict()
    frame['N-county_count'] = frame['regionidcounty'].map(city_count)  
    # # ===============================================================
    # # structuretaxvaluedollarcnt related
    # # ===============================================================
    # # polnomials of the variable
    frame["N-structuretaxvaluedollarcnt-2"] = frame["structuretaxvaluedollarcnt"] ** 2
    frame["N-structuretaxvaluedollarcnt-3"] = frame["structuretaxvaluedollarcnt"] ** 3
    
    # # Average structuretaxvaluedollarcnt by city
    group = frame.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
    frame['N-Avg-structuretaxvaluedollarcnt'] = frame['regionidcity'].map(group)
    #
    # # Deviation away from average
    frame['N-Dev-structuretaxvaluedollarcnt'] = \
        abs((frame['structuretaxvaluedollarcnt']
             - frame['N-Avg-structuretaxvaluedollarcnt'])) / frame['N-Avg-structuretaxvaluedollarcnt']
    return frame


def add_date_features(frame):
    frame["transaction_year"] = frame["transactiondate"].dt.year
    frame["transaction_month"] = frame["transactiondate"].dt.month
    frame["transaction_day"] = frame["transactiondate"].dt.day
    frame["transaction_quarter"] = frame["transactiondate"].dt.quarter
    frame.drop(["transactiondate"], inplace=True, axis=1)
    return frame
