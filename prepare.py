# imports to run my functions
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

#-------------------------------------------------

def change_zillow(df):
    
    ''' a function to change data types of my columns and map names to fips'''
    
    # drop nulls
    df = df.dropna()
    
    # mapping fips code to the county
    df['fips'] = df.fips.map({ 06037.0: 'Los Angeles', 06059.0: 'Orange', 06111.0: 'Ventura'})

    # this series of steps change data types to ints as seen fit
    df["fips"] = df["fips"].astype(int)
    
    df["yearbuilt"] = df["yearbuilt"].astype(int)
    
    df["bedroomcnt"] = df["bedroomcnt"].astype(int)
    
    df["taxvaluedollarcnt"] = df["taxvaluedollarcnt"].astype(int)
    
    df["calculatedfinishedsquarefeet"] = df["calculatedfinishedsquarefeet"].astype(int)
    
    return df

#-------------------------------------------------

def rename_cols(df):
    
    ''' a function to rename columns and make them easier to read '''
    
    # renaming method performed 
    df = df.rename(columns={'bedroomcnt':'bedrooms', 
                            'bathroomcnt':'bathrooms', 
                            'calculatedfinishedsquarefeet':'sq_feet', 
                            'taxvaluedollarcnt':'tax_value',
                            'yearbuilt':'year_built',
                            'taxamount':'tax_amount'})
    return df

#-------------------------------------------------

def clean_zillow(df):
    
    '''
    takes data frame and changes datatypes and renames columnns, returns dataframe
    '''
    # calls other functions
    df = change_zillow(df)
    
    df = handle_outliers(df)
    
    df = rename_cols(df)
    
    # save df to csv
    df.to_csv("zillow.csv", index=False)

    return df

#-------------------------------------------------

def handle_outliers(df):
    
    '''handle outliers that do not represent properties likely for 99% of buyers and zillow visitors'''
    
    # this series of steps is how outliers were determined and removed
    df = df[df.bathroomcnt <= 6]
    
    df = df[df.bedroomcnt <= 6]

    df = df[df.taxvaluedollarcnt < 2_000_000]

    df = df[df.calculatedfinishedsquarefeet < 10000]
    
    df = df[df.yearbuilt > 1850]

    return df

#-------------------------------------------------