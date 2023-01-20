# imports to run my functions
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
# sql ace credentials
import env


#----------------------------------------------

# setting connectiong to sequel server using env

def get_connection(db, user=env.username, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#----------------------------------------------
 
# acquiring zillow data using a get_connection

def get_zillow_data():
    """returns a dataframe from SQL of all 2017 properties that are single family residential"""

    sql = """
    select 
    bedroomcnt, 
    bathroomcnt, 
    calculatedfinishedsquarefeet, 
    taxvaluedollarcnt, 
    yearbuilt,
    taxamount, 
    fips
    from properties_2017
    join propertylandusetype using (propertylandusetypeid)
    join predictions_2017 using (parcelid)
    where propertylandusedesc = "Single Family Residential"
    """
    return pd.read_sql(sql, get_connection("zillow"))

#-------------------------------------------------

def change_zillow(df):
    
    df = df.dropna()
    
    df["fips"] = df["fips"].astype(int)
    
    df["yearbuilt"] = df["yearbuilt"].astype(int)
    
    df["bedroomcnt"] = df["bedroomcnt"].astype(int)
    
    df["taxvaluedollarcnt"] = df["taxvaluedollarcnt"].astype(int)
    
    df["calculatedfinishedsquarefeet"] = df["calculatedfinishedsquarefeet"].astype(int)
    
    return df

#-------------------------------------------------

def rename_cols(df):
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
    
    df = change_zillow(df)
    
    df = handle_outliers(df)
    
    df = rename_cols(df)
    
    df_d = pd.get_dummies(df,columns= ['bedrooms','bathrooms'],drop_first = True)

    df.to_csv("zillow.csv", index=False)

    return df, df_d

#-------------------------------------------------

def handle_outliers(df):
    '''handle outliers that do not represent properties likely for 99% of buyers and zillow visitors'''
    
    df = df[df.bathroomcnt <= 6]
    
    df = df[df.bedroomcnt <= 6]

    df = df[df.taxvaluedollarcnt < 2_000_000]

    df = df[df.calculatedfinishedsquarefeet < 10000]
    
    df = df[df.yearbuilt > 1850]

    return df

#-------------------------------------------------

def wrangle_zillow():
    """
   Acquires zillow data and uses the clean function to call other functions and returns a clean data        frame with new names, dropped nulls, new data types.
    """

    filename = "zillow2017.csv"

    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        
        df = get_zillow_data()

        df = clean_zillow(df)
        
        df.to_csv('zillow_2017.csv')

    return df , df_d

#-------------------------------------------------

def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    
    print(train.shape , validate.shape, test.shape)

          
    return train, validate, test


#-------------------------------------------------

def x_and_y(train,validate,test,target):
    
    """
    splits train, validate, and target into x and y versions
    """

    x_train = train.drop(columns= target)
    y_train = train[target]

    x_validate = validate.drop(columns= target)
    y_validate = validate[target]

    x_test = test.drop(columns= target)
    y_test = test[target]

    return x_train,y_train,x_validate,y_validate,x_test, y_test


#-------------------------------------------------

def scaled_data(x_train,x_validate,x_test,num_cols,return_scaler = False):

    # intializing scaler
    scaler = MinMaxScaler()
    
    # fit scaler
    scaler.fit(x_train[num_cols])
    
    # creating new scaled dataframes
    x_train_s = scaler.transform(x_train[num_cols])
    x_validate_s = scaler.transform(x_validate[num_cols])
    x_test_s = scaler.transform(x_test[num_cols])

    # making a copy of train to hold scaled version
    x_train_scaled = x_train.copy()
    x_validate_scaled = x_validate.copy()
    x_test_scaled = x_test.copy()

    x_train_scaled[num_cols] = x_train_s
    x_validate_scaled[num_cols] = x_validate_s
    x_test_scaled[num_cols] = x_test_s

    if return_scaler:
        return scaler, x_train_scaled, x_validate_scaled, x_test_scaled
    else:
        return x_train_scaled, x_validate_scaled, x_test_scaled
    
#-------------------------------------------------

#-------------------------------------------------
 