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
  unitcnt,
  lotsizesquarefeet,
  heatingorsystemtypeid,
  garagetotalsqft,
  fullbathcnt,
 fireplacecnt,
 basementsqft,
     parcelid,
    assessmentyear,
    poolcnt,
    bedroomcnt, 
    bathroomcnt, 
    calculatedfinishedsquarefeet, 
    taxvaluedollarcnt, 
    yearbuilt,
    taxamount, 
    fips,
    latitude,
    longitude
    from properties_2017
    join propertylandusetype using (propertylandusetypeid)
    where propertylandusedesc = "Single Family Residential"
    """
    
    return pd.read_sql(sql, get_connection("zillow"))

#-------------------------------------------------

def change_zillow(df):
    
    ''' a function to change data types of my columns and map names to fips'''
    
    # replacing nulls with zero
    df['poolcnt'] = df['poolcnt'].replace(np.nan, 0)
    
    df['unitcnt'] = df['unitcnt'].replace(np.nan, 0)
    
    df['heatingorsystemtypeid'] = df['heatingorsystemtypeid'].replace(np.nan, 0)
    
    df['garagetotalsqft'] = df['garagetotalsqft'].replace(np.nan, 0)
    
    df['fullbathcnt'] = df['fullbathcnt'].replace(np.nan, 0)
   
    df['fireplacecnt'] = df['fireplacecnt'].replace(np.nan, 0)
    
    df['basementsqft'] = df['basementsqft'].replace(np.nan, 0)
    
    # drop nulls
    df = df.dropna()
    
    # mapping fips code to the county
    df['fips'] = df.fips.map({ 06037.0: 'Los Angeles', 06059.0: 'Orange', 06111.0: 'Ventura'})
    
    
     # change data types
    df['unitcnt'] = df['unitcnt'].astype(int)
    
    df['heatingorsystemtypeid'] = df['heatingorsystemtypeid'].astype(int)
    
    df['garagetotalsqft'] = df['garagetotalsqft'].astype(int)
    
    df['fullbathcnt'] = df['fullbathcnt'].astype(int)
    
    df['fireplacecnt'] = df['fireplacecnt'].astype(int)
    
    df['basementsqft'] = df['basementsqft'].astype(int)
    
    df["yearbuilt"] = df["yearbuilt"].astype(int)
    
    df["bedroomcnt"] = df["bedroomcnt"].astype(int)
    
    df["taxvaluedollarcnt"] = df["taxvaluedollarcnt"].astype(int)
    
    df["calculatedfinishedsquarefeet"] = df["calculatedfinishedsquarefeet"].astype(int)
    
    df["poolcnt"] = df["poolcnt"].astype(int)
    
    df["assessmentyear"] = df["assessmentyear"].astype(int)
    
    df["longitude"] = df["longitude"].astype(int)
    
    df["latitude"] = df["latitude"].astype(int)
     
    
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
    
    df_d = pd.get_dummies(df,columns= ['bedrooms','bathrooms','assessmentyear','poolcnt','unitcnt','heatingorsystemtypeid','fullbathcnt','fireplacecnt'],drop_first = True)
        
    # save df to csv
    df.to_csv("zillow.csv", index=False)

    return df, df_d

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

def wrangle_zillow():
    """
   Acquires zillow data and uses the clean function to call other functions and returns a clean data        frame with new names, dropped nulls, new data types.
    """

    # create a variable name
    filename = "zillow2017.csv"

    # searching for that variable name
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        
        # perform other functions to make a new data acquistion
        df = get_zillow_data()

        df = clean_zillow(df)
        
        df.to_csv('zillow2017.csv')

    df_d = df.copy()
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

    ''' a function to scale my data appropriately ''' 
    
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
 