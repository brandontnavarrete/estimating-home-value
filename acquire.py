# imports to run my functions
import pandas as pd
import os
import numpy as np
import env

# setting connectiong to sequel server using env

def get_connection(db, user=env.username, host=env.host, password=env.password):
    
    ''' a function to handle my sql ace creds'''
    
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
    where propertylandusedesc = "Single Family Residential"
    """
    return pd.read_sql(sql, get_connection("zillow"))
