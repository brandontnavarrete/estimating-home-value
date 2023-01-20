# imports to run my functions
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import scipy.stats as stats
import statistics as s




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
import sklearn.preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
# sql ace credentials
import env

# for chart image
from IPython.display import Image


def bathroom_ttest(train):
    
    a = 0.05
    
    avg_bath = train['bathrooms'].mean()
    
    above_bath = train[train.bathrooms >avg_bath].bathrooms
    below_bath = train[train.bathrooms <= avg_bath].bathrooms
    
    t, p = stats.ttest_ind(above_bath, below_bath, equal_var=False)

    if p / 2 > a:
        print("We fail to reject $H_{0}$")
    elif t < 0:
        print("We fail to reject $H_{0}$")
    else:
        print("We reject $H_{0}$")
        
    print( t, p )
    
#-----------------------------------------------------

def bedroom_ttest(train):
    
    a = 0.05
    
    avg_bed = train['bedrooms'].mean()
    
    abov_bed_sample = train[train.bedrooms > avg_bed].bedrooms
    
    t, p = stats.ttest_1samp(abov_bed_sample , avg_bed)
    
    if p > a:
        print('We fail to reject $H_{0}$ : There is not a significant difference in the mean')
    else:
        print("We reject $H_{0}$ : There is some significant difference in the mean")
        
    print (t,p)
    
#-----------------------------------------------------

def baseline(y_train,y_validate):
    
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # predict tax value mean

    y_train['t_value_mean'] = y_train['tax_value'].mean()
    y_validate['t_value_mean'] = y_validate['tax_value'].mean()

    # predict tax value median
    y_train['t_value_median'] = y_train['tax_value'].median()
    y_validate['t_value_median'] = y_validate['tax_value'].median()

    # RMSE OF tax mean
    rmse_train_mean = mean_squared_error(y_train.tax_value, y_train.t_value_mean)**(1/2)
    rmse_validate_mean = mean_squared_error(y_validate.tax_value, y_validate.t_value_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train_mean, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate_mean, 2),
      "\nDifference: ", abs(rmse_train_mean - rmse_validate_mean))


    # RMSE Of tax median
    rmse_train_median = mean_squared_error(y_train.tax_value, y_train.t_value_median)**(1/2)
    rmse_validate_median = mean_squared_error(y_validate.tax_value, y_validate.t_value_median)**(1/2)
    print('\n')
    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train_median, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate_median, 2),
      "\nDifference: ", abs(rmse_train_median - rmse_validate_median))

    return y_train,y_validate

#-----------------------------------------------------
def lm_rmse(x_train_scaled,y_train,x_validate_scaled,y_validate):
    # create object
    lm = LinearRegression(normalize= True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series
    lm.fit(x_train_scaled, y_train['tax_value'])

    # predict train
    y_train['tax_lm_pred'] = lm.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train_lm = mean_squared_error(y_train['tax_value'], y_train['tax_lm_pred'])**(1/2)
                                                                                   
    # predict validate
    y_validate['tax_lm_pred'] = lm.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate_lm = mean_squared_error(y_validate['tax_value'], y_validate['tax_lm_pred'])**(1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train_lm, 
      "\nValidation/Out-of-Sample: ", rmse_validate_lm,
      "\nDifference: ", abs(rmse_train_lm - rmse_validate_lm))

    return y_train,y_validate


#-----------------------------------------------------
def lars_rmse(x_train_scaled,y_train,x_validate_scaled,y_validate):
    # object instance
    lars = LassoLars(alpha = .3)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(x_train_scaled, y_train['tax_value'])

    # predict train
    y_train['t_value_lrs'] = lars.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train_lrs = mean_squared_error(y_train['tax_value'], y_train['t_value_lrs'])**(1/2)

    # predict validate
    y_validate['t_value_lrs'] = lars.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate_lrs = mean_squared_error(y_validate['tax_value'], y_validate['t_value_lrs'])**(1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train_lrs, 
      "\nValidation/Out-of-Sample: ", rmse_validate_lrs,
       "\nDifference: ", abs(rmse_train_lrs - rmse_validate_lrs))
    
    return y_train,y_validate


#-----------------------------------------------------

def glm_rmse(x_train_scaled,y_train,x_validate_scaled,y_validate):
    
    # create the model object
    glm = TweedieRegressor(power=0, alpha=0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(x_train_scaled, y_train['tax_value'])

    # predict train
    y_train['t_value_tr'] = glm.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train_tr = mean_squared_error(y_train['tax_value'], y_train['t_value_tr'])**(1/2)

    # predict validate
    y_validate['t_value_tr'] = glm.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate_tr = mean_squared_error(y_validate['tax_value'], y_validate['t_value_lrs'])**(1/2)

    print("RMSE for GLM using Tweedie, power=0 & alpha=0\nTraining/In-Sample: ", rmse_train_tr, 
      "\nValidation/Out-of-Sample: ", rmse_validate_tr,
       "\nDifference: ", abs(rmse_train_tr - rmse_validate_tr))
    
    return y_train,y_validate

    