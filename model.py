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

# setting alpha
a = 0.05
    

    
#-----------------------------------------------------
def bathroom_ttest(train):
    
    '''a function which takes in a train data set and calculates and returns a specific t-test for the mean of bathroom data'''
    
    # setting alpha
    a = 0.05
    # creating a mean of bathrooms
    avg_bath = train['bathrooms'].mean()
    
    # using the mean to create mask data frames on either side
    above_bath = train[train.bathrooms >avg_bath].bathrooms
    below_bath = train[train.bathrooms <= avg_bath].bathrooms
    
    # performing a t test
    t, p = stats.ttest_ind(above_bath, below_bath, equal_var=False)

    # if statement to return our results
    if p / 2 > a:
        print("We fail to reject $H_{0}$")
    elif t < 0:
        print("We fail to reject $H_{0}$")
    else:
        print("We reject $H_{0}$")
        
    print( t, p )
    
#-----------------------------------------------------

def bedroom_ttest(train):
    
    # creating an average metric of bedroom count
    avg_bed = train['bedrooms'].mean()
    
    # creating a mask for a new data frame
    abov_bed_sample = train[train.bedrooms > avg_bed].bedrooms
    
    # perform a t test
    t, p = stats.ttest_1samp(abov_bed_sample , avg_bed)
    
    # if statement to return our results
    if p > a:
        print('We fail to reject $H_{0}$ : There is not a significant difference in the mean')
    else:
        print("We reject $H_{0}$ : There is some significant difference in the mean")
        
    print (t,p)
    
#-----------------------------------------------------

def baseline(y_train,y_validate):
    
    '''a function to return our baseline and create a dataframe to hold all the models and their features'''
    
    # turning our series into a data frame
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

    # print our results for an easier interpretion
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
    
    # creating a new series to hold our results of all model performance
    evals = {'metric': ['RMSE'], 'model': ['baseline'],'rmse':[rmse_train_median],'overfit':[rmse_train_median - rmse_validate_median]}

    # creating a data frame from our series to pass on
    evals = pd.DataFrame(data=evals)
    
    return y_train,y_validate,evals

#-----------------------------------------------------
def lm_rmse(x_train_scaled,y_train,x_validate_scaled,y_validate,evals):
   
    '''a function to create our linear regression model to run on train and validate'''
    
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

    evals1 = {'metric': 'RMSE', 'model': 'linear regression','rmse':rmse_train_lm,'overfit':rmse_train_lm - rmse_validate_lm}

    evals = evals.append(evals1,ignore_index=True)
        
    return y_train,y_validate,evals


#-----------------------------------------------------
def lars_rmse(x_train_scaled,y_train,x_validate_scaled,y_validate,evals):
    
    '''a function to create our lars model to run on train and validate'''

    
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
    
    evals1 = {'metric': 'RMSE', 'model': 'Lars','rmse':rmse_train_lrs,'overfit':rmse_train_lrs - rmse_validate_lrs}

    evals = evals.append(evals1,ignore_index=True)
        
    
    return y_train,y_validate,evals


#-----------------------------------------------------

def glm_rmse(x_train_scaled,y_train,x_validate_scaled,y_validate,evals):
    
    '''a function to create our glm model to run on train and validate'''

    
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
    
    evals1 = {'metric': 'RMSE', 'model': 'glm','rmse':rmse_train_tr,'overfit':rmse_train_tr - rmse_validate_tr}

    evals = evals.append(evals1,ignore_index=True)
        
    
    return y_train,y_validate,evals


#-----------------------------------------------------

def pr_rmse(x_train_scaled,y_train,x_validate_scaled,y_validate,evals):
   
    '''a function to create our ploynomial regression dataframe and linear regression model to run on train and validate'''


    # create object 
    pf = PolynomialFeatures(degree=1)

    # fit and transform X_train_scaled
    x_train_degree2 = pf.fit_transform(x_train_scaled)

    # transform x_validate_scaled 
    x_validate_degree2 = pf.transform(x_validate_scaled)

    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(x_train_degree2, y_train['tax_value'])

    # predict train
    y_train['t_value_lm2'] = lm2.predict(x_train_degree2)

    # evaluate: rmse
    rmse_train_lm2 = mean_squared_error(y_train['tax_value'], y_train['t_value_lm2'])**(1/2)

    # predict validate
    y_validate['t_value_lm2'] = lm2.predict(x_validate_degree2)

    # evaluate: rmse
    rmse_validate_lm2 = mean_squared_error(y_validate['tax_value'], y_validate['t_value_lm2'])**(1/2)

    print("RMSE for Polynomial Model, degrees=1\nTraining/In-Sample: ", rmse_train_lm2, 
      "\nValidation/Out-of-Sample: ", rmse_validate_lm2,
       "\nDifference: ", abs(rmse_train_lm2 - rmse_validate_lm2))
    
    evals1 = {'metric': 'RMSE', 'model': 'polynomial','rmse':rmse_train_lm2,'overfit':rmse_train_lm2 - rmse_validate_lm2}

    evals = evals.append(evals1,ignore_index=True)
        
    
    
    
    return y_train,y_validate,evals

#-----------------------------------------------------

def model_compare(evals):
    
    ''' function to return a bar graph to compare baseline and model RMSE'''
    
    idx = np.where ((evals['model'] == 'baseline') | (evals['model'] == 'Lars') )
    
    # Visualizing model performance
    plt.figure(figsize = (15,7))
    # Load model performance from csv
    plt.title('Lars Model RMSE Performs Better than Baseline', fontsize = 18)
    sns_plot = sns.barplot(x='model', y='rmse', data = evals.loc[idx], palette = ['red','teal'])

    
#-----------------------------------------------------

    
def test_lars_rmse(x_train_scaled,y_train,y_test,x_test_scaled):
    
    ''' a function to test our test data set on our best model and return the results'''
    
    y_test = pd.DataFrame(y_test)
    y_test['baseline'] = y_test.tax_value.mean()
    
    # object instance
    lars = LassoLars(alpha = .3)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(x_train_scaled, y_train['tax_value'])

    # predict train
    y_test['lars_test_predict'] = lars.predict(x_test_scaled)

    # evaluate: rmse
    lars_test_rmse = mean_squared_error(y_test.tax_value,y_test['lars_test_predict'], squared = False)
    
    y_test['lars_test_rmse'] = lars_test_rmse
    
    return y_test

#-----------------------------------------------------

def test_visual(y_test):
    
    ''' a function to create a scatter plot with baseline and model comparison to the actual value tax ''' 

    # running the best model on test
    plt.figure(figsize=(16,8))
    plt.title('Lars Performance', fontsize = 18)


    # Plotting the model predictions
    plt.plot(y_test.lars_test_predict, y_test.lars_test_predict, alpha=0.7, linewidth=3, color='green',label = 'model prediction')

    # Plotting baseline predictions
    plt.plot(y_test, y_test.baseline ,alpha=0.4, linewidth= 3,color='red', label = 'baseline prediction')


    # Plotting the tax values compared to the models predictions
    plt.scatter(y_test['tax_value'],y_test['lars_test_predict'], alpha = 0.9, color = 'gray', s=100)

    plt.show()
    