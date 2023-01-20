# Evaluating Home Prices by Brandon Navarrete
<a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-013243.svg?logo=python&logoColor=white"></a>
<a href="#"><img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458.svg?logo=pandas&logoColor=white"></a>
<a href="#"><img alt="NumPy" src="https://img.shields.io/badge/Numpy-2a4d69.svg?logo=numpy&logoColor=white"></a>
<a href="#"><img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-8DF9C1.svg?logo=matplotlib&logoColor=white"></a>
<a href="#"><img alt="seaborn" src="https://img.shields.io/badge/seaborn-65A9A8.svg?logo=pandas&logoColor=white"></a><a href="#"><img alt="sklearn" src="https://img.shields.io/badge/sklearn-4b86b4.svg?logo=scikitlearn&logoColor=white"></a>
<a href="#"><img alt="SciPy" src="https://img.shields.io/badge/SciPy-1560bd.svg?logo=scipy&logoColor=white"></a>

# Project Description

Usinng statistical test, we will analyze the key drivers of property values for single family properties that were sold in 2017. Using our key drivers we will develope a machine learning model (regression model) to predict the properties assessed value, and provide recommendations on make a better model for our predictions.

# Project Goal:

* Find key assessed value drivers in single family properties sold in 2017 

* Construct a ML regressionn model that accurately predicts property tax value (assessed value) 

* Deliver a report that a non-data scientist can read through and understand what steps were taken, why and what was the outcome?  


# Inital Questions

* Does having an above average bedroom count affect the `tax_value`
* 
* Does having an above average bathroom count affect the `tax_value`
* 
* Does `sq_feet`  affect the `tax_value`


# The Plan

* Acquire data from Sequel Ace database
  * Pull Columns :
      `bedroomcnt`
      `bathroomcnt`
      `calculatedfinishedsquarefeet` 
      `taxvaluedollarcnt`
      `yearbuilt`
      `taxamount`
      `fips`

* Prepare Data
  * drop nulls  
  * change data types to ints
  * rename columns    
  * create dummies  
  * drop outliers    
 
  
## :memo: Explore data in search of drivers of churn /  Answer the following initial questions
  
* what is correlated with `tax_value` wether that be negative or positive?
     
* Does having an above average bedroom count affect the `tax_value`?
       
* Does having an above average bathroom count affect the `tax_value`?
 
* Does `sq_feet`  affect the `tax_value`?
 
       

# Data Dictionary

**Variable** |    **Value**    | **Meaning**
---|---|---
*Bedrooms* | Integer ranging from 1-6 | Number of bedrooms in home 
*Bathrooms* | Float ranging from 0.5-6.5| Number of bathrooms in home including fractional bathrooms
*Square Feet* | Float number | Calculated total finished living area of the home 
*Age* | Integer |  This indicate the age of the property in 2017, calculated using the year the principal residence was built 
*Tax Value* | Float number | The total tax assessed value of the parcel
*Tax Amount*| Float number | The total property tax paid for a given year


