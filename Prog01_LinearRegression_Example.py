
#################################################################################################################################
######################################## PROGRAM WITH EXAMPLE LINEAR REGRESSION #################################################
#################################################################################################################################

#### Good Url to refer to for Scikit Learn ###########
# http://scikit-learn.org/stable/index.html



import pandas as pd
import numpy as np


#################################### Linear Regression Using StatModels #########################################################
import statsmodels.api as sm

from sklearn import datasets ## imports datasets from scikit-learn
data = datasets.load_boston() ## loads Boston dataset from datasets library 

# define the data/predictors as the pre-set feature names  
df = pd.DataFrame(data.data, columns=data.feature_names)
df.dtypes
df.describe()

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])
target.dtypes
target.describe()

# Execute linear Regression. Note the difference in argument order
model = sm.OLS(target, df).fit()
predictions = model.predict(df) # make the predictions by the model

# Print out the statistics
model.summary()


df = sm.add_constant(df) ## let's add an intercept (beta_0) to our model
# Execute linear Regression. Note the difference in argument order
model = sm.OLS(target, df).fit()
predictions = model.predict(df) # make the predictions by the model

# Print out the statistics
model.summary()


#################################### Linear Regression Using SK Learn #########################################################


from sklearn import linear_model

#Fit Linear Regression Model
lm = linear_model.LinearRegression()
skmodel = lm.fit(df, target)

skpredictions = lm.predict(df)

# R-Sqaure
lm.score(df, target)

# Co-efficients & Intercept
print('Intercept: \n', lm.intercept_)
print('Coefficients: \n', lm.coef_)


###### Another Function ##############
# https://stackabuse.com/linear-regression-in-python-with-scikit-learn/

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(df, target) 

coeff_df = pd.DataFrame(np.append(regressor.intercept_,regressor.coef_), df.columns, columns=['Variable','Coefficient']) 
#coeff_df = pd.DataFrame(np.append(regressor.intercept_,regressor.coef_), df.columns, columns=['Coefficient'])  
coeff_df  






