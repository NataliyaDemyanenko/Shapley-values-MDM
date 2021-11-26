# import required libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import shap                
shap.initjs()

# read the data
df = pd.read_csv('Advertising.csv',dtype={'sales': np.float64})

# Create train test  split.
Y = df['sales']
X =  df[['TV', 'radio', 'newspaper']]
# Split the data into train and test data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fit Random Forest Regressor Model
rf = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=20)
rf.fit(X_train, Y_train) 

# Predict
Y_predict=rf.predict(X_test)

# RMSE
print('RMSE: ',mean_squared_error(Y_test, Y_predict)**(0.5))

# create explainer model by passing trained model to shap
explainer = shap.TreeExplainer(rf)

# Get SHAP values for the training data from explainer model
shap_values_train = explainer.shap_values(X_train)

# Get SHAP values for the test data from explainer model
shap_values_test = explainer.shap_values(X_test)

# Create a dataframe of the shap values for the training set and the test set
df_shap_train=pd.DataFrame(shap_values_train,columns=['TV_Shap','radio_Shap','newspaper_Shap'])
df_shap_test=pd.DataFrame(shap_values_test,columns=['TV_Shap','radio_Shap','newspaper_Shap'])

# Base Value: Training data 
print('base array: ','\n',shap_values_train.base[0], '\n')
print('basevalue: ',shap_values_train.base[0][3])

# Base Value: Test data 
#print('base array: ','\n',shap_values_test.base[0], '\n')
#print('basevalue: ',shap_values_test.base[0][3])

# Base Value is is calculated from the training dataset so its same for test data
base_value=shap_values_train.base[0][3][0]

# Create a  new column for base value
df_shap_train['BaseValue']= base_value

# Add shap values and Base Value
df_shap_train['(ShapValues + BaseValue)']=df_shap_train.iloc[:,0]+df_shap_train.iloc[:,1]+df_shap_train.iloc[:,2]+base_value

# Also create a new columns for the prediction values from training dataset
df_shap_train['prediction']=pd.DataFrame(list(rf.predict(X_train)))

# Note: Prediction Column is added to compare the values of Predicted with the sum of Base and SHAP Values

df_shap_train.head()

#SHAP Summary plot for training data
shap.summary_plot(shap_values_train,X_train)

#SHAP Summary plot for test data
shap.summary_plot(shap_values_test,X_test)

#Plot to see the average impat on the model output by each feature.
shap.summary_plot(shap_values_train,X_train,plot_type='bar',color=['#093333','#ffbf00','#ff0000'])

# Market Response for TV : Plot between TV ads spend and SHAP Values
shap.dependence_plot("TV",shap_values_train,X_train,interaction_index=None)

# Interaction Summary: Get the interaction values
shap_interaction_values = shap.TreeExplainer(rf).shap_interaction_values(X_train)

# Plot Interaction Summary
shap.summary_plot(shap_interaction_values, X_train)