# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
STEP 1:
Read the given Data
STEP 2:
Clean the Data Set using Data Cleaning Process
STEP 3:
Apply Feature selection techniques to all the features of the data set
STEP 4:
Save the data to the file


CODE:
```
from sklearn.datasets import load_boston
boston_data=load_boston()
import pandas as pd
boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston['MEDV'] = boston_data.target
dummies = pd.get_dummies(boston.RAD)
boston = boston.drop(columns='RAD').merge(dummies,left_index=True,right_index=True)
X = boston.drop(columns='MEDV')
y = boston.MEDV
boston.head(10)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from math import sqrt

cv = KFold(n_splits=10, random_state=None, shuffle=False)
classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
print("R_squared: " + str(round(r2_score(y,y_pred),2)))

boston.var()

X = X.drop(columns = ['NOX','CHAS'])
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
print("R_squared: " + str(round(r2_score(y,y_pred),2)))

# Filter Features by Correlation
import seaborn as sn
import matplotlib.pyplot as plt
fig_dims = (12, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sn.heatmap(boston.corr(), ax=ax)
plt.show()
abs(boston.corr()["MEDV"])
abs(boston.corr()["MEDV"][abs(boston.corr()["MEDV"])>0.5].drop('MEDV')).index.tolist()
vals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
for val in vals:
    features = abs(boston.corr()["MEDV"][abs(boston.corr()["MEDV"])>val].drop('MEDV')).index.tolist()
    
    X = boston.drop(columns='MEDV')
    X=X[features]
    
    print(features)

    y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
    print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
    print("R_squared: " + str(round(r2_score(y,y_pred),2)))

# Feature Selection Using a Wrapper

boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston['MEDV'] = boston_data.target
boston['RAD'] = boston['RAD'].astype('category')
dummies = pd.get_dummies(boston.RAD)
boston = boston.drop(columns='RAD').merge(dummies,left_index=True,right_index=True)
X = boston.drop(columns='MEDV')
y = boston.MEDV

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs1 = SFS(classifier_pipeline, 
           k_features=1, 
           forward=False, 
           scoring='neg_mean_squared_error',
           cv=cv)

X = boston.drop(columns='MEDV')
sfs1.fit(X,y)
sfs1.subsets_

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

boston[['CRIM','RM','PTRATIO','LSTAT','MEDV']].corr()

boston['RM*LSTAT']=boston['RM']*boston['LSTAT']

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

sn.pairplot(boston[['CRIM','RM','PTRATIO','LSTAT','MEDV']])

boston = boston.drop(boston[boston['MEDV']==boston['MEDV'].max()].index.tolist())

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT','RM*LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

boston['LSTAT_2']=boston['LSTAT']**2

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

```


OUPUT:
![1](https://github.com/LavanyaSIT/Ex-07-Feature-Selection/assets/130207418/d267d266-050d-4258-bf65-306b956d26f3)
![3](https://github.com/LavanyaSIT/Ex-07-Feature-Selection/assets/130207418/ae6bcca6-e4f9-4586-9d87-b571c1d99c44)
![5](https://github.com/LavanyaSIT/Ex-07-Feature-Selection/assets/130207418/27358d63-d445-45d1-817e-527e83f275c7)
![6](https://github.com/LavanyaSIT/Ex-07-Feature-Selection/assets/130207418/ea038e5f-b6fa-4572-a2f1-63a0953555f0)
![8](https://github.com/LavanyaSIT/Ex-07-Feature-Selection/assets/130207418/d8a0b7f2-f9ad-48d2-8260-65049281ceed)
![9](https://github.com/LavanyaSIT/Ex-07-Feature-Selection/assets/130207418/b69734c1-41bd-4426-81c6-4575285bc2cd)
![10](https://github.com/LavanyaSIT/Ex-07-Feature-Selection/assets/130207418/06f659ef-89d9-4baa-b460-c129e0fe8355)
![12](https://github.com/LavanyaSIT/Ex-07-Feature-Selection/assets/130207418/d9d91e2d-f96b-44d5-8ce8-7303eddf35fe)
![14](https://github.com/LavanyaSIT/Ex-07-Feature-Selection/assets/130207418/c99aa13e-cfc0-4d63-8e7f-1ec35b8f0b31)


RESULT:
The various feature selection techniques has been performed on a dataset and saved the data to a file.
