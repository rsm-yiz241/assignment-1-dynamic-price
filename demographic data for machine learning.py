#import packages 
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt  
from sklearn.model_selection import cross_val_score

#read the data 
cps = pd.read_csv('cps09mar.csv')
print(cps.head())

#transfrom data into DataFrame
cps_frame = pd.DataFrame(cps, columns = ['age','female','race','education', \
                                         'union', 'region', 'marital', 'everage earnings'])

#prepare for the data
print(cps_frame.isnull().any())
print(np.isnan(cps_frame).any())
cps_frame.fillna(0)

#set educaiton variableoat64').
education_to_label = {0:'0', 4:"0", 6:"0", 8:"0", 9:"0", 10:"0", \
                    11:"0", 12:"0", 13:"1", 14:"1", 16:"1", \
                    18:"1", 20:"1"}
cps_frame['educationtype'] = cps_frame['education'].map(education_to_label)
cps_frame['educationtype'] = cps_frame['educationtype'].fillna('0')

#set race variable
race_to_label = {1:"1", 2:"0", 2:"0", 4:"0", 5:"0", 6:"0", 7:"0", 8:"0", \
                 9:"0", 10:"0", 11:"0", 12:"0", 13:"0", 14:"0", \
                 15:"0", 16:"0", 17:"0", 18:"0", 19:"0", 20:"0", 21:"0"}
cps_frame['racetype'] = cps_frame['race'].map(race_to_label)
cps_frame['racetype'] = cps_frame['racetype'].fillna('0')

#set marital variable
marital_to_label = {1:"1", 2:"1", 3:"1", 4:"0", 5:"0", 6:"0", 7:"0"}
cps_frame['maritaltype'] = cps_frame['marital'].map(marital_to_label)
cps_frame['maritaltype'] = cps_frame['marital'].fillna('0')

#One Hot encoder
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(sparse = False)
ans = one.fit_transform(cps_frame)
print(ans)

#set x y
X = cps_frame[['age','female','racetype','educationtype','union','region','maritaltype']]
y = cps_frame[['everage earnings']]

#split into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Lasso
from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train, y_train)
'''print("Training set score: {:.2f}".format(Lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(Lasso.score(X_test, y_test)))
print("Number of features used:", np.sum(Lasso.coef_ != 0))'''

#Interactions 
from sklearn.preprocessing import PolynomialFeatures
interaction = PolynomialFeatures(degree=1, interaction_only=True, include_bias=False)
X_transformed = interaction.fit(X)


#PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_polied = poly.fit(X)

#Scaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train) 
X_test_scaled = scaler.transform(X_test)

#Automatic Feature Selection 

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

select = RFE(RandomForestRegressor(n_estimators=100, random_state=42),
             n_features_to_select=4)
select.fit(X_train, y_train)


'''select = RFE(estimator=RandomForestRegressor, n_features_to_select=2)
select.fit(X, y)'''

# split data set into training ,testing and validation sets
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, random_state=1)

#Grid Search with Cross-Validation
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
              'max_iter': [1, 10, 100, 1000, 10000, 100000]}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(Lasso(), param_grid, cv=4,
                          return_train_score=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
grid_search.fit(X_train, y_train)
print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("Best estimator:\n{}".format(grid_search.best_estimator_))



























