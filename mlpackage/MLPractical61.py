#2) Mean Squared Error.
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

filename = 'housing.csv'

hnames=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

dataframe = pd.read_csv(filename,  names=hnames)
array = dataframe.values

X = array[: , 0:13]
Y = array[:,13]

kfold = KFold(n_splits=10 )
model = LinearRegression()

scoringMethod = 'neg_mean_squared_error'    #MSE = value of mse is always high than mae
results = cross_val_score(model, X, Y, cv=kfold,  scoring=scoringMethod)

print( "MSE: %.3f (%.3f)" % ( results.mean(), results.std() )     )
