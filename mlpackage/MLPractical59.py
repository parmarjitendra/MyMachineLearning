# precision(= exactness / quality)
#           = how useful the search results are
#           = measure of exactness or quality


# recall ( = completeness / quantity / sensitivity)
#          = how complete the results are
#          = measure of completeness or quantity.



#F1 score = is the harmonic mean of precision and sensitivity(=recall)

# F1 = 2 x ( (precission x recall) / (precission + recall) )


#Question-1:

# precission=.77
# recall = .87
# F1 = ?



#Question-2:

# precission=.71
# recall = .55
# F1 = ?


#A.5) Cross Validation Classification Classification Report

import warnings
warnings.simplefilter(action="ignore", category=Warning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
filename = 'indians-diabetes.data.csv'
names=['preg', 'plas', 'pres', 'skin', 'test',
                'mass', 'pedi', 'age', 'class']

dataframe = pd.read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

test_size = 0.33
seed = 7
#,random_state=seed # seed value is used to fix the random no. generator

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=test_size , random_state=seed )

model = LogisticRegression()
model.fit(X_train, Y_train)  # 67% of training data

predicted = model.predict(X_test)  # 33% of test INPUT data


matrix = classification_report(Y_test, predicted)

print(matrix)
