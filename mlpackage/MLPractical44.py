#3.2. Normalize data.  ( 0 to 1)
# by manually
import numpy as np
from sklearn.preprocessing import Normalizer
X = np.array( [ [4, 1, 2, 2],
                [1, 3, 9, 3],
                [5, 7, 5, 1]  ]  )

transformer = Normalizer().fit_transform(X)  # fit does nothing.

#print( transformer.transform(X) )
print(transformer)