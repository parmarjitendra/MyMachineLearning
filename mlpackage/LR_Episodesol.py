

#tryed by me
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('LR_Episodes.csv')
print(data)
print(data.shape)

X1 = data.iloc[ : ,0:1].values
Y1 = data.iloc[ : ,1].values
print(X1)
print(Y1)

print("X1.shape=" , X1.shape,"\n X1 \n",X1)
print("Y2.shape=" ,Y1.shape,"\n X1 \n",Y1)

X2 = data.iloc[ : ,2:3].values
Y2 = data.iloc[ : ,3].values
print(X2)
print(Y2)

print("X2.shape=" , X2.shape,  "\n X1 \n" ,X2)
print("Y2.shape=",Y2.shape,"\n X1 \n",Y2)

from sklearn.linear_model import LinearRegression

lin = LinearRegression()
lin.fit(X1, Y1)
lin.fit(X2,Y2)
y_dash1 = lin.predict(X1)
y_dash2 = lin.predict(X2)

plt.scatter(X1, Y1, color='blue')
plt.plot(X1, y_dash1 , color='red',)

plt.scatter(X2, Y2, color='black')
plt.plot(X2, y_dash2 , color='yellow')

plt.title('LR_Episode')
plt.xlabel('Flash_episode_number,Flash_us_viewers')
plt.ylabel('GoT_episode_number,GoT_us_viewers')

plt.show()



from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=8)
X_poly1 = poly.fit_transform(X1)
X_poly2 = poly.fit_transform(X2)


#poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly1, Y1)
lin2.fit(X_poly2, Y2)


plt.scatter(X1, Y1, color='blue')
plt.scatter(X2, Y2, color='blue')



plt.plot(X1, lin2.predict(poly.fit_transform(X1)), color='red')
plt.plot(X2, lin2.predict(poly.fit_transform(X2)), color='yellow')

plt.title('Polynomial Regression')
plt.xlabel('Flash_episode_number,Flash_us_viewers')
plt.ylabel('GoT_episode_number,GoT_us_viewers')

plt.show()





# Predicting a new result with Linear Regression
print( "LinearRegresion: ", lin.predict([[10]])  )

# Predicting a new result with Polynomial Regression
print( "PolynomialRegresion: ",lin2.predict(poly.fit_transform([[10]]))   )
