import numpy as np

print("\n   The true value of the target feature Y and the predicted feature Y' are "
      "y = [1,2,3,4,5,-1,-2,-3,-4,-5] and y_pred = [0,2,2,5,3,-1,-1,-4,-6,-5]. Values MAE and MAPE are equal")

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
y = np.array([1,2,3,4,5,-1,-2,-3,-4,-5])
y_pred = np.array([0,2,2,5,3,-1,-1,-4,-6,-5])
print('MAE =', mean_absolute_error(y, y_pred))
print('MAPE =', int(mean_absolute_percentage_error(y, y_pred)*100))

print('\n Build a linear regression model (x = [0, 1, 2, 3], y = [0, 1, 0, 3]) to predict feature Y')
from sklearn.linear_model import LinearRegression
X = np.array([0, 1, 2, 3]).reshape(-1, 1)
y = np.array([0, 1, 0, 3]).reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)
print('coefficient of regression =', model.coef_)
print('interception', model.intercept_)

print('\n   What is the partial derivative of the function f(x,y)=2x2-3xy+4y2-2x+y+10 with respect to the variable x?')
from sympy import *
x,y = symbols('x y')
f = 2*x*x - 3*x*y + 4*y*y - 2*x + y + 10
print(diff(f, x))

print('\n   There is a strong (linear) dependence between features X1, X2 [[0,3], [1,2], [2,1], [3,0]]. Construct '
      'a linear regression model with regularization to predict feature Y (set the value of the regularization constant C equal to 1).')
from sklearn.linear_model import Ridge
alpha = 1
X = np.array([[0,3], [1,2], [2,1], [3,0]])
y = np.array([0, 1, 0, 3]).reshape(-1, 1)
      #expand X by 1.0 column
X = np.insert(X, 0, values=1, axis=1)
model = Ridge(alpha=alpha, fit_intercept=False)
model.fit(X, y)
print('coefficients =', model.coef_)

a = np.array([[6, 6, 5], [4, 15, 6], [15, 4, 6]])
b = np.array([4, 10, 2])
x = np.linalg.solve(a, b)
print(x)

X = np.array([[0, 3, 1], [1, 2, 1], [2, 1, 1], [3, 0, 1]])
y = np.array([0, 1, 0, 3]).reshape(-1, 1)
model = Ridge(alpha=1., fit_intercept=False)
model.fit(X, y)
print(model.coef_)