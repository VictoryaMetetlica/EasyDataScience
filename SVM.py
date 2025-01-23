import numpy as np
from numpy.linalg import inv

print('\n   Given the feature X = [[1, 0], [0, 1], [2, 0], [1, 1], [0, 2]] and y = [-1, -1, 1, 1, 1].'
      'By SVM: if a*X1+b*X2+c>0, then the object related to the class 1, else -1. Define a, b, c')
print('\n   the numpy way')
X = np.array([[1, 0, 1], [0, 1, 1], [2, 0, 1], [1, 1, 1], [0, 2, 1]])
y = np.array([-1, -1, 1, 1, 1])

w = inv(X.T @ X) @ X.T @ y
print('a, b, c:', w / w[0])

print('\n   sklearn way')
from sklearn.linear_model import LinearRegression
x = np.array([[1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]).reshape(-1, 2)
y = np.array([-1, -1, 1, 1, 1])
model = LinearRegression(fit_intercept=True)
model.fit(x, y)
print('a = ', round(model.coef_[0]))
print('b = ', round(model.coef_[1]))
print('c = ', round(model.intercept_))

print('\n   Given the function f(u,v)=u2-2v2-2u+4v+1. Its gradient at the point (0,0) is (?, ?). Find the minimum point '
      'using the gradient descent method. We assume that the step length is 0.5. What point will we get to from '
      'the point (0,0) after one iteration of the gradient descent method?')
print("\n   solving: "
      "1 - take partial derivatives f'u = 2u-2 and f'v = -4v + 4."
      "2 - the gradient grad = (f'u, f'v) at the point (0, 0) is "
      "grad(0, 0) = (2u-2, -4v + 4) = (2*0-2, -4*0+4) = (-2, 4)."
      "3 -  first iteration of SVM:"
      "the point (0,0) of gradient + h * (coordinates at the point (0,0) of gradient)"
      "but we need to find minimum (SVM finding maximum), that's whi we multiply (-1) coordinates - changing "
      "the vector direction:"
      "(0,0) + h * (-1) * (coordinates at the point (0,0) of gradient)")

from sympy import *
u, v = symbols('u v')
f = u ** 2 - 2 * v ** 2 - 2 * u + 4 * v + 1
du = diff(f, u)
dv = diff(f, v)
print("f'u = ", du)
print("f'v = ", dv)
    # the point (0,0) of gradient
a0 = np.array([0, 0])
grad = np.array([du, dv])
print("the coordinates of gradient at the point (0, 0) =", grad)
h = 0.5
a1 = a0 + h * (-1) * grad
print(a1)


print('\n   What is the gradient of the function f(u,v)=u*v+1 at point (1,2)')
from sympy import symbols, diff

u, v = symbols('u v')
L = u * v + 1
coord = {v: 2, u: 1}
f1 = diff(L, u).subs(coord)
f2 = diff(L, v).subs(coord)
print('(' + str(f1) + ', ' + str(f2) + ')')

print("\n   Given the feature X =[[-1,3], [3,-1], [1,-1], [-1,1]] and y=[1, 1, -1, -1]. By SVM: if a*X1+b*X2+c>0, then "
      "the object related to the class 1, else -1. Define a, b, c")
x = np.array([[-1,3], [3,-1], [1,-1], [-1,1]]).reshape((-1, 2))
print(x)
y = np.array([1, 1, -1, -1])
model = LinearRegression(fit_intercept=True)
model.fit(x, y)
print('a = ', round(model.coef_[0]))
print('b = ', round(model.coef_[1]))
print('c = ', round(model.intercept_))

print("\n   Given a table: x = [-2, -1, 0, 1, 2], y = [-1, -1, 1, 1, 1].Construct a linear classifier in which the "
      "expression [Mi<0] is majorized by the function (1-M)^2 (^2 is squaring). Obtain a classification rule: if X> ? ,"
      "then the object is from class 1, otherwise the object is from class -1 (round the answer to two decimal places).")
print("solving:"
      "1) the indents M of the objects: Mi=Yi*(w0+w1*Xi), where Yi,Xi are particular values"
      "2) the majorizing function (1-Mi)^2"
      "3) make a function L = the sum of the expressions of the majorizing function"
      "4) find the partial derivatives, and from them the weights w0, w1 "
      "5) solve expression x > w0/w1")
print('\n   sympy way')
w1, w0 = symbols('w1 w0')
y= simplify((1 - 2*w1 + w0)**2
            + (1 - w1 + w0)**2
            + (1-w0)**2
            + (1 - w1 - w0)**2
            + (1 - 2*w1 - w0)**2)
print('y = ', y)
print(solve((diff(y, w0), diff(y, w1)), (w0, w1)))
print('\n   sklearn way')
x = np.array([-2, -1, 0, 1, 2]).reshape((-1, 1))
y = np.array([-1, -1, 1, 1, 1])
model = LinearRegression(fit_intercept=True)
model.fit(x, y)
print('w0', model.intercept_)
print('w1', model.coef_)
