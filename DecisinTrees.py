print('\n   Given features p1 = [0, 0, 1, 1, 0], p2 = [1, 1, 1, 1, 1], p3 = [0, 0, 0, 1, 1], y = [1, 1, 1, 0, 0]. '
      'Define a impurity Gini for each of features:')

import numpy as np

X = np.array([[0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]])
y = np.array([1, 1, 1, 0, 0])

def gini_impurity(x, y):
    gini = lambda y: y.mean() * (1 - y.mean()) if y.size > 0 else 0
    return x.mean() * gini(y[x==1]) + (1 - x.mean()) * gini(y[x==0])

for x in X.T:
    print('for feature', x, 'Gini impurity:', gini_impurity(x, y).round(2))