
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
X = np.array([[-1, 0],[1, 2],[2, -2],[-3, -1], [3, 2]])
y = np.array([1, 0, 0, 1, 1])
k = [3,5]
for i in k:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X, y)
    print(f'if k={i} the prediction class is', model.predict([[0, 0]]))
