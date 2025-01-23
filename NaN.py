import numpy as np

print("\n   Movie ratings are given. It's required to estimate what rating Sasha will give to the movie 'Harry Potter'. "
      "Use the Manhattan metrics without the standardization of the data")

print('\n   mathematical way')
P1 = np.array([5,5,5,3])
P2 = np.array([5,3,4,4])
P3 = np.array([2,5,3,5])
P4 = np.array([3,4,4,None])   # Саша
p1p4 = sum(abs(P1[:-1]-P4[:-1]))
p2p4 = sum(abs(P2[:-1]-P4[:-1]))
p3p4 = sum(abs(P3[:-1]-P4[:-1]))
print('p1p4, p2p4, p3p4 =', p1p4, p2p4, p3p4)
print("Sasha will give to the movie 'Harry Potter'", round(1/(1/p1p4+1/p2p4+1/p3p4)*(P1[3]/p1p4+P2[3]/p2p4+P3[3]/p3p4), 1))


from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

print('\n    sklearn way')
X = np.array([[5,5,5,3], [5,3,4,4], [2,5,3,5], [3,4,4,None]])

simple_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
print('\n   The imputing missing values defined by The SimpleImputer class', simple_imputer.fit_transform(X)[3, -1], sep='\n')

knn_imputer = KNNImputer(n_neighbors=2)
print('\n   The imputing missing values defined by The KNNImputer class', knn_imputer.fit_transform(X)[3, -1], sep='\n')

