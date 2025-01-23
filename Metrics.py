import math
import numpy as np


print('\n Two vectors of feature values are given Р1=(0,1,2) и Р2=(2,1,0). Euclidean distans, Manhattan distance and '
      'max metric distans between them are equals:')

print('\n function way')
euclidean_metric = lambda x, y: ((x - y)**2).sum()**0.5
manhattan_metric = lambda x, y: abs(x - y).sum()
max_metric = lambda x, y: abs(x - y).max()
P1 = np.array([0, 1, 2])
P2 = np.array([2, 1, 0])
print(P1, P2, sep='\n')
print('Euclidean distances =', round(euclidean_metric(P1, P2), 2))
print('Euclidean distances math =', round(math.dist(P1, P2), 2))
print('Manhattan distances =', manhattan_metric(P1, P2))
print('Max-metrics =', max_metric(P1, P2))

print('\n   numpy way')
P1 = np.array([0,1,2])
P2 = np.array([2,1,0])
print(P1, P2, sep='\n')
print('Euclidean distances (L2 norm, ord=2 by default) =', np.linalg.norm(P1-P2).round(2))
print('Manhattan distances (L1 norm, ord=1) =', np.linalg.norm(P1-P2, ord=1))
print('Max-metrics =', max(np.abs(P1-P2)))

print('\n   sklearn way')
from sklearn.metrics.pairwise import euclidean_distances    # Returns the distances between the row vectors of X and the row vectors of Y
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics import max_error
P1 = [0,1,2]
P2 = [2,1,0]
print(P1)
print(P2)
print('Euclidean distances =', round(euclidean_distances([P1, P2])[1][0], 2))
print('Manhattan distances =', manhattan_distances([P1, P2])[1][0])
print('Max-metrics =', max_error(P1, P2))

#from sklearn.metrics import DistanceMetric
#P1 = np.array([0,1,2]).reshape(-1, 1)
#P2 = np.array([2,1,0]).reshape(-1, 1)
#print('Euclidean distances =', DistanceMetric.get_metric('euclidean').pairwise(P1, P2))
#print('Manhattan distances =', DistanceMetric.get_metric('manhattan').pairwise(P1, P2))
#print('Max-metrics =', DistanceMetric.get_metric('chebyshev').pairwise(P1, P2))

print('\n   The products are given: 1 - in a stock, 0 - out of stock. Evaluate the degree of similarity of products '
      'by calculating the Euclidian metric. The most similar to product A is C and the distance between these products is:')

print('\n   numpy way')
A = np.array([1,0,1,0,1,0])
B = np.array([0,1,1,1,0,0])
C = np.array([1,1,0,1,1,0])
D = np.array([1,1,0,1,1,1])
distance = []
distance.append(round(np.linalg.norm(A-C), 2))
print("Euclidean distance = ", *distance)

print('\n   sklearn way')
ACdistance = euclidean_metric(A.reshape(1, -1), C.reshape(1, -1))
print("Euclidean distance = ", round(ACdistance, 2))
