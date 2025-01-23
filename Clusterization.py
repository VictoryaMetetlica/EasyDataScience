from Metrics import manhattan_metric

print('Distances in some metric are given. Present the data in the shape of graph distance and launch the algorithm of '
      'clusterization without a number of clusters. Input param R = 2.5. Which objects will be the same cluster with A?')

import numpy as np

a = [0, 3, 4, 4, 1]
b = [3, 0, 1 ,2, 5]
c = [4, 1, 0, 3, 3]
d = [3, 2, 3, 0, 4]
e = [1, 5, 3, 4, 0]
r = 2.5

def dist(a, b):
    return sum(np.abs(np.array(a) - np.array(b)))**0.5

for i in b, c, d, e:
    if dist(a, i) < r:
        print(i, round(dist(a, i), 2))

print('\n   Divide objects into 2 clusters. At the first iteration of the k-means algorithm, points (2,3) and (1,1) '
      'were selected. After the first iteration of the algorithm, objects will be assigned to the cluster determined by '
      'the first point (the Manhattan metric is used).')

manhattan = lambda x, y: np.linalg.norm(x - y, ord=1)
data = {'A' : np.array([4,  2]),
        'B' : np.array([3, 2]),
        'C' : np.array([1, -1]),
        'D' : np.array([-1, 1]),
        'E' : np.array([0, 4])}
center1 = np.array([2, 3])
center2 = np.array([1, 1])

for coordinate, vector in data.items():
        if dist(center1, vector) < dist(center2, vector):
            print(coordinate, 'in cluster of point 1')
        else:
            print(coordinate, 'in cluster of point 2')

print('\n   second way')
data = np.array([(4, 2),
                 (3, 2),
                 (1, -1),
                 (-1, 1),
                 (0, 4)])
indx = np.array(list('ABCDE'))
centers = np.array([(2, 3),
                    (1, 1)])
print('distances between center and each coordinate:')
manhattan_distances = np.apply_along_axis(lambda x: np.linalg.norm(x - centers, 1, 1), 1, data)
print(manhattan_distances)
print('cluster with first point (2,3):', *indx[np.argmin(manhattan_distances, 1) == 0])