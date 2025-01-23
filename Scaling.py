import numpy as np

print('\n   Given a vector of feature values Р=(1,0,5,2,2). Normalize this vector using the formula that uses the minimum and '
      'maximum values of the feature Р. The value of the first coordinate of the normalized vector will be equal to')

print('\n   mathematical way')
P = [1, 0, 5, 2, 2]
print(P)
print('Min-max normalization, the value of the first coordinate = ', [(x-min(P))/(max(P) - min(P)) for x in P][0])

print('\n   sklearn way')
from sklearn.preprocessing import MinMaxScaler
P = np.array([1, 0, 5, 2, 2]).reshape(-1,1)
print(P)
scaler = MinMaxScaler()
print('Min-max normalization, the value of the first coordinate = ', scaler.fit_transform(P)[0])


print('\n   Given a vector of feature values Р=(1,0,5,2,2). Normalize this vector using a formula that uses the mean and'
      'standard deviation of the feature Р. The value of the last coordinate of the normalized vector will be equal to')

print('\n   numpy way')
P = np.array([1, 0, 5, 2, 2])
print(P)
print('The value of the last coordinate of the normalized vector = ', (((P - P.mean())/P.std()).round(2))[-1])

print('\n   sklearn way')
from sklearn.preprocessing import StandardScaler
P = np.array([1, 0, 5, 2, 2])
print(P)
P = np.array([1, 0, 5, 2, 2]).reshape(-1,1)
print(P)
std_metric = StandardScaler()
print('The value of the last coordinate of the normalized vector = ', std_metric.fit_transform(P)[-1])
print('Standartization', std_metric.fit_transform(P), sep='\n')