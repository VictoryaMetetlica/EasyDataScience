import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

print("\n   The table shows the accuracy of the model built using non-target features U, V, W, X (if the table shows 0,"
      "then the corresponding feature was not used in building the model). Using a greedy feature selection algorithm "
      "(which starts with an empty set), find all features, that fall into the optimal set.")
y = np.array([0, 0.45, 0.4, 0.76, 0.5, 0.65, 0.7, 0.75, 0.3, 0.65, 0.7, 0.76, 0.6, 0.8, 0.71, 0.7])
features = 'XWVU'

cols = list(range(len(features)))
print('cols =', cols)
best_features = ""
score, row = 0, 0
while cols:
    rows = np.array([row | (1 << i) for i in cols])
    print('row =', rows)
    scores = y[rows]
    print('scores =', scores)
    index = scores.argmax()
    print('index =', index)
    if scores[index] < score:
        print(scores[index], score)
        break
    score, row = scores[index], rows[index]
    best_features += features[cols[index]]
    del cols[index]
print(best_features)

print('\n   sklearn way')

data = pd.DataFrame({'U':[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],'V':[0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                     'W':[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],'X':[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
                     'Y':[0,0.45,0.4,0.76,0.5,0.65,0.7,0.75,0.3,0.65,0.7,0.76,0.6,0.8,0.71,0.7]
                  })
X = data.iloc[:, :-1]
y = data.iloc[:, -1:]
lr = LinearRegression()
sfs = SequentialFeatureSelector(lr, n_features_to_select=3)
sfs.fit(X, y)
print(sfs.get_feature_names_out())
rfe_selector = RFE(lr, n_features_to_select=3)
rfe_selector.fit(X, y)
print('rfe_selector:', rfe_selector.get_feature_names_out())

print("\n   The table shows the accuracy of the model built using non-target features U, V, W, X (if the table shows 0,"
      "then the corresponding feature was not used in building the model). Using a greedy feature selection algorithm "
      "(which starts with a full set), find all features, that fall into the optimal set.")
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

data = pd.DataFrame({'U':[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],'V':[0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                     'W':[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],'X':[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
                     'Y':[0,0.45,0.4,0.76,0.5,0.65,0.7,0.75,0.3,0.65,0.7,0.76,0.6,0.8,0.71,0.7]
                  })
X = data.iloc[:, :-1]
y = data.iloc[:, -1:]
lr = LinearRegression()
sfs = SequentialFeatureSelector(lr, direction='backward')
sfs.fit(X, y)
print(sfs.get_feature_names_out())
rfe_selector = RFE(lr, n_features_to_select=2)
rfe_selector.fit(X, y)
print(rfe_selector.get_feature_names_out())
