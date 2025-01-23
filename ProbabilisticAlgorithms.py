print("\n   Let's throw a dice. Determine the probabilities of the event A = {an odd number came up}, B = {a number "
      "strictly greater than 4 came up}, C = {a number not less than 2 came up}. Conditional probabilities Pr(A|B), "
      "Pr(B|A), Pr(A|C), Pr(C|A).")

import numpy as np

def prob(x, conds):
    for conds in reversed(conds):
        N = len(x)
        x = conds(x)
        print('condition', x)
    return len(x) / N

x = np.arange(1, 7)
a = lambda x: x[x % 2 > 0]
b = lambda x: x[x > 4]
c = lambda x: x[x >= 2]

for conds in ((a,), (b,), (c,), (a, b), (b, a), (a, c), (c, a)):
    print(round(prob(x, conds), 2))

print('\n   The sample consists of 100 objects, 70 of which belong to class 1, other belong to class 0. It is known '
      'that the non-target binary feature P is equal to 1 for 10% of objects of class 1 and for 50% of objects of class 0.')
print('\n   solving:'
      'Of the 70 objects of class "1", only 10% have the feature P=1, and this is 7 objects (Pr(1) = 0.7).'
      'Of the 30 objects of class "0", only 50% have the feature P=1, and this is 15 objects.'
      'In total we have 7+15=22 (Pr(P=1) = 0.22) objects with the required feature, and only 7 of them are from class "1".')
pr_1 = 70/100       # probability that a feature belong to class 1
pr_0 = 30/100       # probability that a feature belong to class 0
pr_p_1 = 0.1        # probability that a feature is 1 if class is 1
pr_p_0 = 0.5        # probability that a feature is 1 if class is 0
pr_p = pr_1*pr_p_1 + pr_0*pr_p_0 # probability that a feature is 1
pr_1_p1 = (pr_p_1*pr_1)/pr_p

print('probability that a new feature belong to class 1 = ',round(pr_1_p1, 2))

print('\n   Given a tabledata X = [[1, 1], [1, 0], [0, 0], [1, 1]] and y = [0, 1, 0, 1]. Based on this sample, construct '
      'a naive Bayes classifier, which predict for a new object [0, 0] a probability of belonging to class 1')
print('\n   sklearn way')
import numpy as np
from sklearn.naive_bayes import GaussianNB
X = np.array([[1, 1], [1, 0], [0, 0], [1, 1]])
y = np.array([0, 1, 0, 1])
x_test = np.array([0, 0])
clf = GaussianNB()
clf.fit(X, y)
print(clf.predict([[0, 0]]))

print('\n   For objects from the table, the probabilities of belonging to the class 1 were obtained using some algorithm'
      '[0.6,0.81,0.5,0.9,0.7,0.75]. The true class labels are also known [0,1,0,1,0,1]. Define the AUC')
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
y_preds = [0,1,0,1,0,1]
y_score = [0.6,0.81,0.5,0.9,0.7,0.75]
print('AUC =', roc_auc_score(y_preds, y_score))
print("\n   Displaying auc on the Roc curve")
display = RocCurveDisplay.from_predictions(y_preds, y_score)
display.plot()
plt.show()


print('\n   For objects from the table, the probabilities of belonging to the class 1 were obtained using some algorithm'
      '[0.6,0.81,0.5,0.9,0.7,0.75]. The true class labels are also known [1,1,0,1,0,0]. Define the AUC')
from sklearn.metrics import roc_auc_score
y_preds = [1,1,0,1,0,0]
y_score = [0.6,0.81,0.5,0.9,0.7,0.75]
print('AUC =', roc_auc_score(y_preds, y_score))
print("\n   Displaying auc on the Roc curve")
display = RocCurveDisplay.from_predictions(y_preds, y_score)
display.plot()
plt.show()