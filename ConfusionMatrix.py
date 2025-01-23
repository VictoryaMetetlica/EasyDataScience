print('\n   The confusion matrix is given (tn, fn, fp, tp = 25, 20, 10, 15). Calculate an accuracy, a precision and '
      'a recall:')

print('\n   numpy way')
import numpy as np
g = np.array([[25, 20],
              [10, 15]])
accuracy = np.sum(np.diag(g))/np.sum(g)
print(np.around(accuracy, 2))
precision = np.diag(g)[-1]/np.sum(g[-1])
print(np.around(precision, 2))
recall = np.diag(g)[-1]/sum(g)[-1]
print(np.around(recall, 2))

print('\n   classical way')
tn, fn, fp, tp = 25, 20, 10, 15
A = (tn + tp)/(tn + tp + fn + fp)
P = tp/(tp + fp)
R = tp/(tp + fn)
print("%0.2f" % A, "%0.2f" % P, "%0.2f" % R)


print('\n One stupid clasificator define all object as a class 1. Supposedly, the set contain 50 objects.: '
      '20 of them actually belong to class 0, and 30 of them actually belong to class 1.')
print('\n   numpy way')
matrix = np.array([[0, 0],
                   [20, 30]])
accuracy = np.sum(np.diag(matrix))/np.sum(matrix)
print('accuracy=', accuracy)
precision = np.diag(matrix)[-1]/np.sum(matrix[-1])
print('precision=', precision)
recall = np.diag(matrix)[-1]/np.sum(matrix[:, -1])
print('recall=', recall)
