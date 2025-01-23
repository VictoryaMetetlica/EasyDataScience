import numpy as np

print('\n   The committee consists of three classifiers. The final decision is made by majority vote. The classifiers have '
      'an accuracy of 0.8 0.9 1 respectively. Define the accuracy of ensemble algorithm')

print('Possibilities that the committee will be wrong:')
mistakes = [['True: accuracy = 0.8', 'False: mistake = 1-0.9', 'False: mistake = 1-1'],
            ['False: mistake = 1-0.8', 'True: accuracy = 0.9', 'False: mistake = 1-1'],
            ['False: mistake = 1-0.8', 'False: mistake = 1-0.9', 'True: accuracy = 1'],
            ['False: mistake = 1-0.8', 'False: mistake = 1-0.9', 'False: mistake = 1-1']]
m = np.array([[0.8, 1-0.9, 1-1],
              [1-0.8, 0.9, 1-1],
              [1-0.8, 1-0.9, 1],
              [1-0.8, 1-0.9, 1-1]])
print('multiplying over the axis = 1:', m.prod(axis=1))
print('the accuracy of ensemble algorithm =', 1-m.prod(axis=1).sum())

print("\n   A weighted voting scheme is used for classification. Let's assume that object A was defined by the first and"
      " third classifiers as an object from class 1, and the second classifier defined A as an object of class -1. "
      "The result of weighted voting, if the weights of the classifiers are 2, 4, 3 respectively, will classify object A as?")
print('Theoretical simple way')
w1, w2, w3 = 2, 4, 3
cl1, cl2, cl3 = 1, -1, 1
func = w1*cl1 + w2*cl2 + w3*cl3
if func > 0:
    print('The object A classified as', 1)
elif func < 0:
    print('The object A classified as', -1)
else:
    print('Something wrong')

print('\n   numpy way')
y = np.array([1, -1, 1])
w = np.array([2, 4, 3])

classes = np.unique(y)
print('Lots of classes:', classes)
voices = np.array([w[y == cls].sum() for cls in classes])
print('Counting votes for each class', voices)
print('Class with the highest number of votes:', classes[voices.argmax()])

print('\n   The committee consists of three classifiers. The final decision is made by majority vote. The classifiers have '
      'accuracy of 0.8 0.9 1 respectively. Define the accuracy of ensemble algorithm')
print("Possibilities that the committee will be wrong (identical classifiers (dependent) do not need to be multiplied):")
mistakes = [['True: accuracy = 0.8', 'False: mistake = 1-0.9'],
            ['False: mistake = 1-0.8', 'True: accuracy = 0.9'],
            ['False: mistake = 1-0.8', 'False: mistake = 1-0.9']]
m = np.array([[0.8, 1-0.9],
              [1-0.8, 1-0.9]])
print('multiplying over the axis = 1:', m.prod(axis=1))
print('the accuracy of ensemble algorithm =', 1-m.prod(axis=1).sum())
