import numpy as np
from network import RBFN
from mnist_loader import load_data_wrapper
import matplotlib.pyplot as plt
import sys
import time

#live printing on terminal
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)
sys.stdout = Unbuffered(sys.stdout)

(train_x, train_y), (test_x, test_y), test_x_original = load_data_wrapper()
hidden_shape = 10000 
print("Data dimensionality: " + str(train_x.shape))
print("Hidden layer size: " + str(hidden_shape))
model = RBFN(hidden_shape)
start_time = time.time()
model.fit(train_x, train_y)
print("Elapsed time for training: " + str(time.time() - start_time))
y_pred = model.predict(train_x)
comp = np.array((y_pred == train_y), dtype=np.float)
print('Accuracy on train: ' + str(np.mean(comp)))
y_pred = model.predict(test_x)
comp = np.array((y_pred == test_y), dtype=np.float)
print('Accuracy on test: ' + str(np.mean(comp)))
i=0
j=0
k=0
false_classification_indices = []
correct_classification_indices = []
while i<3 or j<3:
    if comp[k]<1:
        false_classification_index = k
        false_classification_indices.append(false_classification_index)
        i += 1
    else:
        correct_classification_index = j
        correct_classification_indices.append(correct_classification_index)
        j += 1
    k += 1
plt.show()
correct = [np.reshape(test_x_original[j], (28,28)) for j in correct_classification_indices]
false = [np.reshape(test_x_original[j], (28,28)) for j in false_classification_indices]
for i in range(3):
    print("Correct example:")
    plt.imshow(correct[i], cmap='gray')
    plt.show(block=True)
for i in range(3):
    print("False example:")
    plt.imshow(false[i], cmap='gray')
    plt.show(block=True)
