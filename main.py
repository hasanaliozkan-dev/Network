import nnfs 
import numpy as np
from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data
import pickle
import copy


nnfs.init()




    
## Data Loading

import os 
import cv2

def load_mnist_dataset(dataset,path):

    labels = os.listdir(os.path.join(path,dataset))
    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path,dataset,label)):
            image = cv2.imread(os.path.join(path,dataset,label,file),cv2.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)
    
    return np.array(X),np.array(y).astype('uint8')

def create_data_mnist(path):
    X,y = load_mnist_dataset('train',path)
    X_test,y_test = load_mnist_dataset('test',path)
    return X,y,X_test,y_test

X,y,X_test,y_test = create_data_mnist('fashion_mnist_images')

X = (X.astype(np.float32) - 127.5)/127.5
X_test = (X_test.astype(np.float32) - 127.5)/127.5

print(X.min(),X.max())

print(X.shape)

example = np.array([[1,2],[3,4]])
flattened = example.reshape(-1)

print(example)
print(example.shape)

print(flattened)
print(flattened.shape)

"""
print(labels)

import cv2
image_data = cv2.imread('fashion_mnist_images/train/7/0002.png', cv2.IMREAD_UNCHANGED)

print(image_data)

np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(image_data) 
plt.show()
"""

"""
X,y = sine_data()
dense1 = Layer.Dense(1, 64)

model = Model()
model.add(Layer.Dense(1,64))
model.add(ActivationFunctions().ReLU())
model.add(Layer.Dense(64,64))
model.add(ActivationFunctions().ReLU())
model.add(Layer.Dense(64,1))
model.add(ActivationFunctions().Linear())

model.set(loss= LossFunctions.MeanSquaredError(),optimizer=Optimizers.Adam(learning_rate=0.005,decay=1e-3),accuracy=Accuracy.Regression())
print(model.layers)

model.finalize()
model.train(X,y,epochs=10000,print_every=100)
"""
#dropout1= Layer.Dropout(0.1)
"""
dense2 = Layer.Dense(64,64)

dense3 = Layer.Dense(64,1)
"""

"""
relu1 = ActivationFunctions().ReLU()
relu2 = ActivationFunctions().ReLU()
linear = ActivationFunctions.Linear()

MSE= LossFunctions.MeanSquaredError()
optimizer = Optimizers.Adam(learning_rate=0.005,decay=1e-3)

accuracy_precision = np.std(y)/250

for epoch in range(10001):

    dense1.forward(X)
    relu1.forward(dense1.output)
    dense2.forward(relu1.output)
    relu2.forward(dense2.output)
    dense3.forward(relu2.output)
    linear.forward(dense3.output)
    data_loss = MSE.calculate(linear.output,y)

    regularization_loss = MSE.regularization_loss(dense1) + MSE.regularization_loss(dense2) + MSE.regularization_loss(dense3)
    loss = data_loss + regularization_loss

    preds = linear.output.copy()
    accuracy = np.mean(np.abs(preds - y) < accuracy_precision)

    if not epoch % 100:
        print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate:.3f}")



    MSE.backward(linear.output,y)

    linear.backward(MSE.dinputs)
    dense3.backward(linear.dinputs)
    relu2.backward(dense3.dinputs)
    dense2.backward(relu2.dinputs)
    relu1.backward(dense2.dinputs)
    
    dense1.backward(relu1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)

    optimizer.post_update_params()
 
X_test,y_test = sine_data()
y_test = y_test.reshape(-1,1)
dense1.forward(X_test)
relu1.forward(dense1.output)
dense2.forward(relu1.output)
relu2.forward(dense2.output)
dense3.forward(relu2.output)
linear.forward(dense3.output)


import matplotlib.pyplot as plt
plt.plot(X_test,y_test)
plt.plot(X_test,linear.output)
plt.show()"""