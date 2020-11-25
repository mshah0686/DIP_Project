import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import cv2
import numpy as np
'''
This script will train and save a TensorFlow model for recognizing a digit
This will use the MNIST data set to train
'''

# Import the data set from tf
print("=> Loading data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("=> Preprocessing...")
# Reshaping the training array to work with TF Keras
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalization
x_train /= 255
x_test /= 255

#Create Validation set
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

index = 129
print(y_test[index])
cv2.imshow("Sample", x_test[index])

cv2.waitKey(0)


# To Load the model do this:
new_model = tf.keras.models.load_model('model_data/model_ver_0')
x_temp = x_test[index].copy()
x_temp = x_temp.reshape(1, 28, 28, 1)
cv2.imshow("Reshaped", x_temp[0])
cv2.waitKey(0)
p = new_model.predict(x_temp)
max_index = np.argmax(p)
max_val = np.max(p)
if(max_val * 100 > 50):
    print("Predicted: ", max_index)