import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
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

print("=> Dataset data....")
print('    x_train shape:', x_train.shape)
print('    Number of images in x_train', x_train.shape[0])
print('    Number of images in x_test', x_test.shape[0])

# Building the model
print('=> Building Model...')
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.summary()

print('=> Training model...')

model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
# Training model
# TODO: Use validation set
model.fit(x=x_train,y=y_train, epochs=1)

print('=> Evaluate model...')
model.evaluate(x_test, y_test)

print('=> Saving model...')
model.save('model_data/model_ver_0')

# To Load the model do this:
# new_model = tf.keras.models.load_model('model_data/model_ver_0')
# new_model.summary()
