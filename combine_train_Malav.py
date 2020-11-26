import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import PIL
from PIL import Image, ImageOps
import numpy as np
import cv2

'''
This script will train and save TF model
Trying to combine MNIST dataset with +,- symbol data
+ : tagged with 10 (index)
- : tagged with 11 (index)
'''

def gather_images_from_directory(dir):
  print("   => Gather from directories, this may take a while")
  ret_images = []
  ret_label = []
  subdirs = [x[0] for x in os.walk(dir)]
  for folder in subdirs[1:]:
    path = folder
    label = int(folder.split("/")[-1])
    for image in os.listdir(path):
      img = Image.open(path + "/" + image)
      img = img.convert('1')
      img = img.resize((28,28))
      arr = np.array(img)
      arr = np.invert(arr)
      arr = arr.astype('float32')
      ret_images.append(arr)
      ret_label.append(label)
  return ret_images, ret_label

# Import the data set from tf
print("=> Loading MNIST data...")
(x_train_M, y_train_M), (x_test_M, y_test_M) = tf.keras.datasets.mnist.load_data()
# Reshaping the training array to work with TF Keras
x_train_M = x_train_M.reshape(x_train_M.shape[0], 28, 28, 1)
x_test_M = x_test_M.reshape(x_test_M.shape[0], 28, 28, 1)
y_train_M = y_train_M.reshape(y_train_M.shape[0], 1)
y_test_M = y_test_M.reshape(y_test_M.shape[0], 1)
print("MNIST Data shape: ", x_train_M.shape)


print("=> Loading directory data")
x_train_sym, y_train_sym = gather_images_from_directory("training_data/")

x_train_sym = np.asarray(x_train_sym)
x_train_sym = x_train_sym.reshape(x_train_sym.shape[0], 28, 28, 1)
y_train_sym = np.asarray(y_train_sym)
y_train_sym = y_train_sym.reshape(y_train_sym.shape[0], 1)
print("Directory data shape: ", x_train_sym.shape)

print("=> Combining data sets")
x_train = np.concatenate((x_train_M, x_train_sym), axis = 0)
y_train = np.concatenate((y_train_M, y_train_sym), axis = 0)
print("Final data shape: ", x_train.shape)
# print("=> Preprocessing...")
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')

# Normalization
x_train /= 255
# x_test /= 255

# #Create Validation set
# x_val = x_train[-10000:]
# y_val = y_train[-10000:]
# x_train = x_train[:-10000]
# y_train = y_train[:-10000]

print("=> Dataset data....")
print('    x_train shape:', x_train.shape)
print('    Number of images in x_train', x_train.shape[0])

# Building the model
print('=> Building Model...')
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(12,activation=tf.nn.softmax))

model.summary()

print('=> Training model...')

model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
# Training model
# TODO: Use validation set
model.fit(x=x_train,y=y_train, batch_size=500, epochs=15, shuffle=True)

# print('=> Evaluate model...')
# model.evaluate(x_test, y_test)

print('=> Saving model...')
model.save('model_data/model_ver_1')


# # To Load the model do this:
# # new_model = tf.keras.models.load_model('model_data/model_ver_0')
# # new_model.summary()
