import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
'''
This script will train and save TF model
Trying to combine MNIST dataset with +,- symbol data
+ : tagged with 10 (index)
- : tagged with 11 (index)
'''

# # Import the data set from tf
# print("=> Loading MNIST data...")
# (x_train_M, y_train_M), (x_test_M, y_test_M) = tf.keras.datasets.mnist.load_data()
# print(y_train_M[0])


print("=> Loading directory data")
data_dir = "training_data/"
batch_size = 32
label_list = [10, 11]
train_data_sym = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(45, 45),
  batch_size=batch_size,
  label_mode = 'int')

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(45, 45),
  batch_size=batch_size,
  label_mode = 'int')

print(val_ds[0])

# print("=> Preprocessing...")
# # Reshaping the training array to work with TF Keras
# x_train = x_train_M.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test_M.reshape(x_test.shape[0], 28, 28, 1)
# input_shape = (28, 28, 1)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')

# # Normalization
# x_train /= 255
# x_test /= 255

# #Create Validation set
# x_val = x_train[-10000:]
# y_val = y_train[-10000:]
# x_train = x_train[:-10000]
# y_train = y_train[:-10000]

# print("=> Dataset data....")
# print('    x_train shape:', x_train.shape)
# print('    Number of images in x_train', x_train.shape[0])
# print('    Number of images in x_test', x_test.shape[0])
# print('    Number of images in x_val', x_val.shape[0])

# # Building the model
# print('=> Building Model...')
# model = Sequential()
# model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
# model.add(Dense(128, activation=tf.nn.relu))
# model.add(Dropout(0.2))
# model.add(Dense(10,activation=tf.nn.softmax))

# model.summary()

# print('=> Training model...')

# model.compile(optimizer='adam', 
#                 loss='sparse_categorical_crossentropy', 
#                 metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
# # Training model
# # TODO: Use validation set
# model.fit(x=x_train,y=y_train, batch_size=200, epochs=10, shuffle=True,validation_data=(x_val, y_val))

# print('=> Evaluate model...')
# model.evaluate(x_test, y_test)

# print('=> Saving model...')
# model.save('model_data/model_ver_1')


# # To Load the model do this:
# # new_model = tf.keras.models.load_model('model_data/model_ver_0')
# # new_model.summary()
