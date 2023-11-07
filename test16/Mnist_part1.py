# import our needed libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tf2onnx
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from optimizer import *
import sys 
sys.path.insert(0,"/home/ehasn/esp-dl/tools/quantization_tool/linux/")
from calibrator import *
from evaluator import *


# Loading our dataset

# define random creating image classifier
def random_image_classifier(class_num=5,
                            data_num=1000,
                            rows=1,
                            columns=1,
                            test_percentage=0.3
                           ):
    
    # initialize parameters
    data_num = int(data_num / class_num)
    data_size = (data_num, rows, columns)
    
    for i in range(class_num):

        # initialize parameters 
        band_width = np.floor(255 / class_num)
        lowwer_band = i * band_width
        higher_band = (i + 1) * band_width

        # creating data
        data_X = np.random.randint(low=lowwer_band, high=higher_band, size=data_size)
        data_y = np.ones((data_num, 1)) * i 
        
        # concatenating datas
        if i == 0:             
            X_data = data_X.copy()
            y_data = data_y.copy()

        else:             
            X_data = np.concatenate((X_data, data_X), axis=0)
            y_data = np.concatenate((y_data, data_y), axis=0)
            
    # split data 
    X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                        y_data,
                                                        test_size=test_percentage,
                                                        random_state=42
                                                       )
        
    # return output
    return ((X_train, y_train), (X_test, y_test))

# get the working directory path
current_dir = os.getcwd()
print('my directory : ', current_dir)

# model / data parameters
num_classes = 2

# load the data and split it between train and test sets
(X_train, y_train), (X_test1, y_test1) = random_image_classifier(num_classes)

# test/train split
ts = 0.3 # percentage of images that we want to use for testing
X_test, X_cal, y_test, y_cal = train_test_split(X_test1, y_test1, test_size=ts, random_state=42)

# scale images to the [0, 1] range
# X_train = X_train.astype(np.float32) / 255
# X_test = X_test.astype(np.float32) / 255
# X_cal = X_cal.astype(np.float32) / 255

# make sure images have shape (28, 28, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
X_cal = np.expand_dims(X_cal, -1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_cal = keras.utils.to_categorical(y_cal, num_classes)

# define input shape
input_shape = X_train.shape[1:]

# check our variable's dimension
print('X_train shape : ', X_train.shape)
print('y_train shape : ', y_train.shape)
print('------------')
print('X_test shape : ', X_test.shape)
print('y_test shape : ', y_test.shape)
print('------------')
print('x_cal shape : ', X_cal.shape)
print('y_cal shape : ', y_cal.shape)
print('------------------------------')
print('has been loaded succesfully')
print('------------------------------')
print('max(X) : ', np.max(X_train[0]))
print('min(X) : ', np.min(X_train[0]))
print('mean(X) : ', np.mean(X_train[0]))


# 2) Visualizing our data

# show some pictures of dataset
fig = plt.figure(figsize=(25, 12))
for i in range(8):
    ax = fig.add_subplot(2, 4, i+1)
    ax.set_title(y_train[i])
    plt.imshow(X_train[i], cmap='gray')
plt.show()
print('------------------------------')
print('has been visualized succesfully')
print('------------------------------')

# 3) Build our model
model = keras.Sequential()
model.add(layers.Flatten(input_shape=input_shape))
model.add(layers.Dense(units=num_classes, activation='softmax'))
model.summary()
print('------------------------------')
print('has been builded succesfully')
print('------------------------------')


# 4) Training the model

# model configuration
batch_size = 64
epochs = 512

# model compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
             )

# training process
history = model.fit(X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    validation_data=(X_test, y_test)
                   )
print('------------------------------')
print('has been trained succesfully')
print('------------------------------')

# 5) Evaluate the trained model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# 6) Test our model
test_num = 20
X = X_cal[:test_num]
y = y_cal[:test_num]
y = np.argmax(y, axis=1)

# claculating prediction
pred = model.predict(X)
pred = np.argmax(pred, axis=1)
t = y == pred
print('performance on claibration dataset : ',
      np.sum(t),
      ' / ',
      test_num
     )
print('------------------------------')
print('has been tested succesfully')
print('------------------------------')

# 7) Saving the model
model.save('Mnist_model.h5')
print('------------------------------')
print('has been saved succesfully')
print('------------------------------')

# save our dataset
np.save('X_train', X_train)
np.save('y_train', y_train)

