# import our needed libraries
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tf2onnx
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from keras_flops import get_flops

from optimizer import *
import sys 
# sys.path.insert(0,"/home/ehasn/esp-dl/tools/quantization_tool/linux/")
from calibrator import *
from evaluator import *

import tensorflow.keras.backend as K;
import random
from tensorflow.keras.models import Model;




# define our classes ============================================================

#acdnet in functional way for safe serialization for embedded devices
class ACDNet:
    def __init__(self, input_length=66650, n_class=50, sr=44100, ch_conf=None):

        self.input_length = input_length;
        self.ch_config = ch_conf;

        stride1 = 2;
        stride2 = 2;
        channels = 4;
        k_size = (1, 3);
        n_frames = (sr/1000)*10; #No of frames per 10ms

        self.sfeb_pool_size = int(n_frames/(stride1*stride2));
        self.tfeb_pool_size = (1,2);
        if self.ch_config is None:
            self.ch_config = [channels, channels*4, channels*4, channels*4, channels*4, channels*4, channels*4, channels*4, channels*4, channels*4, channels*2, n_class];
        self.avg_pool_kernel_size = (1,4) if self.ch_config[1] < 64 else (2,4);

        self.conv1 = ConvBlock(self.ch_config[0], (1, 9), (1, stride1));
        self.conv2 = ConvBlock(self.ch_config[1], (1, 5), (1, stride2));
        self.conv3 = ConvBlock(self.ch_config[2], k_size, padding='same');
        self.conv4 = ConvBlock(self.ch_config[3], k_size, padding='same');
        self.conv5 = ConvBlock(self.ch_config[4], k_size, padding='same');
        self.conv6 = ConvBlock(self.ch_config[5], k_size, padding='same');
        self.conv7 = ConvBlock(self.ch_config[6], k_size, padding='same');
        self.conv8 = ConvBlock(self.ch_config[7], k_size, padding='same');
        self.conv9 = ConvBlock(self.ch_config[8], k_size, padding='same');
        self.conv10 = ConvBlock(self.ch_config[9], k_size, padding='same');
        self.conv11 = ConvBlock(self.ch_config[10], k_size, padding='same');
        self.conv12 = ConvBlock(self.ch_config[11], (1, 1));

        self.fcn = layers.Dense(n_class, kernel_initializer=keras.initializers.he_normal());

    def createModel(self):
        #batch, rows, columns, channels
        input = layers.Input(shape=(1, self.input_length, 1));

        #Start: SFEB
        sfeb = self.conv1(input);
        sfeb = self.conv2(sfeb);
        sfeb = layers.MaxPooling2D(pool_size=(1, self.sfeb_pool_size))(sfeb);
        # #End: SFEB

        # #swapaxes
        # tfeb = layers.Permute((3, 2, 1))(sfeb);

        # # exit();
        # #Start: HLFE
        tfeb = self.conv3(sfeb);
        tfeb = layers.MaxPooling2D(pool_size=self.tfeb_pool_size)(tfeb);

        tfeb = self.conv4(tfeb);
        tfeb = self.conv5(tfeb);
        tfeb = layers.MaxPooling2D(pool_size=self.tfeb_pool_size)(tfeb);

        tfeb = self.conv6(tfeb);
        tfeb = self.conv7(tfeb);
        tfeb = layers.MaxPooling2D(pool_size=self.tfeb_pool_size)(tfeb);

        tfeb = self.conv8(tfeb);
        tfeb = self.conv9(tfeb);
        tfeb = layers.MaxPooling2D(pool_size=self.tfeb_pool_size)(tfeb);

        # tfeb = self.conv10(tfeb);
        # tfeb = self.conv11(tfeb);
        # tfeb = layers.MaxPooling2D(pool_size=self.tfeb_pool_size)(tfeb);

        # tfeb =  layers.Dropout(rate=0.2)(tfeb);

        # tfeb = self.conv12(tfeb);
        # tfeb = layers.AveragePooling2D(pool_size=self.avg_pool_kernel_size)(tfeb);

        tfeb = layers.Flatten()(tfeb);
        tfeb = self.fcn(tfeb);
        #End: tfeb

        output = layers.Softmax()(tfeb);

        model = Model(inputs=input, outputs=output);
        # model.summary();
        return model;

class ConvBlock:
    def __init__(self, filters, kernel_size, stride=(1,1), padding='valid', use_bias=False):
        self.conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding, kernel_initializer=keras.initializers.he_normal(), use_bias=use_bias);

    def __call__(self, x):
        layer = self.conv(x);
        layer = layers.BatchNormalization()(layer);
        layer = layers.ReLU()(layer);
        return layer;

def GetAcdnetModel(input_length=66650, n_class=50, sr=44100, ch_config=None):
    acdnet = ACDNet(input_length, n_class, sr, ch_config);
    return acdnet.createModel();

# =======================================================================================================

# # create our model 
# input_size = 1024 * 64
# net = GetAcdnetModel(input_length=input_size);
# net.summary();

# # Generate random input data
# input_shape = (1, 1, input_size, 1)  # Replace with your actual dimensions
# random_input = tf.random.uniform(shape=input_shape, minval=0, maxval=1)

# # Get model output
# model_output = net(random_input)

# # print model output 
# print(model_output)
# ------------------------------------------------------------------------------------------------------


# seperate program to two part including 'training', 'deployment'
train_flag = input('Do you want to train model?(y/n) : ')
# train_flag = 'n'


# Loading our dataset

# get the working directory path
current_dir = os.getcwd()
print('my directory : ', current_dir)

# model / data parameters
num_classes = 10

# load the data and split it between train and test sets
# (X_train, y_train), (X_test1, y_test1) = keras.datasets.mnist.load_data(path=current_dir+'/mnist.npz')

n_f = 12000
X_train = np.random.randint(0, 255, size=(400, 1, n_f, 1))
X_test1 = np.random.randint(0, 255, size=(100, 1, n_f, 1))
y_train = np.random.randint(0, num_classes, size=(400))
y_test1 = np.random.randint(0, num_classes, size=(100))


# additional ____
X_train = X_train.reshape(X_train.shape[0], 1, -1)
X_test1 = X_test1.reshape(X_test1.shape[0], 1, -1)
# _______________

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
print('input shape : ', input_shape)
print('------------')

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
# fig = plt.figure(figsize=(25, 12))
# for i in range(8):
#     ax = fig.add_subplot(2, 4, i+1)
#     ax.set_title(y_train[i])
#     plt.imshow(X_train[i], cmap='gray')
# plt.show()
print('------------------------------')
print('has been visualized succesfully')
print('------------------------------')


# 3) Build our model

# model configuration
batch_size = 64
epochs = 1

# save parameters report
old_stdout = sys.stdout
log_file = open("../logs/parameters.log","w")
sys.stdout = log_file

# ============================================================================
# model = keras.Sequential()
# model.add(keras.Input(shape=input_shape))
# model.add(layers.Conv2D(8, kernel_size=(1, 9), activation="relu", strides=(1, 2)))
# # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Conv2D(32, kernel_size=(1, 5), activation="relu", strides=(1, 2)))
# # model.add(layers.MaxPooling2D(pool_size=(1, 2)))

# # model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
# # model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
# # model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
# # model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
# # model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
# # model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
# # model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
# # model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
# # model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))

# model.add(layers.Flatten())
# # model.add(layers.Dropout(0.5))
# model.add(layers.Dense(num_classes, activation="softmax"))
# model.summary()

input_size = input_shape[1]
n_class = num_classes
model = GetAcdnetModel(input_length=input_size, n_class=n_class);
model.summary();
# ===============================================================================

# calculate flops of our model
flops = get_flops(model, batch_size=batch_size)
print(f"FLOPS : {flops / 10 ** 9:.03} G")

# save parameters report
sys.stdout = old_stdout
log_file.close()
print('------------------------------')
print('has been builded succesfully')
print('------------------------------')


# 4) Training the model

# model compile
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"]
             )

if train_flag == 'y': 

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
    print('------------------------------')
    print('has been evaluated succesfully')
    print('------------------------------')
    
    
    # 6) Test our model
    test_num = 50
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


# 8) Convert the model
model = tf.keras.models.load_model("Mnist_model.h5")
tf.saved_model.save(model, "tmp_model")
cmd = 'python3.7 -m tf2onnx.convert --opset 13 --saved-model tmp_model --output "Mnist_model.onnx"'
os.system(cmd)
print('------------------------------')
print('has been converted succesfully')
print('------------------------------')


# 9) Calibration

# load the ONNX model 
onnx_model = onnx.load("Mnist_model.onnx")

# optimize the ONNX model 
optimized_model_path = optimize_fp_model("Mnist_model.onnx")

test_images = X_cal.copy()
test_labels = y_cal.copy()

calib_dataset = test_images[0:1800:20]
pickle_file_path = 'Mnist_calib.picle'


# calibration
model_proto = onnx.load(optimized_model_path)
print('Generating the quantization table:')

calib = Calibrator('int16', 'per-tensor', 'minmax')
# calib = Calibrator('int8', 'per-tensor', 'minmax')

calib.set_providers(['CPUExecutionProvider'])

# Obtain the quantization parameter
calib.generate_quantization_table(model_proto, calib_dataset, pickle_file_path)

# save coefficient output file in a log file
old_stdout = sys.stdout
log_file = open("../logs/coefficient.log","w")
sys.stdout = log_file

# Generate the coefficient files for esp32s3
coefficient_path = '../model/'
coefficient_name = 'Mnist_coefficient'
calib.export_coefficient_to_cpp(model_proto,
                                pickle_file_path,
                                target_chip='esp32',
                                output_path=coefficient_path,
                                file_name=coefficient_name,
                                print_model_info=True)

# save coefficient output file in a log file
sys.stdout = old_stdout
log_file.close()

print('---------------------------------')
print('has been calibrated successfuly')
print('---------------------------------')


# 10) Evaluation 

# save prediction output file in a log file
old_stdout = sys.stdout
log_file = open("../logs/prediction.log","w")
sys.stdout = log_file


# 11)prediction
print('prediction values : ')
cal_num = 30
pred = model.predict(X_cal[:cal_num])
for i in range(cal_num):
    x = X_cal[i] 
    print('pred(', i, ') : ', pred[i])
    pred_arg = np.argmax(pred[i])
    print('arg : ', pred_arg)
    print('y : ', np.argmax(y_cal[i]))
    print('--------------------')
print('------------------------------')
print('has been predicted succesfully')
print('------------------------------')

# additional __________________________________

# Create a tensor with the random integer
my_in = X_cal[8]
my_input = my_in.reshape(1, my_in.shape[0], my_in.shape[1], my_in.shape[2])


model_output = model.predict(my_input)
print('model output ------------------------\n', model_output)
my_input = my_input.reshape(-1)
# _____________________________________________

# save prediction output file in a log file
sys.stdout = old_stdout
log_file.close()


# 12) insert input image to app_main.cpp file 
line_number = 12

# Read the existing content of the file
with open('../main/image.hpp', 'r') as file:
    lines = file.readlines()

# my_in = X_cal[8]
print('my in shape: ', my_in.shape)
print('my_input shape: ', my_input.shape)

# Convert each element to a string and join them with commas
output_string = ', '.join(map(str, my_input))

# Modify the desired line with the array string
lines[line_number - 1] = output_string + '\n'


with open('../main/image.hpp', 'w') as file:
    
    # Write the output string to the file
    file.writelines(lines)


# plt.imshow(my_in, cmap="gray")
# plt.show()


print('\n==============================================End of this part')












