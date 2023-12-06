# import our needed libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tf2onnx
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from keras_flops import get_flops

from optimizer import *
import sys 
sys.path.insert(0,"/home/ehasn/esp-dl/tools/quantization_tool/linux/")
from calibrator import *
from evaluator import *


# seperate program to two part including 'training', 'deployment'
train_flag = input('Do you want to train model?(y/n) : ')
# train_flag = 'n'


# # Loading our dataset


# # get the working directory path
# current_dir = os.getcwd()
# print('my directory : ', current_dir)

# model / data parameters
batch_size = 256
img_height = 60
img_width = 60
input_shape = (img_height, img_width, 3)
num_classes = 2


train_dir = '../archive/training'
valid_dir = '../archive/validation'


train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  valid_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)


# 2) Visualizing our data
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
# plt.show()

print('------------------------------')
print('has been visualized succesfully')
print('------------------------------')


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 3) Build our model

# save coefficient output file in a log file
old_stdout = sys.stdout
log_file = open("../logs/parameters.log","w")
sys.stdout = log_file

model = keras.Sequential()
model.add(keras.Input(shape=input_shape))
model.add(layers.Conv2D(8, kernel_size=(3, 3), activation="relu"))
model.add(layers.Conv2D(8, kernel_size=(3, 3), activation="relu"))
model.add(layers.Conv2D(8, kernel_size=(3, 3), activation="relu"))
model.add(layers.Conv2D(8, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))


model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation="softmax"))
model.summary()

# calculate flops of our model
flops = get_flops(model, batch_size=batch_size)
print(f"FLOPS : {flops / 10 ** 9:.03} G")

# save coefficient output file in a log file
sys.stdout = old_stdout
log_file.close()

print('------------------------------')
print('has been builded succesfully')
print('------------------------------')


# 4) Training the model

# model configuration
batch_size = 64
epochs = 5

# model compile
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'],
  experimental_run_tf_function=False
)


if train_flag == 'y': 

    # training process
    history = model.fit(train_ds,
                        # validation_data=val_ds,
                        epochs=epochs,
                        batch_size=batch_size
                       )
    print('------------------------------')
    print('has been trained succesfully')
    print('------------------------------')
    
    
    
    # 5) Evaluate the trained model
    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f'Test loss: {loss:.4f}')
    print(f'Test accuracy: {accuracy:.4f}')
    print('------------------------------')
    print('has been evaluated succesfully')
    print('------------------------------')
    
    
    # 6) Test our model
    
    # calculating prediction for a batch of data
    batch = next(iter(val_ds.take(1)))
    images, labels = batch[0].numpy().astype("uint8"), batch[1].numpy()
    pred = model.predict(images)
    pred = np.argmax(pred, axis=1)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i]])
        ax.set_title("{}({})".format(class_names[labels[i]], class_names[int(pred[i])]),
                 color='green' if labels[i]==pred[i] else 'red')
        plt.axis("off")
    plt.show()
    
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
# cmd = 'zip -r /content/tmp_model.zip /content/tmp_model'
# os.system(cmd)                   #  (3) -- we can run this command to -- / 
print('------------------------------')
print('has been converted succesfully')
print('------------------------------')


# 9) Calibration

# load the ONNX model 
onnx_model = onnx.load("Mnist_model.onnx")

# optimize the ONNX model 
optimized_model_path = optimize_fp_model("Mnist_model.onnx")

batch = next(iter(val_ds.take(1)))
images, labels = batch[0].numpy().astype("uint8"), batch[1].numpy()

test_images = images.copy()
test_labels = labels.copy()

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

# print('======================')
# print(type(model_proto))
# print('model proto : ', model_proto)
# print('======================')

print('---------------------------------')
print('has been calibrated successfuly')
print('---------------------------------')


# 10) Evaluation 



# save prediction output file in a log file
old_stdout = sys.stdout
log_file = open("../logs/prediction.log","w")
sys.stdout = log_file

batch = next(iter(val_ds.take(1)))
images, labels = batch[0].numpy().astype("uint8"), batch[1].numpy()
pred = model.predict(images)
for i in range(10):
    print('pred(', i, ') : ', pred[i])
    pred_arg = np.argmax(pred[i])
    print('arg : ', pred_arg)
    print('y : ', labels[i])
    print('--------------------')
print('------------------------------')
print('has been predicted succesfully')
print('------------------------------')

# save prediction output file in a log file
sys.stdout = old_stdout
log_file.close()



# 12) insert input image to app_main.cpp file 
line_number = 12

# Read the existing content of the file
with open('../main/image.hpp', 'r') as file:
    lines = file.readlines()

my_in = images[8]
my_input = my_in.reshape(-1)

# Convert each element to a string and join them with commas
output_string = ', '.join(map(str, my_input))

# Modify the desired line with the array string
lines[line_number - 1] = output_string + '\n'


with open('../main/image.hpp', 'w') as file:
    
    # Write the output string to the file
    file.writelines(lines)

plt.imshow(my_in)
plt.show()





print('\n==============================================End of this part')


