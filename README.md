# Converting Deep Learning Models to Esp32 Format via Esp-dl

This repository contains the code and documentation for a project that focuses on converting deep learning models to Esp32 format using Esp-dl. The project aims to deploy a classifier model on the Esp32 platform.

## How to run this project? 
In this report, we will explain how to implement the conversion of artificial intelligence models written in
Python to the Esp32 using Esp-dl. The initial step of this process involves building your model in Python
using torch or TensorFlow-Keras. Then, you train the model on your desired dataset and finally transfer the
model to the Esp32. As you have already realised, familiarity with the following items is necessary for this
process.  
- Proficient in building and training neural networks
- Expertise in configuring the ESP-IDF release 4.4 environment
- Working knowledge of C and C++ programming languages

This can be highly applicable in many IoT (Internet of Things) and embedded systems projects. Being able to
perform artificial intelligence processing with the help of a small processor can be useful in various
applications, including smart factories, home automation, and more. The esp32, which is a low-cost WiFi
module with multiple features, allows us to perform a wide range of tasks at a relatively affordable price. 

I have prepared a Python code for this project that automatically generates all the necessary components to run our deep learning model on the ESP32. Since I first prepared this project on the Mnist dataset, all the names I used to create different files are saved with those names. In fact, I initially used this project specifically for running the Mnist dataset, and afterwards, when implementing different networks and datasets, I did not change the names and continued to use them. However, the professional approach in this matter is to determine the names according to the project you are working on. Therefore, the name of the Python file I have prepared is "Mnist.py". This code is located in the "src" directory. In order to correctly run this project for your deep learning model and dataset, You need to create the following subfolders so that the execution of the Python code I mentioned does not encounter an error.
- archive
- logs
- main
  - app_main.cpp
  - CMakeLists.txt
  - image.hpp
- model
  - Mnist_coefficient.cpp
  - Mnist_coefficient.hpp
  - model_define.hpp
- src
  - calibrator.pyd
  - calibrator.so
  - calibrator_acc.pyd
  - calibrator_acc.so
  - evaluator.pyd
  - evaluator.so
  - Mnist.py
  - optimizer.py
- CMakeLists.txt

Here, all you need to do is simply execute the "Mnist.py" code. By doing so, two files, "Mnist_coefficient.cpp" and "Mnist_coefficient.hpp," will be automatically generated. However, it is necessary for you to create the files "app_main.cpp," "image.hpp," and "model_define.hpp" exactly as I have written them.  

Now, let's move on to the main part of the project, which is executing the Python code that I mentioned earlier.

### 0. import our needed libraries


```python
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
from calibrator import *
from evaluator import *
```

### 1. Loading our dataset
In this part, you need to choose a dataset on which you want to implement your deep learning model and input it into your program. Here, I have worked with the Mnist dataset initially and then selected another dataset for gender classification. I have provided the links to both datasets below.
- The dataset used in this project is sourced from Kaggle [link](https://www.kaggle.com/datasets/vikramtiwari/mnist-numpy/ ).
- The dataset used in this project is sourced from Kaggle [link](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset/code).  


The first dataset is structured in a way that you download a file named "mnist.npz" and read it in a similar manner in‍‍ your program.
```python
# load the data and split it between train and test sets
(X_train, y_train), (X_test1, y_test1) = keras.datasets.mnist.load_data(path=current_dir+'/mnist.npz')

# test/train split
ts = 0.3 # percentage of images that we want to use for testing
X_test, X_cal, y_test, y_cal = train_test_split(X_test1, y_test1, test_size=ts, random_state=42)
```

And the second dataset is structured in a way that you need to download the data from the provided link and place them in the specified folders. Afterwards, you should put the downloaded data in the "archive" folder. So, you need to have the following directories, which contain your dataset, inside the "archive" folder.
- archive
  - training
    - female
    - male
  - validation
    - female
    - male
   
Finally, you should read the data using the following code.
```python
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
```
### 2. Visualizing our data
If you need to inspect your initial data and view some of them, you can execute the following code snippet. As you have noticed so far, running this section is not mandatory and is only used for initial data exploration.  

```python
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()
```

### 3. Build our model
In this part, you need to design your deep learning model. For example, I have used a neural network with 4 Conv_2d layers and a MaxPooling_2d layer, followed by a Softmax layer for classification. The only point to consider here is that you should use layers supported by esp-dl.  

```python
model = keras.Sequential()
model.add(keras.Input(shape=input_shape))
model.add(layers.Conv2D(16, kernel_size=(3, 3), activation="relu"))
model.add(layers.Conv2D(16, kernel_size=(3, 3), activation="relu"))
model.add(layers.Conv2D(16, kernel_size=(3, 3), activation="relu"))
model.add(layers.Conv2D(16, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation="softmax"))
model.summary()
```

### 4. Training the model
In this part, you first build the neural network that you have designed, and then train it on your dataset. Here, I have used a train_flag that indicates whether the neural network should start training or not, based on the command given by the user. That is, we receive this flag from the user at the beginning. The reason for this is that in many cases, we need to perform multiple tests on a trained model and do not want to start training from scratch again. Instead, we want to test the initial model results.  

```python
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
```

### 5. Evaluate the trained model (optional)
```python
loss, accuracy = model.evaluate(val_ds, verbose=0)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')
```

### 6. Test our model (optional)
```python
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
```









## Project Overview

The project consists of six folders, each representing a step in the process of converting and deploying deep learning models. Folder number 21 is the most complete version of the implementation of a deep learning model on Mnist data, which we transferred and implemented on the board. Therefore, in this section, I have put all the codes, except for the dataset, which you should download and put in the folder I said. But in folders number 15 to 20, I have placed only the Python code, which if you are familiar with the process, you can use it by placing it in a suitable directory. Here's a brief overview of each folder:

### test15
In this folder, a simple one-layer model is converted to Esp32 format. The model consists of a sigmoid neuron that multiplies the input by a proper value to create the output.

### test16
This folder builds upon the previous step by using a Softmax layer to classify two classes. The goal is to demonstrate the classification capabilities of Esp32 with a more complex model.

### test17
Here, an expansion model is created, which includes a conv2d layer. The purpose is to showcase the integration of convolutional layers in the Esp32 format for image processing tasks.

### test18
The test18 folder contains a complete model with two conv2d layers and two maxpool layers. It demonstrates the use of multiple convolutional and pooling layers for more advanced image processing tasks. The model is trained on a simple dataset consisting of images with a size of 1x1.

### test19
In this step, the model from test18 is repeated, but with a more complex dataset. The dataset includes images with a size of 15x15 and 10 different classes. The goal is to showcase the scalability and performance of Esp32 with larger and more diverse datasets.

### test20
The test20 folder concludes the project by using the popular Mnist dataset. The same model architecture from test19 is applied to classify handwritten digits. This step demonstrates the successful deployment of a classifier model on the Esp32 platform using Esp-dl.

### test21(End of Mnist dataset)
In this part, we implemented the same as the previous part, with the difference that we tried to make our model more complicated so that we could check the maximum parameters that we can transfer on the board. To be able to run this part correctly, you need to download the Mnist dataset and place it in the "src" folder. You can do this from the link below. The format of the file you put must be mnist.npz.  
The dataset used in this project is sourced from Kaggle [link](https://www.kaggle.com/datasets/vikramtiwari/mnist-numpy/ ).
 
In order to run this part of the project correctly, you must create a directory like the one below and run the Mnist.py file in the src folder.  
- main
    - app_main.cpp
    - CMakeLists.txt
- model
    - Mnist_coefficient.cpp
    - Mnist_coefficient.hpp
    - model_define
- src
- CMakeLists.txt


### test22
This episode was a failed attempt. I was planning to do this project with the help of torch. My reason was that I wanted to read the input images from my local memory this time, and this was done well with the help of torch. I could also use it to process my deep learning model on the GPU. Finally, we did this with the help of tensorflow in folder 23, but the processing is still done on the CPU.

### test23
In this part, I tried to do the gender classification project with the help of deep learning. I used the Gender classification dataset in the link below.  
The dataset used in this project is sourced from Kaggle [link](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset/code).  
In this section, you must create a directory similar to the following.
- archive
- logs
- main
  - app_main.cpp
  - CMakeLists.txt
  - image.hpp
- model
  - Mnist_coefficient.cpp
  - Mnist_coefficient.hpp
  - model_define.hpp
- src
  - calibrator.pyd
  - calibrator.so
  - calibrator_acc.pyd
  - calibrator_acc.so
  - evaluator.pyd
  - evaluator.so
  - Mnist.py
  - optimizer.py
- CMakeLists.txt


Due to the fact that I have used the same files related to the Mnist code in the continuation of my project, all the names in this part and probably the next parts have the same name. But it is professional behavior to update the names according to your current project. Therefore, the main code that you should run in this section is the Mnist.py code in the same src folder.  
By running this code, your model will be trained. Then the appropriate files are created in appropriate folders with the help of which you can move your model on the board. Therefore, in order not to encounter an error while running the code, it is necessary to create the folders according to what I said above. Finally, a sample image is saved in the image.hpp file, which by running your program with the help of esp-idf, this image is given to your program as an input, which finally predicts the model on it.  
In order to correctly transfer your model on the board, you need to transfer the input and output size that you defined in the Python code to the model_define code and app_main.cpp. Also, in order for your program to predict the output with the highest accuracy, you must enter the production coefficients in the file I mentioned correctly. All these items are stored in the logs folder, which you can remove all of them from this folder. Finally, I have prepared a report of the performance of several examples that I have implemented in a file called report_prediction.log, which shows the accuracy of the program execution on the board compared to its execution in Python code.

### Results
I have reached various results by performing these parts that I mentioned above, which I must mention here.  
1. The first important thing we wanted to achieve was how big a model we can transfer on the board. Therefore, by enlarging my model in the test21 folder, I tried to find the answer to this question. Finally I realized that we can port any model with less than 300k parameters. Therefore, the maximum amount of parameters that we can transfer is equal to 300k. Of course, this is if quantization is done with 16-byte data. If we use uint8_t, we can transfer up to 600k parameters. I used the following model and it got the following error which had more parameters than the maximum valid limit for parameter passing.  
![Image Description](/pics/test2.png)
![Image Description](/pics/test1.png)
2. The next item was to input images with a size larger than 64x64x3. In this case, we would encounter an error.
3. And finally, we should mention the accuracy of this method for transferring deep learning models to the Esp32 module. I have been able to show the accuracy of quantization by this method in the report_prediction.log file by writing a set of predictions made by Python and Esp, which is around 0.05%. ![Image Description](/pics/test3.png)


## Getting Started

To get started with this project, follow the instructions in each folder's README file. The README files in each folder provide detailed explanations and instructions specific to that step.



