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



