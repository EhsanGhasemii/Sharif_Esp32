# Converting Deep Learning Models to Esp32 Format via Esp-dl

This repository contains the code and documentation for a project that focuses on converting deep learning models to Esp32 format using Esp-dl. The project aims to deploy a classifier model on the Esp32 platform.

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



## Getting Started

To get started with this project, follow the instructions in each folder's README file. The README files in each folder provide detailed explanations and instructions specific to that step.



