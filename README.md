# Converting Deep Learning Models to Esp32 Format via Esp-dl

This repository contains the code and documentation for a project that focuses on converting deep learning models to Esp32 format using Esp-dl. The project aims to deploy a classifier model on the Esp32 platform.

## Project Overview

The project consists of six folders, each representing a step in the process of converting and deploying deep learning models. Here's a brief overview of each folder:

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

### test20
In this part, we implemented the same as the previous part, with the difference that we tried to make our model more complicated so that we could check the maximum parameters that we can transfer on the board. To be able to run this part correctly, you need to download the Mnist dataset and place it in the "src" folder. You can do this from the link below. The format of the file you put must be mnist.npz.  https://www.kaggle.com/datasets/vikramtiwari/mnist-numpy/

## Getting Started

To get started with this project, follow the instructions in each folder's README file. The README files in each folder provide detailed explanations and instructions specific to that step.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to modify and use the code according to your needs.

## Acknowledgements

We would like to express our appreciation to the contributors and developers of Esp-dl, as well as the creators of the datasets used in this project.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome any suggestions, bug fixes, or improvements.

---

We hope you find this project informative and useful. If you have any questions or suggestions, please feel free to reach out.

Happy coding!
