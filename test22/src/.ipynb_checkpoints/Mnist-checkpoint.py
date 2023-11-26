# import our needed libraries
import torch
import torchvision
import os
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import onnx
import onnxruntime
from torch import nn
from optimizer import *
from calibrator import *

train_flag = input("Do you want to train it?(y/n) : ")





# find GPU to transfer our process on it
device = 'cuda' if torch.cuda.is_available else 'cpu'
print('device : ', device)
print('==================')


# define parameters
batch_size = 256
input_size = (32, 32)


# Loading our dataset

# get the working directory path
current_dir = os.getcwd()
print('my directory : ', current_dir)

training_dataset = torchvision.datasets.ImageFolder(
    root = '../archive/Training/',
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.ToTensor()
    ])
)

test_dataset = torchvision.datasets.ImageFolder(
    root = '../archive/Validation/',
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.ToTensor()
    ])    
)

training_loader = torch.utils.data.DataLoader(training_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                             )

validation_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                               )

validation_iter = iter(validation_loader)
X_test, y_test= next(validation_iter)
X_test = X_test.numpy()
y_test = y_test.numpy()
print('X test shape : ', X_test.shape)
print('y test shaep : ', y_test.shape)
print('------------------------------')
print('has been loaded succesfully')
print('------------------------------')


# 2) Check the dataset
data_iter = iter(training_loader)
images, labels = next(data_iter)
classes = training_loader.dataset.classes
print('image shape : ', images.shape)
print('lebels shape : ', labels.shape)

def image_converter(tensor):
    image = tensor.clone().detach().numpy()
    image = image.transpose(1, 2, 0)
#     image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    # image = image.clip(0, 1)
#     image = color.rgb2gray(image)
    return image

print('image shape : ', image_converter(images[3]).shape)
print(training_loader.dataset.class_to_idx)
fig = plt.figure(figsize=(12, 6))
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    plt.imshow(image_converter(images[i]))
    ax.set_title(labels[i].item())

plt.show()



# create input file 
print('input label : ', labels[0])
print('----------------------------')
my_in = image_converter(images[0])
my_input = my_in.reshape(-1)
with open('input.txt', 'w') as file:
    # Convert each element to a string and join them with commas
    output_string = ', '.join(map(str, my_input))
    
    # Write the output string to the file
    file.write(output_string)




print('------------------------------')
print('has been visualized succesfully')
print('------------------------------')


# 3) Build our model 
class Mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, (3, 3), 1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, (3, 3), 1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, (3, 3), 1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, (3, 3), 1, padding=1)
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, (2, 2))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
torch_model = Mnist().to(device)
print(torch_model)

print('------------------------------')
print('has been builded succesfully')
print('------------------------------')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.001)


# 4) Train our model 
if train_flag == 'y':
    
    epochs = 1
    training_loss = []
    training_acc = []
    validation_loss = []
    validation_acc = []
    for e in range(epochs):
        losses = 0.0
        accuracies = 0.0
        for images, labels in training_loader: 
    
            inputs = images.to(device)
            outputs = torch_model(inputs).to(device)
            loss = criterion(outputs.cpu(), labels)
            
            losses += loss.item()
            _, max_index = torch.max(outputs, 1)
            accuracies += torch.sum(max_index.cpu() == labels.cpu()) / labels.shape[0]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        else: 
            epoch_loss = losses / len(training_loader)
            training_loss.append(epoch_loss)
            epoch_acc = accuracies / len(training_loader)
            training_acc.append(epoch_acc)
            
            with torch.no_grad():
                losses = 0.0
                accuracies = 0.0
                for images, labels in validation_loader:
                    inputs = images.to(device)
                    outputs = torch_model(inputs).to(device)
                    loss = criterion(outputs.cpu(), labels)
                    
                    losses += loss.item()
                    _, max_index = torch.max(outputs, 1)
                    accuracies += torch.sum(max_index.cpu() == labels.cpu()) / labels.shape[0]
                    
                validation_epoch_loss = losses / len(validation_loader)
                validation_loss.append(validation_epoch_loss)
                validation_epoch_acc = accuracies / len(validation_loader)
                validation_acc.append(validation_epoch_acc)
                
            print("epoch :{}, T_acc={:.4f}, T_loss={:.4f} || V_acc={:.4f}, V_loss={:.4f}".format(e, 
                                                                                                 epoch_acc,
                                                                                                 epoch_loss,
                                                                                                 validation_epoch_acc,
                                                                                                 validation_epoch_loss
                                                                                                 ))                            
    print('------------------------------')
    print('has been trained succesfully')
    print('------------------------------')



    # 5) plot loss and accuracy curves
    plt.plot(range(epochs), training_loss, label='training loss')
    plt.plot(range(epochs), validation_loss, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
    plt.plot(range(epochs), training_acc, label='training accuracy')
    plt.plot(range(epochs), validation_acc, label='validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
    # 6) Testing our model
    # data_iter = iter(validation_loader)
    images, labels = next(data_iter)
    outputs = torch_model(images.to(device))
    _, preds = torch.max(outputs, 1)
    
    fig = plt.figure(figsize=(25, 14))
    for idx in range(20): 
        ax = fig.add_subplot(4, 5, idx+1)
        plt.imshow(image_converter(images[idx]))
        ax.set_title("{}({})".format(classes[labels[idx]], classes[preds[idx]]),
                     color='green' if labels[idx]==preds[idx] else 'red')
    
    plt.show()
    
    
    print('------------------------------')
    print('has been tested succesfully')
    print('------------------------------')



# # 7) Saving the model
x = torch.randn(batch_size, 3, input_size[0], input_size[1], requires_grad=True)

# Export the model
torch.onnx.export(torch_model.cpu(),               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "Mnist_model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=13,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                )


print('------------------------------')
print('has been saved succesfully')
print('------------------------------')


# load model
model = onnx.load('Mnist_model.onnx')
onnx.checker.check_model(model)
print('------------------------------')
print('has been loaded succesfully')
print('------------------------------')











# 9) Calibration

# optimize the ONNX model 
optimized_model_path = optimize_fp_model("Mnist_model.onnx")
test_images = X_test.copy()
test_labels = y_test.copy()
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
log_file = open("coefficient.log","w")
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





# ----------------------
images, labels = next(validation_iter)

# x = torch.randn(batch_size, 3, 128, 128, requires_grad=True)

# ort_session = onnxruntime.InferenceSession("Mnist_model.onnx")

# # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# # ort_inputs = [
# #     {ort_session.get_inputs()[0].name: x[i, :, :, :]} for i in range(2)
# # ]
# ort_inputs = {ort_session.get_inputs()[0].name: x.detach().cpu().numpy()}
# ort_outs = ort_session.run(None, ort_inputs)
# ort_outs = np.concatenate(ort_outs, axis=0)

# #############################################
# torch_outs = torch_model(images)
# np.testing.assert_allclose(torch_outs.detach().cpu().numpy(), ort_outs, rtol=1e-03, atol=1e-05)
# #############################################


# print(np.mean(labels.numpy()==ort_outs.argmax(axis=1)))



# # compare ONNX Runtime and PyTorch results
# # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

# print("Exported model has been tested with ONNXRuntime, and the result looks good!")
# ----------------------










# 10) Evaluation 


# 11)prediction

# save prediction output file in a log file
# old_stdout = sys.stdout
# log_file = open("prediction.log","w")
# sys.stdout = log_file


# 11)prediction
# print('---------------------------------')
# my_weights = model.get_weights() 
# print('weights : ', my_weights)
# print('---------------------------------')

# print('prediction values : ')
# cal_num = 30
# # X_test = X_test[1]
# print('shape : : ', X_test.shape)


# pred = model.predict(X_test)
# for i in range(cal_num):
#     x = X_cal[i] 
# #     print('x : ', x)
#     print('pred(', i, ') : ', pred[i])
#     pred_arg = np.argmax(pred[i])
#     print('arg : ', pred_arg)
#     print('y : ', np.argmax(y_cal[i]))
#     print('--------------------')
# print('------------------------------')
# print('has been predicted succesfully')
# print('------------------------------')



# # 12) create input for esp32
# # my_in = X_cal[23]
# my_in = X_cal[17]
# my_input = my_in.reshape(-1)
# with open('input.txt', 'w') as file:
#     # Convert each element to a string and join them with commas
#     output_string = ', '.join(map(str, my_input))
    
#     # Write the output string to the file
#     file.write(output_string)

# cv2.imshow('input image', my_in)
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 


# save prediction output file in a log file
# sys.stdout = old_stdout
# log_file.close()


print('\n==============================================End of this part')








