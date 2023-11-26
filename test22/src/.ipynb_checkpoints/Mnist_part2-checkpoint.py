# import our needed libraries
import torch
import torchvision
import numpy as np 
import sys 
import onnx
import onnxruntime
from optimizer import *
from calibrator import *


# find GPU to transfer our process on it
device = 'cuda' if torch.cuda.is_available else 'cpu'
print('device : ', device)
print('==================')


# load our dataset

batch_size = int(np.load('batch_size.npy'))
input_size = np.load('input_size.npy')

test_dataset = torchvision.datasets.ImageFolder(
    root = '../archive/Validation/',
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size[0], input_size[1])),
        torchvision.transforms.ToTensor()
    ])    
)

validation_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                               )
print('------------------------------')
print('has been loaded succesfully')
print('------------------------------')


validation_iter = iter(validation_loader)
X_test, y_test= next(validation_iter)
X_test = X_test.numpy()
y_test = y_test.numpy()
print('X test shape : ', X_test.shape)
print('y test shaep : ', y_test.shape)
print('----------------------------')



# load model
model = onnx.load('Mnist_model.onnx')
onnx.checker.check_model(model)


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
# images, labels = next(validation_iter)
# outputs = model(images)

x = torch.randn(batch_size, 3, 128, 128, requires_grad=True)
# torch_out = torch_model(x)


EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

ort_session = onnxruntime.InferenceSession("Mnist_model.onnx", providers=["CPUExecutionProvider"])
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)


# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
# ----------------------










# 10) Evaluation 


# # 11)prediction

# # save prediction output file in a log file
# # old_stdout = sys.stdout
# # log_file = open("prediction.log","w")
# # sys.stdout = log_file


# # 11)prediction
# # print('---------------------------------')
# # my_weights = model.get_weights() 
# # print('weights : ', my_weights)
# # print('---------------------------------')

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


# # save prediction output file in a log file
# sys.stdout = old_stdout
# log_file.close()


print('\n==============================================End of this part')
