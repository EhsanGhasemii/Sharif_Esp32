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


# load our dataset
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

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

# load calibration dataset
# with open('X_cal.pkl', 'rb') as f:
#     (test_images) = pickle.load(f)
# with open('y_cal.pkl', 'rb') as f:
#     (test_labels) = peckle.load(f)

test_images = X_train.copy()
test_labels = y_train.copy()

calib_dataset = test_images  
pickle_file_path = 'Mnist_calib.picle'


# calibration
model_proto = onnx.load(optimized_model_path)
print('Generating the quantization table:')

calib = Calibrator('int16', 'per-tensor', 'minmax')
# calib = Calibrator('int8', 'per-tensor', 'minmax')

calib.set_providers(['CPUExecutionProvider'])

# Obtain the quantization parameter
calib.generate_quantization_table(model_proto, calib_dataset, pickle_file_path)

# Generate the coefficient files for esp32s3
calib.export_coefficient_to_cpp(model_proto, pickle_file_path, 'esp32', '.', 'Mnist_coefficient', True)

# print('======================')
# print(type(model_proto))
# print('model proto : ', model_proto)
# print('======================')

# / (1)-- we can change 'esp32s3' to our target -- /

print('---------------------------------')
print('has been calibrated successfuly')
print('---------------------------------')


# 10) Evaluation 

# / (4) we can run thia part to -- / 
"""
print('Evaluating the performance on esp32s3:')
eva = Evaluator('int16', 'per-tensor', 'esp32s3')
eva.set_providers(['CPUExecutionProvider'])
eva.generate_quantized_model(model_proto, pickle_file_path)

output_names = [n.name for n in model_proto.graph.output]
providers = ['CPUExecutionProvider']
m = rt.InferenceSession(optimized_model_path, providers=providers)

batch_size = 64
batch_num = int(len(test_images) / batch_size)
res = 0
fp_res = 0
input_name = m.get_inputs()[0].name
for i in range(batch_num):
    # int8_model
    [outputs, _] = eva.evalaute_quantized_model(test_images[i * batch_size:(i + 1) * batch_size], False)
    res = res + sum(np.argmax(outputs[0], axis=1) == test_labels[i * batch_size:(i + 1) * batch_size])

    # floating-point model
    fp_outputs = m.run(output_names, {input_name: test_images[i * batch_size:(i + 1) * batch_size].astype(np.float32)})
    fp_res = fp_res + sum(np.argmax(fp_outputs[0], axis=1) == test_labels[i * batch_size:(i + 1) * batch_size])
print('accuracy of int8 model is: %f' % (res / len(test_images)))
print('accuracy of fp32 model is: %f' % (fp_res / len(test_images)))
"""





# 11)prediction
print('---------------------------------')
my_weights = model.get_weights() 
print('weights : ', my_weights)
print('---------------------------------')
(a1, a2) = my_weights[0][0]
(b1, b2) = my_weights[1]
print('a1 : ', a1)
print('a2 : ', a2)
print('b1 : ', b1)
print('b2 : ', b2)
print('--------------------')
print('prediction values : ')
for i in range(0, 250, 10):
    x = np.array([[[i]]])
    pred = model.predict(x)
    print('pred(', i, ') : ', pred)
    pred_arg = np.argmax(pred)
    print('arg : ', pred_arg)
    inp1 = a1*i + b1
    inp2 = a2*i + b2
    sm1 = np.exp(inp1) / (np.exp(inp1) + np.exp(inp2))
    print('softmax output : ', sm1, 1-sm1)
print('------------------------------')
print('has been predicted succesfully')
print('------------------------------')


print('\n==============================================End of this part')
