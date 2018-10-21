#------ modified by Xiaoxi Gong 2018/05/21 ------#
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D 
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from matplotlib import pyplot as plt
from keras.layers.core import Lambda
from keras import backend as K
import os
import cv2
import math
import numpy as np
import random

data = '0521'
read_and_save_path = 'D:/datasets_pre_01/'
test_dataset_Ratio = 0.92 # 0.86

### Images for training
img_Num = 21291   
theEpoch = 200   
fcn_out = 6
h5File_path = read_and_save_path + 'EDI_based_Pose_Estimation_' + data + '.h5'
batchSize = 64 

### Load ground truth
gnd_diff = np.loadtxt(read_and_save_path + 'gnd_diff.txt')
### Datasets path
img_path = read_and_save_path + 'rgb/'
### Outputs
y_gndTruth = np.empty((img_Num, fcn_out), dtype='float32')

for gnd_conv in range(img_Num):
    y_gndTruth[gnd_conv, 0:] = gnd_diff[gnd_conv, 2:]

imgFile = open(read_and_save_path + 'gnd_diff.txt')
imgList = imgFile.readlines()

### Resize to 240 x 320 x 3
img_resize_rate = 0.5
img_height = math.floor(img_resize_rate * 480)
img_width = math.floor(img_resize_rate * 640)
img_channel = 3
net_layer = [24, 36, 48, 64, 64, fcn_out *32, fcn_out *16, fcn_out *4, fcn_out]
model_activation = 'elu'
model = Sequential() ### x: x/127.5-1.0
### Normalize input 
#model.add(BatchNormalization(input_shape=(img_height,img_width,img_channel)))
model.add(Lambda(lambda x: x, input_shape=(img_height,img_width,img_channel)))
model.add(Conv2D(net_layer[0], (5, 5), activation=model_activation, strides=(2, 2)))
model.add(Conv2D(net_layer[1], (5, 5), activation=model_activation, strides=(2, 2)))
model.add(Conv2D(net_layer[2], (5, 5), activation=model_activation, strides=(2, 2)))
model.add(Conv2D(net_layer[3], (3, 3), activation=model_activation))
model.add(Conv2D(net_layer[4], (3, 3), activation=model_activation))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(net_layer[5], activation=model_activation))
model.add(Dense(net_layer[6], activation=model_activation))
model.add(Dense(net_layer[7], activation=model_activation))
model.add(Dense(net_layer[8]))
model.summary()
learning_rate = 1.0e-4
# mean_squared_error  mean_absolute_error  mean_absolute_percentage_error 
model.compile(loss='mean_absolute_error', optimizer=Adam(lr=learning_rate), metrics=['binary_accuracy'])

def train_mode(prediction):
    cnt_rgb = 0
    x_imgeCrop = np.empty((img_Num, img_height, img_width, img_channel), dtype='float32')
    while 1:
        ### Reading images
        img_name, res = imgList[cnt_rgb].split(' ', 1)
        imgEDI = img_path + img_name + '.png'
        if (os.path.exists(imgEDI)):
            imgR = cv2.imread(imgEDI)
            imgR = cv2.resize(imgR, (img_width, img_height), interpolation=cv2.INTER_AREA)
            x_imgeCrop[cnt_rgb, :, :, :] = imgR[:, :, :]
            cnt_rgb = cnt_rgb + 1
            print('Image load: ' + str(img_Num) + ' / ' + str(cnt_rgb + 1))

        if (not imgList or cnt_rgb >= img_Num - 1):
            cnt_rgb = 0
            break

    ### Shuffled dataset
    index = [f for f in range(img_Num - 1)]
    np.random.seed(1028)
    random.shuffle(index)
    x_imgeCrop_index = x_imgeCrop[index,:,:,:]
    y_gndTruth_index = y_gndTruth[index,:]
    splitpoint = int(round((img_Num - 1) * test_dataset_Ratio))
    (x_cTrain, x_cTest) = (x_imgeCrop_index[0:splitpoint,:,:,:], x_imgeCrop_index[splitpoint:,:,:,:])
    (y_cTrain, y_cTest) = (y_gndTruth_index[0:splitpoint,:], y_gndTruth_index[splitpoint:,:])

    history_callback = model.fit(x_cTrain, y_cTrain, epochs=theEpoch, batch_size=batchSize, shuffle=True)
    loss_and_metrics = model.evaluate(x_cTest, y_cTest, batch_size=batchSize)
    model.save_weights(h5File_path)

    ### Pose prediction
    if (os.path.exists(h5File_path)):
        x_img = np.empty((1, img_height, img_width, img_channel), dtype='float32')
        y_pre = np.empty(( (img_Num - splitpoint - 1), fcn_out ), dtype='float32')
        model.load_weights(h5File_path)

        for cnt_rgb1 in range(0, img_Num - splitpoint - 1):
            x_img[0, :, :, :] = x_cTest[cnt_rgb1, :, :, :]
            pred = model.predict(x_img) 
            y_pre[cnt_rgb1,:] = pred[0, :]  

        plt.subplot(231)
        plt.plot(y_pre[:, 0], 'r')
        plt.plot(y_cTest[:, 0], 'b', )
        plt.subplot(232)
        plt.plot(y_pre[:, 1], 'r')
        plt.plot(y_cTest[:, 1], 'b', )
        plt.subplot(233)
        plt.plot(y_pre[:, 2], 'r')
        plt.plot(y_cTest[:, 2], 'b')

        plt.subplot(234)
        plt.plot(y_pre[:, 3], 'r')
        plt.plot(y_cTest[:, 3], 'b', )
        plt.subplot(235)
        plt.plot(y_pre[:, 4], 'r')
        plt.plot(y_cTest[:, 4], 'b', )
        plt.subplot(236)
        plt.plot(y_pre[:, 5], 'r')
        plt.plot(y_cTest[:, 5], 'b')
        plt.show()

        pre_record = open(read_and_save_path + prediction + '.txt', 'ab')
        np.savetxt(pre_record, y_pre)
        pre_record.close()

train_mode('pose_prediction')

