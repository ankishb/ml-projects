

# Hello! This rather quick and dirty kernel shows how to get started on segmenting nuclei using a neural network in Keras.

# The architecture used is the so-called U-Net, which is very common for image segmentation problems such as this. I believe they also have a tendency to work quite well even on small datasets.

# Let's get started importing everything we need!

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
# deprication warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

seed = 42
random.seed = seed
np.random.seed = seed

# Using TensorFlow backend.
'''
>>> for (root, dirs, files) in os.walk(path, topdown=True):
...     print(root)
...     print(dirs)
...     print(files)
...     print('-'*20)
'''


'''
>>> next(os.walk('omniglot_dataset/images_evaluation/Kannada/character25'))[2]
['1229_14.png', '1229_02.png', '1229_15.png', '1229_20.png', '1229_06.png', '1229_11.png', '1229_05.png', 
'1229_10.png', '1229_19.png', '1229_13.png', '1229_04.png', '1229_03.png', '1229_08.png', '1229_16.png', 
'1229_17.png', '1229_01.png', '1229_07.png', '1229_18.png', '1229_09.png', '1229_12.png']
>>> 

'''
# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# Get the data

# Let's first import all the images and associated masks. I downsample both the training and test images to keep things light and manageable, but we need to keep a record of the original sizes of the test images to upsample our predicted masks and create correct run-length encodings later on. There are definitely better ways to handle this, but it works fine for now!

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    '''
    For 1 image id, we have as many number of mask images as the total number of dots or hole in the image id.
    '''
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img

    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask


# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')

# Let's see if things look all right by drawing some random images and their associated masks.

# Check if training data looks all right
ix = random.randint(0, len(train_ids))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()

# Seems good!
# Create our Keras metric

# # Now we try to define the mean average precision at different intersection over union (IoU) thresholds metric in Keras. 
# TensorFlow has a mean IoU metric, but it doesn't have any native support for the mean over multiple thresholds, so I tried 
# to implement this. I'm by no means certain that this implementation is correct, though! Any assistance in verifying this 
# would be most welcome!

# # Update: This implementation is most definitely not correct due to the very large discrepancy between the results reported here 
# and the LB results. It also seems to just increase over time no matter what when you train ...

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# loss = dice_coef_loss  <<<<===========================================================





# Build and train our neural network

# Next we build our U-Net model, loosely based on U-Net: Convolutional Networks for Biomedical Image Segmentation 
# and very similar to this repo from the Kaggle Ultrasound Nerve Segmentation competition.

# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef_loss])
model.summary()

# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
# input_1 (InputLayer)            (None, 128, 128, 3)  0                                            
# __________________________________________________________________________________________________
# lambda_1 (Lambda)               (None, 128, 128, 3)  0           input_1[0][0]                    
# __________________________________________________________________________________________________
# conv2d_1 (Conv2D)               (None, 128, 128, 8)  224         lambda_1[0][0]                   
# __________________________________________________________________________________________________
# conv2d_2 (Conv2D)               (None, 128, 128, 8)  584         conv2d_1[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_1 (MaxPooling2D)  (None, 64, 64, 8)    0           conv2d_2[0][0]                   
# __________________________________________________________________________________________________
# conv2d_3 (Conv2D)               (None, 64, 64, 16)   1168        max_pooling2d_1[0][0]            
# __________________________________________________________________________________________________
# conv2d_4 (Conv2D)               (None, 64, 64, 16)   2320        conv2d_3[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_2 (MaxPooling2D)  (None, 32, 32, 16)   0           conv2d_4[0][0]                   
# __________________________________________________________________________________________________
# conv2d_5 (Conv2D)               (None, 32, 32, 32)   4640        max_pooling2d_2[0][0]            
# __________________________________________________________________________________________________
# conv2d_6 (Conv2D)               (None, 32, 32, 32)   9248        conv2d_5[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_3 (MaxPooling2D)  (None, 16, 16, 32)   0           conv2d_6[0][0]                   
# __________________________________________________________________________________________________
# conv2d_7 (Conv2D)               (None, 16, 16, 64)   18496       max_pooling2d_3[0][0]            
# __________________________________________________________________________________________________
# conv2d_8 (Conv2D)               (None, 16, 16, 64)   36928       conv2d_7[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_4 (MaxPooling2D)  (None, 8, 8, 64)     0           conv2d_8[0][0]                   
# __________________________________________________________________________________________________
# conv2d_9 (Conv2D)               (None, 8, 8, 128)    73856       max_pooling2d_4[0][0]            
# __________________________________________________________________________________________________
# conv2d_10 (Conv2D)              (None, 8, 8, 128)    147584      conv2d_9[0][0]                   
# __________________________________________________________________________________________________
# conv2d_transpose_1 (Conv2DTrans (None, 16, 16, 64)   32832       conv2d_10[0][0]                  
# __________________________________________________________________________________________________
# concatenate_1 (Concatenate)     (None, 16, 16, 128)  0           conv2d_transpose_1[0][0]         
#                                                                  conv2d_8[0][0]                   
# __________________________________________________________________________________________________
# conv2d_11 (Conv2D)              (None, 16, 16, 64)   73792       concatenate_1[0][0]              
# __________________________________________________________________________________________________
# conv2d_12 (Conv2D)              (None, 16, 16, 64)   36928       conv2d_11[0][0]                  
# __________________________________________________________________________________________________
# conv2d_transpose_2 (Conv2DTrans (None, 32, 32, 32)   8224        conv2d_12[0][0]                  
# __________________________________________________________________________________________________
# concatenate_2 (Concatenate)     (None, 32, 32, 64)   0           conv2d_transpose_2[0][0]         
#                                                                  conv2d_6[0][0]                   
# __________________________________________________________________________________________________
# conv2d_13 (Conv2D)              (None, 32, 32, 32)   18464       concatenate_2[0][0]              
# __________________________________________________________________________________________________
# conv2d_14 (Conv2D)              (None, 32, 32, 32)   9248        conv2d_13[0][0]                  
# __________________________________________________________________________________________________
# conv2d_transpose_3 (Conv2DTrans (None, 64, 64, 16)   2064        conv2d_14[0][0]                  
# __________________________________________________________________________________________________
# concatenate_3 (Concatenate)     (None, 64, 64, 32)   0           conv2d_transpose_3[0][0]         
#                                                                  conv2d_4[0][0]                   
# __________________________________________________________________________________________________
# conv2d_15 (Conv2D)              (None, 64, 64, 16)   4624        concatenate_3[0][0]              
# __________________________________________________________________________________________________
# conv2d_16 (Conv2D)              (None, 64, 64, 16)   2320        conv2d_15[0][0]                  
# __________________________________________________________________________________________________
# conv2d_transpose_4 (Conv2DTrans (None, 128, 128, 8)  520         conv2d_16[0][0]                  
# __________________________________________________________________________________________________
# concatenate_4 (Concatenate)     (None, 128, 128, 16) 0           conv2d_transpose_4[0][0]         
#                                                                  conv2d_2[0][0]                   
# __________________________________________________________________________________________________
# conv2d_17 (Conv2D)              (None, 128, 128, 8)  1160        concatenate_4[0][0]              
# __________________________________________________________________________________________________
# conv2d_18 (Conv2D)              (None, 128, 128, 8)  584         conv2d_17[0][0]                  
# __________________________________________________________________________________________________
# conv2d_19 (Conv2D)              (None, 128, 128, 1)  9           conv2d_18[0][0]                  
# ==================================================================================================
# Total params: 485,817
# Trainable params: 485,817
# Non-trainable params: 0
# __________________________________________________________________________________________________

# Next we fit the model on the training data, using a validation split of 0.1. We use a small batch size because we have so 
# ittle data. I recommend using checkpointing and early stopping when training your model. I won't do it here to make things a 
# bit more reproducible (although it's very likely that your results will be different anyway). I'll just train for 10 epochs, 
#     which takes around 10 minutes in the Kaggle kernel with the current parameters.

# Update: Added early stopping and checkpointing and increased to 30 epochs.

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=30,
                    callbacks=[earlystopper, checkpointer])

# Train on 603 samples, validate on 67 samples
# Epoch 1/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.4475 - mean_iou: 0.4181
# Epoch 00001: val_loss improved from inf to 0.29506, saving model to model-dsbowl2018-1.h5
# 603/603 [==============================] - 53s 88ms/step - loss: 0.4465 - mean_iou: 0.4181 - val_loss: 0.2951 - val_mean_iou: 0.4232
# Epoch 2/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.2412 - mean_iou: 0.4257
# Epoch 00002: val_loss improved from 0.29506 to 0.17167, saving model to model-dsbowl2018-1.h5
# 603/603 [==============================] - 51s 84ms/step - loss: 0.2414 - mean_iou: 0.4258 - val_loss: 0.1717 - val_mean_iou: 0.4368
# Epoch 3/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.1663 - mean_iou: 0.4589
# Epoch 00003: val_loss improved from 0.17167 to 0.14400, saving model to model-dsbowl2018-1.h5
# 603/603 [==============================] - 50s 83ms/step - loss: 0.1663 - mean_iou: 0.4590 - val_loss: 0.1440 - val_mean_iou: 0.4884
# Epoch 4/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.1290 - mean_iou: 0.5220
# Epoch 00004: val_loss improved from 0.14400 to 0.11136, saving model to model-dsbowl2018-1.h5
# 603/603 [==============================] - 50s 84ms/step - loss: 0.1285 - mean_iou: 0.5221 - val_loss: 0.1114 - val_mean_iou: 0.5523
# Epoch 5/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.1155 - mean_iou: 0.5768
# Epoch 00005: val_loss improved from 0.11136 to 0.10928, saving model to model-dsbowl2018-1.h5
# 603/603 [==============================] - 51s 84ms/step - loss: 0.1155 - mean_iou: 0.5769 - val_loss: 0.1093 - val_mean_iou: 0.5990
# Epoch 6/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.1007 - mean_iou: 0.6174
# Epoch 00006: val_loss improved from 0.10928 to 0.09795, saving model to model-dsbowl2018-1.h5
# 603/603 [==============================] - 50s 84ms/step - loss: 0.1006 - mean_iou: 0.6175 - val_loss: 0.0979 - val_mean_iou: 0.6340
# Epoch 7/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0998 - mean_iou: 0.6468
# Epoch 00007: val_loss improved from 0.09795 to 0.08905, saving model to model-dsbowl2018-1.h5
# 603/603 [==============================] - 50s 83ms/step - loss: 0.0998 - mean_iou: 0.6469 - val_loss: 0.0890 - val_mean_iou: 0.6590
# Epoch 8/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0921 - mean_iou: 0.6703
# Epoch 00008: val_loss did not improve
# 603/603 [==============================] - 50s 83ms/step - loss: 0.0923 - mean_iou: 0.6703 - val_loss: 0.0918 - val_mean_iou: 0.6799
# Epoch 9/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0913 - mean_iou: 0.6889
# Epoch 00009: val_loss improved from 0.08905 to 0.08385, saving model to model-dsbowl2018-1.h5
# 603/603 [==============================] - 50s 84ms/step - loss: 0.0912 - mean_iou: 0.6889 - val_loss: 0.0839 - val_mean_iou: 0.6968
# Epoch 10/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0925 - mean_iou: 0.7038
# Epoch 00010: val_loss did not improve
# 603/603 [==============================] - 50s 83ms/step - loss: 0.0922 - mean_iou: 0.7038 - val_loss: 0.0901 - val_mean_iou: 0.7101
# Epoch 11/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0861 - mean_iou: 0.7161
# Epoch 00011: val_loss did not improve
# 603/603 [==============================] - 50s 83ms/step - loss: 0.0862 - mean_iou: 0.7162 - val_loss: 0.0927 - val_mean_iou: 0.7215
# Epoch 12/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0829 - mean_iou: 0.7268
# Epoch 00012: val_loss improved from 0.08385 to 0.08038, saving model to model-dsbowl2018-1.h5
# 603/603 [==============================] - 51s 84ms/step - loss: 0.0830 - mean_iou: 0.7268 - val_loss: 0.0804 - val_mean_iou: 0.7314
# Epoch 13/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0809 - mean_iou: 0.7357
# Epoch 00013: val_loss improved from 0.08038 to 0.07635, saving model to model-dsbowl2018-1.h5
# 603/603 [==============================] - 50s 83ms/step - loss: 0.0813 - mean_iou: 0.7357 - val_loss: 0.0763 - val_mean_iou: 0.7401
# Epoch 14/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0866 - mean_iou: 0.7439
# Epoch 00014: val_loss did not improve
# 603/603 [==============================] - 56s 92ms/step - loss: 0.0864 - mean_iou: 0.7439 - val_loss: 0.0807 - val_mean_iou: 0.7473
# Epoch 15/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0818 - mean_iou: 0.7503
# Epoch 00015: val_loss did not improve
# 603/603 [==============================] - 57s 95ms/step - loss: 0.0816 - mean_iou: 0.7504 - val_loss: 0.0771 - val_mean_iou: 0.7537
# Epoch 16/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0813 - mean_iou: 0.7569
# Epoch 00016: val_loss did not improve
# 603/603 [==============================] - 57s 95ms/step - loss: 0.0812 - mean_iou: 0.7569 - val_loss: 0.0798 - val_mean_iou: 0.7596
# Epoch 17/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0782 - mean_iou: 0.7621
# Epoch 00017: val_loss improved from 0.07635 to 0.07304, saving model to model-dsbowl2018-1.h5
# 603/603 [==============================] - 57s 94ms/step - loss: 0.0783 - mean_iou: 0.7621 - val_loss: 0.0730 - val_mean_iou: 0.7649
# Epoch 18/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0843 - mean_iou: 0.7672
# Epoch 00018: val_loss did not improve
# 603/603 [==============================] - 54s 89ms/step - loss: 0.0842 - mean_iou: 0.7672 - val_loss: 0.0775 - val_mean_iou: 0.7694
# Epoch 19/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0776 - mean_iou: 0.7716
# Epoch 00019: val_loss did not improve
# 603/603 [==============================] - 53s 88ms/step - loss: 0.0778 - mean_iou: 0.7716 - val_loss: 0.0732 - val_mean_iou: 0.7737
# Epoch 20/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0791 - mean_iou: 0.7757
# Epoch 00020: val_loss did not improve
# 603/603 [==============================] - 53s 87ms/step - loss: 0.0790 - mean_iou: 0.7757 - val_loss: 0.0766 - val_mean_iou: 0.7776
# Epoch 21/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0783 - mean_iou: 0.7792
# Epoch 00021: val_loss did not improve
# 603/603 [==============================] - 54s 89ms/step - loss: 0.0781 - mean_iou: 0.7792 - val_loss: 0.0840 - val_mean_iou: 0.7811
# Epoch 22/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0753 - mean_iou: 0.7829
# Epoch 00022: val_loss improved from 0.07304 to 0.07093, saving model to model-dsbowl2018-1.h5
# 603/603 [==============================] - 55s 91ms/step - loss: 0.0755 - mean_iou: 0.7829 - val_loss: 0.0709 - val_mean_iou: 0.7846
# Epoch 23/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0745 - mean_iou: 0.7861
# Epoch 00023: val_loss did not improve
# 603/603 [==============================] - 52s 86ms/step - loss: 0.0744 - mean_iou: 0.7861 - val_loss: 0.0725 - val_mean_iou: 0.7877
# Epoch 24/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0756 - mean_iou: 0.7891
# Epoch 00024: val_loss improved from 0.07093 to 0.06879, saving model to model-dsbowl2018-1.h5
# 603/603 [==============================] - 52s 86ms/step - loss: 0.0756 - mean_iou: 0.7891 - val_loss: 0.0688 - val_mean_iou: 0.7906
# Epoch 25/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0755 - mean_iou: 0.7920
# Epoch 00025: val_loss did not improve
# 603/603 [==============================] - 52s 86ms/step - loss: 0.0756 - mean_iou: 0.7920 - val_loss: 0.0857 - val_mean_iou: 0.7933
# Epoch 26/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0803 - mean_iou: 0.7944
# Epoch 00026: val_loss did not improve
# 603/603 [==============================] - 51s 85ms/step - loss: 0.0803 - mean_iou: 0.7944 - val_loss: 0.0723 - val_mean_iou: 0.7955
# Epoch 27/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0743 - mean_iou: 0.7967
# Epoch 00027: val_loss did not improve
# 603/603 [==============================] - 51s 85ms/step - loss: 0.0744 - mean_iou: 0.7967 - val_loss: 0.0757 - val_mean_iou: 0.7979
# Epoch 28/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0725 - mean_iou: 0.7990
# Epoch 00028: val_loss did not improve
# 603/603 [==============================] - 53s 87ms/step - loss: 0.0723 - mean_iou: 0.7991 - val_loss: 0.0710 - val_mean_iou: 0.8002
# Epoch 29/30
# 600/603 [============================>.] - ETA: 0s - loss: 0.0714 - mean_iou: 0.8013
# Epoch 00029: val_loss did not improve
# 603/603 [==============================] - 53s 88ms/step - loss: 0.0712 - mean_iou: 0.8013 - val_loss: 0.0703 - val_mean_iou: 0.8023
# Epoch 00029: early stopping

# All right, looks good! Loss seems to be a bit erratic, though. I'll leave it to you to improve the model architecture and parameters!
# Make predictions

# Let's make predictions both on the test set, the val set and the train set (as a sanity check). Remember to load the best saved model 
# if you've used early stopping and checkpointing.

# Predict on train, val and test
model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))

# 603/603 [==============================] - 16s 26ms/step
# 67/67 [==============================] - 2s 25ms/step
# 65/65 [==============================] - 2s 26ms/step

# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# The model is at least able to fit to the training data! Certainly a lot of room for improvement even here, but a decent start. 
# How about the validation data?

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()

# Not too shabby! Definitely needs some more training and tweaking.
# Encode and submit our results

# Now it's time to submit our results. I've stolen this excellent implementation of run-length encoding.

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python


'''
     WWWWWWWWWWWWBWWWWWWWWWWWWBBBWWWWWWWWWWWWWWWWWWWWWWWWBWWWWWWWWWWWWWW 

With a run-length encoding (RLE) data compression algorithm applied to the above hypothetical scan line, it can be rendered as follows:

    12W1B12W3B24W1B14W 
'''
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

# Let's iterate over the test IDs and generate run-length encodings for each seperate mask identified by skimage ...

new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# ... and then finally create our submission!

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-1.csv', index=False)

