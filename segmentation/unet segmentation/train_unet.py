from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 96
img_cols = 96

smooth = 1.







def load_train_data():
    imgs_train = np.load('numpy_data/imgs_train_resized.npy')
    imgs_mask = np.load('numpy_data/imgs_mask_train_resized.npy')
    imgs_mask1 = np.load('numpy_data/imgs_mask1_train_resized.npy')
    # imgs_train = np.load('imgs_train.npy')
    # imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask, imgs_mask1


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


'''
new_rows = (rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] + output_padding[0]
new_cols = (cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] + output_padding[1]
'''

def get_unet():
    inputs = Input((96, 96, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    # conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    # up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)


    break1 = Conv2D(32, (3,3), activation='relu', padding="same")(conv8)
    output1 = Conv2D(1, (3,3), activation='sigmoid', padding="same")(break1)


    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    output = conv10

    model = Model(inputs=[inputs], outputs=[output1, output])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


# Using TensorFlow backend.
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
# input_1 (InputLayer)            (None, 96, 96, 1)    0                                            
# __________________________________________________________________________________________________
# conv2d_1 (Conv2D)               (None, 96, 96, 32)   320         input_1[0][0]                    
# __________________________________________________________________________________________________
# conv2d_2 (Conv2D)               (None, 96, 96, 32)   9248        conv2d_1[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_1 (MaxPooling2D)  (None, 48, 48, 32)   0           conv2d_2[0][0]                   
# __________________________________________________________________________________________________
# conv2d_3 (Conv2D)               (None, 48, 48, 64)   18496       max_pooling2d_1[0][0]            
# __________________________________________________________________________________________________
# conv2d_4 (Conv2D)               (None, 48, 48, 64)   36928       conv2d_3[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_2 (MaxPooling2D)  (None, 24, 24, 64)   0           conv2d_4[0][0]                   
# __________________________________________________________________________________________________
# conv2d_5 (Conv2D)               (None, 24, 24, 128)  73856       max_pooling2d_2[0][0]            
# __________________________________________________________________________________________________
# conv2d_6 (Conv2D)               (None, 24, 24, 128)  147584      conv2d_5[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_3 (MaxPooling2D)  (None, 12, 12, 128)  0           conv2d_6[0][0]                   
# __________________________________________________________________________________________________
# conv2d_7 (Conv2D)               (None, 12, 12, 256)  295168      max_pooling2d_3[0][0]            
# __________________________________________________________________________________________________
# conv2d_8 (Conv2D)               (None, 12, 12, 256)  590080      conv2d_7[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_4 (MaxPooling2D)  (None, 6, 6, 256)    0           conv2d_8[0][0]                   
# __________________________________________________________________________________________________
# conv2d_9 (Conv2D)               (None, 6, 6, 512)    1180160     max_pooling2d_4[0][0]            
# __________________________________________________________________________________________________
# conv2d_10 (Conv2D)              (None, 6, 6, 512)    2359808     conv2d_9[0][0]                   
# __________________________________________________________________________________________________
# conv2d_transpose_1 (Conv2DTrans (None, 12, 12, 256)  524544      conv2d_10[0][0]                  
# __________________________________________________________________________________________________
# concatenate_1 (Concatenate)     (None, 12, 12, 512)  0           conv2d_transpose_1[0][0]         
#                                                                  conv2d_8[0][0]                   
# __________________________________________________________________________________________________
# conv2d_11 (Conv2D)              (None, 12, 12, 256)  1179904     concatenate_1[0][0]              
# __________________________________________________________________________________________________
# conv2d_12 (Conv2D)              (None, 12, 12, 256)  590080      conv2d_11[0][0]                  
# __________________________________________________________________________________________________
# conv2d_transpose_2 (Conv2DTrans (None, 24, 24, 128)  131200      conv2d_12[0][0]                  
# __________________________________________________________________________________________________
# concatenate_2 (Concatenate)     (None, 24, 24, 256)  0           conv2d_transpose_2[0][0]         
#                                                                  conv2d_6[0][0]                   
# __________________________________________________________________________________________________
# conv2d_13 (Conv2D)              (None, 24, 24, 128)  295040      concatenate_2[0][0]              
# __________________________________________________________________________________________________
# conv2d_14 (Conv2D)              (None, 24, 24, 128)  147584      conv2d_13[0][0]                  
# __________________________________________________________________________________________________
# conv2d_transpose_3 (Conv2DTrans (None, 48, 48, 64)   32832       conv2d_14[0][0]                  
# __________________________________________________________________________________________________
# concatenate_3 (Concatenate)     (None, 48, 48, 128)  0           conv2d_transpose_3[0][0]         
#                                                                  conv2d_4[0][0]                   
# __________________________________________________________________________________________________
# conv2d_15 (Conv2D)              (None, 48, 48, 64)   73792       concatenate_3[0][0]              
# __________________________________________________________________________________________________
# conv2d_16 (Conv2D)              (None, 48, 48, 64)   36928       conv2d_15[0][0]                  
# __________________________________________________________________________________________________
# conv2d_transpose_4 (Conv2DTrans (None, 96, 96, 32)   8224        conv2d_16[0][0]                  
# __________________________________________________________________________________________________
# concatenate_4 (Concatenate)     (None, 96, 96, 64)   0           conv2d_transpose_4[0][0]         
#                                                                  conv2d_2[0][0]                   
# __________________________________________________________________________________________________
# conv2d_17 (Conv2D)              (None, 96, 96, 32)   18464       concatenate_4[0][0]              
# __________________________________________________________________________________________________
# conv2d_18 (Conv2D)              (None, 96, 96, 32)   9248        conv2d_17[0][0]                  
# __________________________________________________________________________________________________
# conv2d_19 (Conv2D)              (None, 96, 96, 1)    33          conv2d_18[0][0]                  
# ==================================================================================================
# Total params: 7,759,521
# Trainable params: 7,759,521
# Non-trainable params: 0
# __________________________________________________________________________________________________




def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask, imgs_mask1 = load_train_data()
    imgs_train  = imgs_train.reshape(-1,96,96,1)
    imgs_mask, imgs_mask1 = imgs_mask.reshape(-1,96,96,1), imgs_mask1.reshape(-1,48,48,1) 

    print("imgs_train: {}, imgs_mask1: {}, imgs_mask: {}".format(imgs_train.shape, 
        imgs_mask1.shape, imgs_mask.shape))
    # imgs_train = preprocess(imgs_train)
    # imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask = imgs_mask.astype('float32')
    imgs_mask /= 255.  # scale masks to [0, 1]
    imgs_mask1 = imgs_mask1.astype('float32')
    imgs_mask1 /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, [imgs_mask1, imgs_mask], batch_size=32, 
                nb_epoch=20*5, verbose=2, shuffle=True,
                validation_split=0.2,
                callbacks=[model_checkpoint])

    print("training done")
    print('-'*30)
    # print('Loading and preprocessing test data...')
    # print('-'*30)
    # imgs_test, imgs_id_test = load_test_data()
    # imgs_test = preprocess(imgs_test)

    # imgs_test = imgs_test.astype('float32')
    # imgs_test -= mean
    # imgs_test /= std

    # print('-'*30)
    # print('Loading saved weights...')
    # print('-'*30)
    # model.load_weights('weights.h5')

    # print('-'*30)
    # print('Predicting masks on test data...')
    # print('-'*30)
    # imgs_mask_test = model.predict(imgs_test, verbose=1)
    # np.save('imgs_mask_test.npy', imgs_mask_test)

    # print('-' * 30)
    # print('Saving predicted masks to files...')
    # print('-' * 30)
    # pred_dir = 'preds'
    # if not os.path.exists(pred_dir):
    #     os.mkdir(pred_dir)
    # for image, image_id in zip(imgs_mask_test, imgs_id_test):
    #     image = (image[:, :, 0] * 255.).astype(np.uint8)
    #     imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

if __name__ == '__main__':
    train_and_predict()





# def get_unet():
#     inputs = Input((img_rows, img_cols, 1))
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

#     up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
#     conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
#     conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

#     up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
#     conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
#     conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

#     up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
#     conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
#     conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

#     up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
#     conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
#     conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

#     conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

#     model = Model(inputs=[inputs], outputs=[conv10])

#     model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

#     return model