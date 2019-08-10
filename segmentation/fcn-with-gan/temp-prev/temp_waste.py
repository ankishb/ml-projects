
import os

# print(os.listdir('../ultrasound-nerve-segmentation-master/raw/'))

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K


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

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

model = get_unet()
print(model.summary())



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
# None