

from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread
from skimage.transform import resize
from skimage.io import imsave

import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K


def get_pool():
    inputs = Input((96, 96, 1))
    pool1 = MaxPooling2D(pool_size=(2, 2))(inputs)

    model = Model(inputs=[inputs], outputs=[pool1])

    return model

# model = get_pool()
# print(model.summary())
# # data_path = 'raw/'

data_path = '../ultrasound-nerve-segmentation-master/raw/'


image_rows = 420
image_cols = 580


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = int(len(images) / 2)

    imgs = np.ndarray((total, 96, 96), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 96, 96), dtype=np.uint8)
    imgs_mask1 = np.ndarray((total, 48, 48), dtype=np.uint8)
    
    # imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    # imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)

        img = resize(img, (96, 96), preserve_range=True)
        img_mask = resize(img_mask, (96, 96), preserve_range=True)
        

        
        img = np.array([img])
        img_mask = np.array([img_mask])
        # img_mask1 = np.array([img_mask1])

        imgs[i] = img
        imgs_mask[i] = img_mask
        # imgs_mask1[i] = img_mask1

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('numpy_data/imgs_train_resized.npy', imgs)
    np.save('numpy_data/imgs_mask_train_resized.npy', imgs_mask)
    print("Saved img and its masks")
    del imgs_mask

    imgs_pool = []
    for i in range(int(5635/5)):
        pick_batch = imgs[i*5:(i+1)*5,::]
        pred_batch = model.predict(imgs.reshape(-1,96,96,1))
        imgs_pool.append(pred_batch)
    imgs_pool = np.array(imgs_pool)
    print("imgs_pooling: ",imgs_pool.shape)
    # imgs_mask1 = preprocess(imgs_pool)

    # img_mask1 = resize(img_pool, (48, 48), preserve_range=True)
    
    np.save('numpy_data/imgs_mask1_train_resized.npy', imgs_pool)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

def create_small_mask_data():
    imgs_mask1 = np.ndarray((5635, 48, 48, 1), dtype=np.uint8)
    # imgs_mask1 = np.ndarray((25, 48, 48, 1), dtype=np.uint8)
    
    model = get_pool()
    imgs = np.load('numpy_data/imgs_mask_train_resized.npy')
    print("images size: ",imgs.shape)
    imgs_pool = []
    for i in range(int(5635/5)):
        pick_batch = imgs[i*5:(i+1)*5,::]
        print("batch size: ", pick_batch.shape)
        pred_batch = model.predict(pick_batch.reshape(-1,96,96,1))
        # imgs_pool.append(pred_batch)

        print("batch size: ", pred_batch.shape)
        imgs_mask1[i*5:(i+1)*5] = pred_batch

        
        if i%100 == 0:
            print("{}/5635".format(i))
    # imgs_pool = np.array(imgs_pool)
    print("imgs_pooling: ",imgs_mask1.shape)
    # imgs_mask1 = preprocess(imgs_pool)

    # img_mask1 = resize(img_pool, (48, 48), preserve_range=True)
    
    np.save('numpy_data/imgs_mask1_train_resized.npy', imgs_mask1)
    print('Saving to .npy files done.')


if __name__ == '__main__':
    create_small_mask_data()
    # create_train_data()
    # create_test_data()
