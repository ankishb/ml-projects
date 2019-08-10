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


for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
'''
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
'''