import os
import numpy as np
import pickle

pulmonary_category = {0: 'CON',
                      1: 'M-GGO',
                      2: 'HCM',
                      3: 'EMP',
                      4: 'NOD',
                      5: 'NOR'}


def loadPickle(pklFilePath, pklFileName):
    with open(pklFilePath + pklFileName, 'rb') as f:
        message = pickle.load(f)

    return message


def savePickle(dataArray, filePath, fileName):
    if not os.path.isdir(filePath):
        os.makedirs(filePath)

    with open(filePath + fileName, 'wb') as f:
        pickle.dump(dataArray, f)


def sortVariousPairs(pairList):
    return sorted(pairList, key=lambda x: x[1])


def getFileNameList(filePath):
    l = os.listdir(filePath)
    l = sorted(l, key=lambda x: x[:x.find('.')])

    return l


def onehotEncoder(lib_array, num_class):
    num = lib_array.shape[0]
    onehot_array = np.zeros((num, num_class))

    for i in range(num):
        onehot_array[i][lib_array[i]] = 1

    return onehot_array


# def random_crop(image_batch, PADDING_SIZE=4, PAD_VALUE=-1):
#     new_batch = []
#     pad_width = ((PADDING_SIZE, PADDING_SIZE), (PADDING_SIZE, PADDING_SIZE), (0, 0))
#
#     for i in range(image_batch.shape[0]):
#         new_batch.append(image_batch[i])
#         new_batch[i] = np.pad(image_batch[i], pad_width=pad_width, mode='constant', constant_values=PAD_VALUE)
#         x_offset = np.random.randint(low=0, high=2 * PADDING_SIZE + 1, size=1)[0]
#         y_offset = np.random.randint(low=0, high=2 * PADDING_SIZE + 1, size=1)[0]
#         new_batch[i] = new_batch[i][x_offset:x_offset + 32, y_offset:y_offset + 32, :]
#
#     return new_batch


def random_flip(image_batch):
    for i in range(image_batch.shape[0]):
        flip_prop = np.random.randint(low=0, high=3)
        if flip_prop == 0:
            image_batch[i] = image_batch[i]
        if flip_prop == 1:
            image_batch[i] = np.fliplr(image_batch[i])
        if flip_prop == 2:
            image_batch[i] = np.flipud(image_batch[i])

    return image_batch


def next_batch(image, label, batch_size):
    index = np.random.randint(low=0, high=len(image), size=batch_size)
    img_batch = image[index]
    lab_batch = label[index]
    img_batch = random_flip(img_batch)

    return img_batch, lab_batch


def next_batch_unpaired(image, batch_size):
    index = np.random.randint(low=0, high=len(image), size=batch_size)
    img_batch = image[index]
    img_batch = random_flip(img_batch)

    return img_batch


def normalizeInput(inputData, mode):
    if mode == 'Paired':
        inputData[0] = inputData[0] / 127.5 - 1.0
    elif mode == 'Unpaired':
        inputData = inputData / 127.5 - 1.0
    else:
        print('Error in Normalize Input')
        exit(0)
    print('Normalization Finish')
    return inputData


def loadData(data_domain):
    experimentalPath = '../experiment_data/'

    if data_domain == 'Source':
        src_name = 'source'
        tar_name = 'target'
    elif data_domain == 'Target':
        src_name = 'target'
        tar_name = 'source'
    else:
        src_name = ''
        tar_name = ''

    src_training = loadPickle(experimentalPath, src_name + '_training.pkl')
    src_validation = loadPickle(experimentalPath, src_name + '_validation.pkl')
    src_test = loadPickle(experimentalPath, src_name + '_test.pkl')

    tar_training = loadPickle(experimentalPath, tar_name + '_' + src_name + '.pkl')
    tar_test = loadPickle(experimentalPath, tar_name + '_test.pkl')

    src_training = normalizeInput(src_training, mode='Paired')
    src_validation = normalizeInput(src_validation, mode='Paired')
    src_test = normalizeInput(src_test, mode='Paired')

    tar_training = normalizeInput(tar_training, mode='Unpaired')
    tar_test = normalizeInput(tar_test, mode='Paired')

    print('source training image shape', str(src_training[0].shape))
    print('source training label shape', src_training[1].shape)
    print('source training image mean/std', str(src_training[0].mean()), str(src_training[0].std()))

    print('source validation image shape', str(src_validation[0].shape))
    print('source validation label shape', src_validation[1].shape)
    print('source validation image mean/std', str(src_validation[0].mean()), str(src_validation[0].std()))

    print('source test image shape', src_test[0].shape)
    print('source test label shape', src_test[1].shape)
    print('source test image mean/std', str(src_test[0].mean()), str(src_test[0].std()))

    print('target training image shape', str(tar_training.shape))
    print('target training image mean/std', str(tar_training.mean()), str(tar_training.std()))

    print('target test image shape', tar_test[0].shape)
    print('target test label shape', tar_test[1].shape)
    print('target test image mean/std', str(tar_test[0].mean()), str(tar_test[0].std()))

    return [src_training, src_validation, src_test], [tar_training, tar_test]


def save2file(message, checkpointPath, model_name):
    if not os.path.isdir(checkpointPath):
        os.makedirs(checkpointPath)
    logfile = open(checkpointPath + model_name + '.txt', 'a+')
    print(message)
    print(message, file=logfile)
    logfile.close()
