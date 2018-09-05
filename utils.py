import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import pickle
import os
from keras.utils import to_categorical

TRAIN_FILE = "CAPS_1-6train.p"
VALID_FILE = "CAPS_1-6eval.p"
TEST_FILE = "CAPS_1-6test.p"

def readfile(file, subject):
    temp_x = []
    temp_y = []
    with open(file, mode='rb') as f:
        x = 0
        while True:
        #for i in range(100):
            try:
                temp = list(pickle.load(f))
                m = [*zip(*temp[0])]
                if temp[1] == subject:
                    temp_x.append(m)
                    temp_y.append(temp[2])
                    x += 1
                else:
                    pickle.load(f)

            except EOFError:
                break
        #temp_x = np.asarray(temp_x)
        #temp_y = np.asarray(temp_y)
        f.close()
    #print(temp_x.shape, temp_y.shape)
    return temp_x, temp_y

def load_EEG(subject):
    path = os.path.dirname(os.path.abspath(__file__))
    trainfile = os.path.join(path + '/data/', TRAIN_FILE)
    testfile = os.path.join(path + '/data/', TEST_FILE)
    #evalfile = os.path.join(path + '/data/', VALID_FILE)
    trX, trY = readfile(trainfile, subject)
    teX, teY = readfile(testfile, subject)
    #x_train = trX.reshape(tr, 250, 62, 1).astype('float32')
    #x_test = teX.reshape(te, 250, 62, 1).astype('float32')
    #x_train = trX[:,:,:,np.newaxis]
    #x_test = teX[:,:,:,np.newaxis]
    #y_train = to_categorical(trY.astype('float32'))
    #y_test = to_categorical(teY.astype('float32'))
    teX = np.array(teX).reshape(-1, 250, 62, 1)
    teY = np.array(teY)
    teY = to_categorical(teY, num_classes=10)
    #return (x_train, y_train), (x_test, y_test)
    return (trX, trY), (teX, teY)

def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:,0] += 1

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

if __name__=="__main__":
    plot_log('result/log.csv')
