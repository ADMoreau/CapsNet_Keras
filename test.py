#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle

TRAIN_FILE = "CAPS_1-6train.p"
VALID_FILE = "CAPS_1-6eval.p"
TEST_FILE = "CAPS_1-6test.p"

def readfile(file, subject):
    temp_x = []
    temp_y = []
    with open(file, mode='rb') as f:
        x = 0
        #while True:
        for i in range(7000):
            try:
                temp = list(pickle.load(f))
                m = [*zip(*temp[0])]
                if temp[1] == subject:
                    temp_x.append(m)
                    temp_y.append(temp[2])
                    x += 1
                    print(x)
                else:
                    pickle.load(f)
              
            except EOFError:
                break
        #temp_x = np.array(temp_x)
        #temp_y = np.array(temp_y)
        f.close()
    return temp_x, temp_y

def load_data(subject):
    path = os.path.dirname(os.path.abspath(__file__))
    trainfile = os.path.join(path + '/data/', TRAIN_FILE)
    testfile = os.path.join(path + '/data/', TEST_FILE)
    #evalfile = os.path.join(path + '/data/', VALID_FILE)
    trX, trY = readfile(trainfile, subject)
    teX, teY = readfile(testfile, subject)
    #evel, eval_length = readfile(evalfile, subject)
    teX = np.array(teX)
    teY = np.array(teY)
    return (trX, trY), (teX, teY)
'''
def train_generator(x, y, batch_size, shift_fraction=0.):
    train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                       height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
    generator = train_datagen.flow(x, y, batch_size=batch_size)
    while 1:	
        x_batch, y_batch = generator.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])
'''
(tx, ty), (tex, tey) = load_data('S1')
t = []
for i in range(20):
    t.append(tx[i])
t = np.array(t).reshape(-1,250,62,1)
print(tex.shape, tey.shape)
