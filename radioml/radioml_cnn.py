#!/usr/bin/env python2
from __future__ import division, print_function, absolute_import

from gnuradio import gr
from gnuradio import audio, analog
from gnuradio import digital
from gnuradio import blocks
from grc_gnuradio import blks2 as grc_blks2
import threading
import time
import numpy
import struct
import numpy as np
import tensorflow as tf   
import specest 
from tensor import *
import matplotlib.pylab as plt
import tflearn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import cPickle
import time
from tensor import *
from numpy import zeros, newaxis

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression



def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]


radioml = cPickle.load(open("2016.04C.multisnr.pkl",'rb'))

data = {}
allm = []

for k in radioml.keys():
    data[k[0]] = {}
    allm.append(k[0])
mod = sorted(set(allm))

for m in mod:
    dat = []
    for k in radioml.keys():
        if k[0] == m and k[1] == 18:
            for sig in range(len(radioml[k])):
    
                a = numpy.array(radioml[k][sig][0])[:, newaxis]
                b = numpy.array(radioml[k][sig][1])[:, newaxis]

                if 18 not in data[k[0]]:
                    data[k[0]][18] = []

                data[k[0]][18].append([a,b])

                
            

print (data["WBFM"][18][1])



# Convolutional network building

X = []
Y = []



bpsk = data["BPSK"][18]
wbfm = data["WBFM"][18]



for v in bpsk[:len(bpsk)//2]:
    X.append(v)
    Y.append([1.,0.])


for v in wbfm[:len(wbfm)//2]:
    X.append(v)
    Y.append([0.,1.])

x = []
y = []

for v in bpsk[len(bpsk)//2:]:
    x.append(v)
    y.append([1.,0.])


for v in wbfm[len(wbfm)//2:]:
    x.append(v)
    y.append([0.,1.])



network = input_data(shape=[None, 2, 128,1])
print(tflearn.utils.get_incoming_shape(network))




network = conv_2d(network, 64,[1,3], activation='relu')
network = conv_2d(network, 16,[2,3], activation='relu')
network = fully_connected(network, 128, activation='relu')
network = fully_connected(network, 2, activation='softmax')

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=50, shuffle=True,show_metric=True, batch_size=96, run_id='cifar10_cnn')

z = 0
gd = 0

for v in x:
    if np.argmax(model.predict ( [ v ])[0]) == np.argmax(y[z]):
        gd += 1

    z = z + 1
    
print ("ACC",gd/z)




