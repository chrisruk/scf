#!/usr/bin/python2

import matplotlib.pyplot as plot
import numpy as np
import sys
import math
import random
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import collections
import scipy
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from itertools import islice
import tensorflow as tf
import tflearn
from scipy import signal
import matplotlib.pyplot as plt
import glob

def graph(za):
    
    nx, ny = za.shape[1], za.shape[0]
    x = np.arange(0,nx)
    y = np.arange(0,ny)

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    ha.set_xlabel('Frequency')
    ha.set_ylabel('Alpha')
    ha.set_zlabel('SCF')

    X, Y = numpy.meshgrid(x, y) 

    ha.plot_surface(X, Y, za,rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.show()

def scf(datfile):

    za = []
    y = np.fromfile(datfile, dtype=np.complex64)
    y = y[0:1024*50]

    d = collections.deque(maxlen=10)

    N = 1000#3000             # Number of frames
    T = int(len(y) / N) # Frame length
    print("Flen",T)
    Fs = T #*2
    al = 1*Fs
    n = 0

    frame = y[(n*int(T)):int(n*T)+int(T)]
    xf = np.fft.fftshift(np.fft.fft(frame))
    xfp = np.append([0]*int(al/2),xf)
    xfm = np.append(xf,[0]*int(al/2))
    Sxf = (1/T) * xfp * np.conj(xfm)
    Sxf = Sxf * (np.e**-(1j*2*np.pi*al*(N*T)))
    mx = len(Sxf)
    alph = []

    for a in np.arange(0,1,0.1):


        Fs = T #*2
        al = a * Fs
        alph.append(a)
    
        out = []

        count = 0
        for n in range(0,N):

            count = n
            frame = y[int(n*T):int(n*T)+T]

            xf = np.fft.fftshift(np.fft.fft(frame))
            
            xfp = np.append([0]*int(al/2),xf)
            xfm = np.append(xf,[0]*int(al/2))
            np.set_printoptions(threshold=np.nan)
            
            # removed 1/T 
            Sxf =  xfp * np.conj(xfm) 
            Sxf = Sxf * (np.e**-(1j*2*np.pi*al*(count*T)))
    
            orig = len(Sxf)
            Sxf.resize((mx,))
            newsize = len(Sxf)

            Sxf = np.roll(Sxf,int((newsize-orig)/2))

            new = []
            for v in Sxf:
                new.append(math.sqrt(v.imag**2+v.real**2))
        
            out.append(new)
    
        tm = np.mean( np.array(out), axis=0 )
        d.append(tm)

        # mean of columns
        smoothed = np.mean(np.array(d),axis=0)

        za.append(smoothed)
 
    out = np.array(za) 
    o = (out/out.max())
    o[o == np.inf] = 0

    return o

print("Loading data")

train = []
train_out = []

valid = []
valid_out = []

mod = ["2psk","4psk","8psk","fsk"]

count = 0
for m in mod :
    z = np.zeros((len(mod),))
    z[count] = 1

    for i in range(0,9):
        train.append(scf("train/%s-snr%d.dat" % (m,i)))
        train_out.append(z)
        break
    count = count + 1

count = 0
for m in mod :
    z = np.zeros((len(mod),))
    z[count] = 1

    for i in range(0,9):
        valid.append(scf("train-0/%s-snr%d.dat" % (m,i)))
        valid_out.append(z)
    count = count + 1

print("Tensor flow starting")

for i in np.arange(0.5,3,0.05):
    print("i val",i)
    with tf.Graph().as_default():
        tflearn.init_graph(num_cores=4)
        tnorm = tflearn.initializations.uniform(minval=0.0, maxval=1.0)
        net = tflearn.input_data(shape=[None,train[0].shape[0],train[0].shape[1]])
        #net = tflearn.conv_2d(net, 32 ,1, activation='relu')
        #net = tflearn.max_pool_2d(net, 2)
        net = tflearn.fully_connected(net, int(((train[0].shape[1]*train[0].shape[0])+3.0)/i), activation='softmax')
        #net = tflearn.fully_connected(net, int(((train[0].shape[1]*train[0].shape[0])+3.0)/i), activation='softmax')
        #net = tflearn.fully_connected(net, int(gfsk_tr.shape[1]*gfsk_tr.shape[0]*0.6), activation='softmax')
        #net = tflearn.dropout(net, 0.5)
        net = tflearn.fully_connected(net, len(mod), activation='softmax')
        regressor = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
        # Training
        m = tflearn.DNN(regressor,tensorboard_verbose=3)
        m.fit(train, train_out,validation_set = (valid,valid_out), n_epoch=4000, snapshot_epoch=False,show_metric=True) 



