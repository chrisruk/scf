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

def scf(y):

    za = []


    d = collections.deque(maxlen=10)
    y = y[0:1024*5]
    N = 100#3000             # Number of frames
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

test = []
test_out = []

mod = ["2psk","4psk","8psk","fsk"]

count = 0
for m in mod :
    z = np.zeros((len(mod),))
    z[count] = 1

    for i in range(0,9):
    
        y = np.fromfile("data/train/%s-snr%d.dat" % (m,i), dtype=np.complex64)
        y = np.array_split(y,int(len(y)/(1024*5)))

        y2 = np.fromfile("data/train-0/%s-snr%d.dat" % (m,i), dtype=np.complex64)
        y2 = np.array_split(y2,int(len(y2)/(1024*5)))

        c=0
        for q in y:
            train.append(scf(q[0:1024*5]))
            train_out.append(z)
            print("Mod",m,":",i,": ",c)
            c = c + 1

        c=0
        for q in y2:
            test.append(scf(q[0:1024*5]))
            test_out.append(z)
            print("Mod",m,":",i,": ",c)
            c = c + 1

        #graph(train[len(train)-1])
        #break
    count = count + 1
print("out",len(train_out))

"""
train = [val for val in train for _ in (0, 1)]
train_out = [val for val in train_out for _ in (0, 1)]


count = 0
for m in mod :
    z = np.zeros((len(mod),))
    z[count] = 1

    for i in range(0,9):
        valid.append(scf("data/train-0/%s-snr%d.dat" % (m,i)))
        valid_out.append(z)
    count = count + 1
"""

print("Tensor flow starting")

print(train[0].shape)


inputs = train[0].shape[0]*train[0].shape[1]
hidden = int(inputs * (2.0/3.0))
print("Inputs ",inputs,"Hidden ", hidden)


def thresh(i):
    if i >= 0.5:
        return 1
    else: 
        return 0

with tf.Graph().as_default():
    tflearn.init_graph(num_cores=8)
    net = tflearn.input_data(shape=[None,train[0].shape[0],train[0].shape[1]])
    net = tflearn.fully_connected(net, hidden,activation='sigmoid') #, activation='sigmoid')
    net = tflearn.fully_connected(net, len(mod), activation='softmax')
    #sgd = tflearn.SGD(learning_rate=0.001)   
    regressor = tflearn.regression(net, optimizer='adam',loss='categorical_crossentropy') #, loss=lossv)
    m = tflearn.DNN(regressor,tensorboard_verbose=3)
    m.fit(train, train_out, n_epoch=100, snapshot_epoch=False,show_metric=True)

    c = 0
    correct = 0.0
    for t in test:
        o = []
        for v in m.predict([t])[0]:
            o.append(thresh(v))
        if o == test_out[c].tolist():
            correct += 1.0
        c = c + 1

    print ((correct/float(c))*100.0,"Number of items",c)
