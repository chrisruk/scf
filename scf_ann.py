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
import pickle


# Load SCF training data from previously pickled file
load_scf_training = True

# Load SCF testing data from previously pickled file

load_scf_testing = True

# Save SCF data to pickled file
save = False

# Modulation schemes
mod = ["2psk","4psk","8psk","fsk"]

# Load ANN from file 
loadann = True


# Generate a graph of SCF data
def graph(za):
    
    nx, ny = za.shape[1], za.shape[0]
    y = np.arange(0,1.0,1.0/ny)
    x = np.arange(-0.5,0.5,1.0/nx)

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    ha.set_xlabel('Frequency')
    ha.set_ylabel('Alpha')
    ha.set_zlabel('SCF')

    X, Y = numpy.meshgrid(x, y) 

    ha.plot_surface(X, Y, za,rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.show()

# Generate 2D array of SCF data
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

"""
# Graph data

# payload 90
y = np.fromfile("data/train-2/fsk-snr0.dat", dtype=np.complex64)
graph(scf(y))

# payload 189
y = np.fromfile("data/train-3/fsk-snr0.dat", dtype=np.complex64)
graph(scf(y))

"""


# Load dataset of different modulation schemes
def load_data(path):

    out = []
    out_o = []

    count = 0
    for m in mod :
        
        z = np.zeros((len(mod),))
        z[count] = 1

        for i in range(0,9):
        
            y = np.fromfile("%s/%s-snr%d.dat" % (path,m,i), dtype=np.complex64)
            y = np.array_split(y,int(len(y)/(1024*5)))

            c=0
            for q in y:
                out.append(scf(q[0:1024*5]))
                out_o.append(z)
                print("Mod",m,":",i,": ",c)
                c += 1

        count += 1

    return (out,out_o)

train = []
train_out = []

test = []
test_out = []

# Load pickled SCF training data
if load_scf_training:
    train = pickle.load(open('train.dat', 'rb'))
    train_out = pickle.load(open('train_o.dat', 'rb'))
else:
    train,train_out = load_data("data/train-rnd1")
    train2,train_out2 = load_data("data/train-rnd2")

    train = train + train2
    train_out = train_out + train_out2

# Load pickled SCF testing data
if load_scf_testing:
    test = pickle.load(open('test.dat', 'rb'))
    test_out = pickle.load(open('test_o.dat', 'rb'))
else:
    test,test_out = load_data("data/train-rnd3")

# Save SCF data to pickled files
if save:
    with open('train.dat','w') as f:
        pickle.dump(train,f)

    with open('train_o.dat','w') as f:
        pickle.dump(train_out,f)

    with open('test.dat','w') as f:
        pickle.dump(test,f)

    with open('test_o.dat','w') as f:
        pickle.dump(test_out,f)


print("Tensor flow starting")
inputs = train[0].shape[0]*train[0].shape[1]
hidden = int(inputs * (0.89))
print("Inputs ",inputs,"Hidden ", hidden)

# Hack - This could be improved likely, by using a built-in function to Tensor Flow, can't
# seem to find one at the moment
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
    regressor = tflearn.regression(net, optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy') #, loss=lossv)
    m = tflearn.DNN(regressor,tensorboard_verbose=3)


    if loadann:
        m.load('ann.tflearn')
    else:
        m.fit(train, train_out, n_epoch=50, snapshot_epoch=False,show_metric=True)

    # Is there a simple Tflearn evaluation function? 
    c = 0
    correct = 0.0
    for t in test:
        o = []
        for v in m.predict([t])[0]:
            o.append(thresh(v))
        if o == test_out[c].tolist():
            correct += 1.0
        c = c + 1
   
    if not loadann: 
        m.save('ann.tflearn')

    # Print accuracy of classifier when run on test data
    print ((correct/float(c))*100.0,"Number of items",c)
