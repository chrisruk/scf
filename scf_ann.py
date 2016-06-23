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


def scf(datfile):

    za = []
    y = np.fromfile(datfile, dtype=np.complex64)
    y = y[0:1024*50]

    d = collections.deque(maxlen=10)

    N = 100             # Number of frames
    T = int(len(y) / N) # Frame length

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
            #Sxf = Sxf #* (np.e**-(1j*2*np.pi*al*(count*T)))
    
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
        break 
    out = np.array(np.array(za).flatten())
    o = (out/out.max())
    o2 = []
    for v in o:
        if np.isinf(v):
            o2.append(0)
        else:
            o2.append(v)
    return np.array(o2)

gfsk_tr = scf("/tmp/gfsk_tr.dat")
psk_tr = scf("/tmp/psk_tr.dat")

gfsk_te = scf("/tmp/gfsk_te.dat")
psk_te = scf("/tmp/psk_te.dat")


X = [gfsk_tr,psk_tr]
Y_xor = [[1.0,0.0],[0.,1.0]]

# Graph definition
with tf.Graph().as_default():
    tnorm = tflearn.initializations.uniform(minval=0.0, maxval=1.0)
    net = tflearn.input_data(shape=[None,gfsk_tr.shape[0]])
    net = tflearn.fully_connected(net, int(gfsk_tr.shape[0]), activation='softmax', weights_init=tnorm)
    net = tflearn.fully_connected(net, 2, activation='softmax', weights_init=tnorm)
    regressor = tflearn.regression(net, optimizer='adam', learning_rate=0.2, loss='softmax_categorical_crossentropy')

    # Training
    m = tflearn.DNN(regressor)
    m.fit(X, Y_xor, n_epoch=10000, snapshot_epoch=False) 

    # Testing
    print("Testing XOR operator")
    print("0 xor 0:", m.predict([gfsk_te]))
    print("0 xor 1:", m.predict([psk_te]))
"""
za = np.array(za)
nx, ny = za.shape[1], za.shape[0]
x = np.arange(0.5,-0.5,-1/mx)
y = alph

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')

ha.set_xlabel('Frequency')
ha.set_ylabel('Alpha')
ha.set_zlabel('SCF')

X, Y = numpy.meshgrid(x, y) 

ha.plot_surface(X, Y, za,rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()
quit()

print("Shape ",np.array(za).shape)
plot.imshow(za,aspect='auto' ,cmap='hot')
plot.show()

myplot[1].plot(frq,abs(Y),'r') # plotting the spectrum
myplot[1].set_xlabel('Freq (Hz)')
myplot[1].set_ylabel('|Y(freq)|')

plt.show()
"""

