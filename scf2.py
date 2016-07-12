#!/usr/bin/python2.7
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
from itertools import tee
za = []
y = np.fromfile("/tmp/out.dat", dtype=np.complex64)
y = y[0:1024*20]

d = collections.deque(maxlen=3)

FFTsize = 1024*3
N = int(len(y)/FFTsize)
T = int(len(y) / N) # Frame length

print(N)

Fs = T #*2
al = 1*Fs
n = 0

frame = y[(n*int(T)):int(n*T)+int(T)]
xf = np.fft.fftshift(np.fft.fft(frame))
xfp = np.append([0]*int(al/2),xf)
xfm = np.append(xf,[0]*int(al/2))
Sxf = xfp * np.conj(xfm)
mx = len(Sxf)
alph = []

def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)
 

for a in np.arange(0,1,0.05):

    
    Fs = T #*2
    al = a * Fs
    alph.append(a)
    print("A",a)
    
    out = []

    count = 0
    for n in range(0,N):
    #for frame in window(y,FFTsize):
    
        count = n
        frame = y[int(n*T):int(n*T)+T]
        xf = np.fft.fftshift(np.fft.fft(frame))
        xfp = np.append([0]*int(al/2),xf)
        xfm = np.append(xf,[0]*int(al/2))
        np.set_printoptions(threshold=np.nan)
        
        #Sxf = (1/T) * xfp * np.conj(xfm)
        Sxf = xfp * np.conj(xfm) 
 
        #Sxf = Sxf * (np.e**-(1j*2*np.pi*al*(count*T)))
    
        orig = len(Sxf)
        Sxf.resize((mx,))
        newsize = len(Sxf)

        Sxf = np.roll(Sxf,int((newsize-orig)/2))

        new = []
        for v in Sxf:
            new.append(math.sqrt(v.imag**2+v.real**2))
    
        out.append(new)
    
    tm = np.mean( np.array(out), axis=0 )

    tm2 = []    
    ff = collections.deque(maxlen=20)
    for v in tm:
        ff.append(v)
        ffo = np.mean(ff,axis=0)
        tm2.append(ffo)





    d.append(tm2)



    # mean of columns
    smoothed = np.mean(np.array(d),axis=0)
    za.append(smoothed)
        

za = np.array(za)
nx, ny = za.shape[1], za.shape[0]

x = np.arange(0.5,-0.5,-1/float(mx))
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

