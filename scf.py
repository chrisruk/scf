#!/usr/bin/python3
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

za = []
y = np.fromfile("/tmp/out.dat", dtype=np.complex64)
y = y[0:10000]

d = collections.deque(maxlen=1)

FFTsize = 200
N = int(len(y)/FFTsize)
T = int(len(y) / N) # Frame length

Fs = T #*2
al = 1*Fs
n = 0
mx = 0
alph = []
print(T)

for a in np.arange(0,int(0.4*T)):

    areal = []
    aimag = []
    
    Fs = T #*2
    alph.append(a)

    print("Alph",a)
    
    out = []

    count = 0
    for n in range(0,N):

        count = n
        frame = y[int(n*T):int(n*T)+T]
        xf = np.fft.fftshift(np.fft.fft(frame))
        xfp = np.roll(xf,-a)
        xfm = np.roll(xf,a)
        Sxf = xfp * np.conj(xfm) 
        mx = len(Sxf)

        oreal = []
        oimag = []
        
        for z in range(0,a):
            Sxf[z] = 0
            Sxf[len(Sxf)-z-1]=0  

        for v in Sxf:
            oreal.append(v.real)
            oimag.append(v.imag)

        areal.append(oreal)
        aimag.append(oimag) 

    tm1 = np.mean( np.array(areal), axis=0 )
    tm2 = np.mean( np.array(aimag), axis=0 )
    tm3 = tm1 + (1j * tm2)
    
    tm = []
    for v in tm3:
        mag = math.sqrt(v.imag**2+v.real**2)
        tm.append(mag)

    za.append(tm)

za = np.array(za)
print (za.shape)
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

