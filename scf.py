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

Fs = 32000;  # sampling rate
Ts = 1.0/Fs; # sampling interval


za = []
y = np.fromfile("/tmp/out.dat", dtype=np.complex64)

d = collections.deque(maxlen=5)

# Number of frames
N = 100

al = 1*Fs
n = 0

T = int(len(y) / N)
frame = y[(n*int(T)):int(n*T)+int(T)]
xf = np.fft.fftshift(np.fft.fft(frame))
xfp = np.append([0]*int(al/2),xf)
xfm = np.append(xf,[0]*int(al/2))
Sxf = (1/T) * xfp * np.conj(xfm)

mx = len(Sxf)

print("Mx",mx)


alph = []

for a in range(0,20,1):
    al = (a/20.0) * Fs
    alph.append(a/20.0)
    print("Alph",al)
    
    T = int(len(y) / N)
    print("Tlen",T)

    out = []
    for n in range(0,N):
    #for frame in window(y,T):
        frame = y[(n*int(T)):int(n*T)+T]
        xf = np.fft.fftshift(np.fft.fft(frame))


        xfp = np.append([0]*int(al/2),xf)
        xfm = np.append(xf,[0]*int(al/2))
        np.set_printoptions(threshold=np.nan)
        
        Sxf = (1/T) * xfp * np.conj(xfm)
        #Sxf = xfp * np.conj(xfm)
    
        orig = len(Sxf)
        Sxf.resize((mx,))
        newsize = len(Sxf)

        Sxf = np.roll(Sxf,int((newsize-orig)/2))

        new = []
        for v in Sxf:
            new.append(math.sqrt(v.imag**2+v.real**2))
    
        out.append(new)
    #tm = out[0]
    tm = np.mean( np.array(out), axis=0 )
    d.append(tm)

    # mean of columns
    smoothed = np.mean(np.array(d),axis=0)
    za.append(smoothed)
        
# Set up grid and test data

za = np.array(za)
nx, ny = za.shape[1], za.shape[0]
x = range(nx)
y = alph
#print("X",x)
#y = range(ny)

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')

ha.set_xlabel('Frequency')
ha.set_ylabel('Alpha')
ha.set_zlabel('SCF (Need to normalise)')

X, Y = numpy.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D

ha.plot_surface(X, Y, za,rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#ha.plot_wireframe(X,Y,za)
plt.show()

quit()



print("Shape ",np.array(za).shape)
plot.imshow(za,aspect='auto' ,cmap='hot')
plot.show()

myplot[1].plot(frq,abs(Y),'r') # plotting the spectrum
myplot[1].set_xlabel('Freq (Hz)')
myplot[1].set_ylabel('|Y(freq)|')

plt.show()

