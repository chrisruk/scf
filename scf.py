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

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result





za = []
#y = np.fromfile("/tmp/out.dat", dtype=np.complex64)
y = np.fromfile("/tmp/out2.dat", dtype=np.float32)
plt.plot(y)
plt.show()
#quit()
d = collections.deque(maxlen=5)

# Number of frames
N = 140
T = int(len(y) / N)
print("Framelen ",T)

Fs = T
al = 1*(Fs*2)
n = 0

frame = y[(n*int(T)):int(n*T)+int(T)]
xf = np.fft.fftshift(np.fft.fft(frame))
xfp = np.append([0]*int(al/2),xf)
xfm = np.append(xf,[0]*int(al/2))
Sxf = (1/T) * xfp * np.conj(xfm)
Sxf = Sxf * (np.e**-(1j*2*np.pi*al*(N*T)))

mx = len(Sxf)

print("Mx",mx)


alph = []

for a in np.arange(0,1,0.05):


    Fs = T #*2
    al = a * (Fs*2)
    alph.append(a)
    print("Alph",al)
    
    T = int(len(y) / N)
    print("Tlen",T)

    out = []

    count = 0
    for n in range(0,N):
    #for frame in window(y,T):
        count = n
        #print("v",count/len(y),len(frame))
        frame = y[int(n*T):int(n*T)+T]
        print("Frame ",int(n*T),int(n*T)+T)
        xf = np.fft.fftshift(np.fft.fft(frame))
        xfp = np.append([0]*int(al/2),xf)
        xfm = np.append(xf,[0]*int(al/2))
        np.set_printoptions(threshold=np.nan)
        
        Sxf = ((1/T) * xfp * np.conj(xfm)) * (np.e**-(1j*2*np.pi*al*(count*T)))
    
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

