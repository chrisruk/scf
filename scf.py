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
from itertools import tee  
za = []
y = np.fromfile("/tmp/out.dat", dtype=np.complex64)
y = y[0:5000]

d = collections.deque(maxlen=1)

FFTsize = 100
T = FFTsize
alph = []

def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)

for a in np.arange(0,100,1):


    alpha = a*(1/T)

    # when a == 1, alpha = 2*1/N
    print (alpha)




    areal = []
    aimag = []
    
    Fs = T #*2
    alph.append(a)

    print("Alph",a)
    out = []

    count = 0
    nth = 0
    sxfs = []
    ffts = []

    for frame in window(y,FFTsize):
        ffts.append(frame)

    blocks = len(y)//T
    #for i in range(blocks):
    #   ffts.append( y[int(i*T):int(i*T)+T] )

    for frame in ffts:
        xf = np.array(np.fft.fftshift(np.fft.fft(frame)))

        # Smooth periodogram
        x2 = []
        for v in xf:
            x2.append((1.0/float(len(xf))) * np.abs(v)**2   )

        xf = np.array(x2)
        xfp = np.roll(xf,-a)
        xfm = np.roll(xf,a)

        Ia = (1/float(T)) * xfp * np.conj(xfm)
        expv = np.exp(-1j*2.0*np.pi*alpha*nth) #*(alpha)*j*T)
        print("Val ",expv,"alpha: ",alpha)
        Sxf = Ia*expv        
        sxfs.append(Sxf)

        reals = []
        imags = []
        for v in Sxf:
            reals.append(v.real)
            imags.append(v.imag)

        areal.append(reals)
        aimag.append(imags)

        nth+=1.0

    sx = np.mean(areal,axis=0) + 1j*np.mean(aimag,axis=0)
    #sx = np.mean(sxfs,axis=0)
    """
    for z in range(0,a):
        sx[z] = 0
        sx[len(Sxf)-z-1]=0  
    """
    
 
    tm = []
    for v in sx:
        mag = math.sqrt(v.imag**2+v.real**2)
        tm.append(mag)


    za.append(tm)

#za = np.array([[10,10,10]])

za = np.array(za)

print (za.shape)
nx, ny = za.shape[1], za.shape[0]
#x = np.arange(0.5,-0.5,-1/nx)
#y = np.arange(0,0.01,0.1/ny)

x = np.arange(nx)
y = np.arange(ny)

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')

ha.set_xlabel('Frequency')
ha.set_ylabel('Alpha')
ha.set_zlabel('SCF')

print("here")
X, Y = numpy.meshgrid(x, y) 
print("here2")
ha.plot_surface(X, Y, za,rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
print("here3")
plt.show()


"""

print("Shape ",np.array(za).shape)
plot.imshow(za,aspect='auto' ,cmap='hot')
plot.show()

myplot[1].plot(frq,abs(Y),'r') # plotting the spectrum
myplot[1].set_xlabel('Freq (Hz)')
myplot[1].set_ylabel('|Y(freq)|')

plt.show()

"""

