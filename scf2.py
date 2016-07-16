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

alldata = np.fromfile("data-20samp/a/8psk-snr0.dat", dtype=np.complex64)
#alldata = np.fromfile("/tmp/weird.dat", dtype=np.complex64)

#y = np.fromfile("/tmp/ofdm_32kS_10_dB.dat", dtype=np.complex64)

SIGLEN = 20000




def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)
 
def scf(y):

    za = []

    FFTsize = 200
    N = int(len(y)/FFTsize)
    T = int(len(y) / N) # Frame length

    print(T)

    Fs = T 
    al = Fs
    n = 0
    frame = y[(n*int(T)):int(n*T)+int(T)]
    xf = np.fft.fftshift(np.fft.fft(frame))
    #xf = np.fft.fft(frame)

    xfp = np.append([0]*int(al/2),xf)
    xfm = np.append(xf,[0]*int(al/2))
    Sxf = xfp * np.conj(xfm)
    mx = len(Sxf)
    alph = []

   
    for a in np.arange(0.9,1,0.005):
        Fs = T 
        al = a * (Fs)
        alph.append(a)
        print("A",a)
    
        areal = []
        aimag = []
        anew = []


        ft = [] 
        oreal = []
        oimag = []
        new = []

        """
        for n in range(0,N):
            ov = 0
            if n > 0:
                ov = int(((1./100.)*50)*FFTsize)
            frame = y[int(n*T)-ov:(int(n*T)+T)-ov]
            ft = np.fft.fftshift(np.fft.fft(frame))

            oreal = []
            oimag = []

            for v in ft:
                oreal.append(v.real)
                oimag.append(v.imag)

            areal.append(oreal)
            aimag.append(oimag)
            

        tm1 = np.mean( np.array(areal), axis=0 )
        tm2 = np.mean( np.array(aimag), axis=0 )
        tmnew = np.mean( np.array(anew), axis=0 )

        tm3 = tm1 + (1j * tm2)
    

        xf = tm3
        xfp = np.append([0]*int(al/2),xf)
        xfm = np.append(xf,[0]*int(al/2))


        np.set_printoptions(threshold=np.nan)
        
        Sxf = xfp * np.conj(xfm)

        orig = len(Sxf)
        Sxf.resize((mx,))
        newsize = len(Sxf)

        Sxf = np.roll(Sxf,int((newsize-orig)/2))
        
        tm = []
        for v in Sxf:
            tm.append(math.sqrt(v.imag**2+v.real**2))
        """
        

            
        

 
        for n in np.arange(0,N,1):
        #for frame in window(y,FFTsize):
            ov = 0
            #if n > 0:
            #    ov = int(((1./100.)*50)*FFTsize)
            frame = y[int(n*T)-ov:(int(n*T)+T)-ov]
            xf = np.fft.fftshift(np.fft.fft(frame))
           
            print(xf)
            quit() 
            #xf = np.fft.fft(frame)

            xfp = np.append([0]*int(al/2),xf)
            xfm = np.append(xf,[0]*int(al/2))
            np.set_printoptions(threshold=np.nan)
        
    
            Sxf = (xfp * np.conj(xfm)) 
            
            orig = len(Sxf)
            Sxf.resize((mx,))
            newsize = len(Sxf)

            Sxf = np.roll(Sxf,int((newsize-orig)/2))
 
            oreal = []
            oimag = []

            for v in Sxf:
                oreal.append(v.real)
                oimag.append(v.imag)
        

            areal.append(oreal)
            aimag.append(oimag)
          

        print(areal[0],"\n")
        print(areal[1],"\n")

        a = np.array(areal[0]) 
        b = np.array(areal[1])

        for v in a:
            if not v == 0.:
                print (v)
                break


        for v in b:
            if not v == 0.:
                print (v)
                break


        quit() 
         

        tm1 = np.mean( np.array(areal), axis=0 )
        tm2 = np.mean( np.array(aimag), axis=0 )

        tm3 = tm1 + (1j * tm2)

        tm = []
        for v in tm3:
            mag = math.sqrt(v.imag**2+v.real**2)
            tm.append(mag)

        



        #tm = out[len(out)-1]

        #tm2 = []    
        #ff = collections.deque(maxlen=2)
        #for v in tm:
        #    ff.append(v)
        #    ffo = np.mean(ff,axis=0)
        #    tm2.append(ffo)

        alph.append(a)
        za.append(tm)
       
    """ 
    za = np.array(za) 
    #za = (za/za.mean())
    #za[za == np.inf] = 0
    
    nx, ny = za.shape[1], za.shape[0]
    
    print(nx,ny)
    x = np.arange(nx)

    print(len(x))
    y = np.arange(ny)


    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    ha.set_xlabel('Frequency')
    ha.set_ylabel('Alpha')
    ha.set_zlabel('SCF')
    X, Y = numpy.meshgrid(x, y)

    ha.plot_surface(X, Y, za,rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.show()
    """










    za = np.array(za)




    dat = []
    x = []
    i = 0
    for zz in range(za.shape[0]):
        dat.append(za[zz][np.argmax(za[zz])])
        x.append(i)
        i += 1

    dat = np.array(dat)
    #dat /= dat.mean() 

    




    return x,dat,za,mx,alph

allv = np.array_split(alldata,int(len(alldata)/(SIGLEN)))

c = 0
for z in allv:
    x,dat,za,mx,alph = scf(z)
    hf = plt.figure()
    ha = hf.add_subplot(111)

    ha.set_xlabel('Alpha')
    ha.set_ylabel('Magnitude')

    ha.plot(x,dat)
    hf.savefig('/tmp/%d.png' % c)
    c += 1

print("Shape ",np.array(za).shape)
plot.imshow(za,aspect='auto' ,cmap='hot')
plot.show()

myplot[1].plot(frq,abs(Y),'r') # plotting the spectrum
myplot[1].set_xlabel('Freq (Hz)')
myplot[1].set_ylabel('|Y(freq)|')

plt.show()

