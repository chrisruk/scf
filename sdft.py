#!/usr/bin/python3
import math
import numpy as np

a = [ 1, 2, 3, 4, 5]
b = [ 2, 3, 4, 5, 6]

afft = np.fft.fft(a)
bfft = np.fft.fft(b)

s = [(afft[k]-a[0]+b[len(b)-1])*np.exp(1j*2*math.pi*k/len(a)) for k in range(len(a))]

print(afft)
print(bfft)
print (s)
