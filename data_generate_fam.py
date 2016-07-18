#!/usr/bin/env python2
from gnuradio import gr
from gnuradio import audio, analog
from gnuradio import digital
from gnuradio import blocks
from grc_gnuradio import blks2 as grc_blks2
import threading
import time
import numpy
import struct
import numpy as np
import tensorflow as tf   
import specest 
from tensor import *
import matplotlib.pylab as plt
import tflearn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tensor import *

snr = ["20","15","10","5","0","-5","-10","-15","-20"] 
snrv = [[1,0.32],[1,0.435],[1,0.56],[1,0.75],[1,1],[0.75,1],[0.56,1],[0.435,1],[0.32,1]]
mod = ["2psk","fsk","qam16"]

Np = 100 # 2xNp is the number of columns
P = 1000 # number of new items needed to calculate estimate
L = 2

np.set_printoptions(threshold=np.nan)

class fam_generate(gr.top_block):
    def __init__(self,modulation,snr):

        self.samp_rate = samp_rate = 32000
        gr.top_block.__init__(self)


        if modulation == "2psk":
            self.digital_mod = digital.psk.psk_mod(
                constellation_points=2,
                mod_code="gray",
                differential=True,
                samples_per_symbol=5,
                excess_bw=0.35,
                verbose=False,
                log=False,
            )
        elif modulation == "4psk":
            self.digital_mod = digital.psk.psk_mod(
                constellation_points=4,
                mod_code="gray",
                differential=True,
                samples_per_symbol=5,
                excess_bw=0.35,
                verbose=False,
                log=False,
            )
        elif modulation == "8psk":
            self.digital_mod = digital.psk.psk_mod(
                constellation_points=4,
                mod_code="gray",
                differential=True,
                samples_per_symbol=5,
                excess_bw=0.35,
                verbose=False,
                log=False,
            )

        elif modulation == "fsk":
            self.digital_mod = digital.gfsk_mod(
        	    samples_per_symbol=5,
        	    sensitivity=1.0,
        	    bt=0.35,
        	    verbose=False,
        	    log=False,
            )
        elif modulation == "qam16":
            self.digital_mod = digital.qam.qam_mod(
                constellation_points=16,
                mod_code="gray",
                differential=True,
                samples_per_symbol=5,
                excess_bw=0.35,
                verbose=False,
                log=False,
            )

        self.sink = blocks.vector_sink_f(2*Np) 
        self.sink = blocks.null_sink(gr.sizeof_float * 2 * Np)    
        self.blocks_add_xx_1 = blocks.add_vcc(1)
        self.specest_cyclo_fam_1 = specest.cyclo_fam(Np, P, L)
        self.blocks_multiply_const_vxx_3 = blocks.multiply_const_vcc((snrv[snr][0], ))
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 1024)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_float*1)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, snrv[snr][1], 0)
        self.analog_random_source_x_0 = blocks.vector_source_b(map(int, numpy.random.randint(0, 256, 2000000)), False)
        self.msgq_out = blocks_message_sink_0_msgq_out = gr.msg_queue(1) 
        self.blocks_message_sink_0 = blocks.message_sink(gr.sizeof_float*2*Np, blocks_message_sink_0_msgq_out, False)   
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_1, 1))    
        self.connect((self.blocks_multiply_const_vxx_3, 0), (self.blocks_add_xx_1, 0))    
        self.connect((self.analog_random_source_x_0, 0), (self.digital_mod, 0)) 
        self.connect((self.digital_mod, 0), (self.blocks_throttle_0, 0))    
        self.connect((self.blocks_throttle_0, 0),(self.blocks_multiply_const_vxx_3, 0))    
        self.connect((self.blocks_add_xx_1, 0),(self.specest_cyclo_fam_1, 0))    
        self.connect((self.specest_cyclo_fam_1,0),(self.sink,0))


def getdata(sn):
    mcount = 0
    inp = [[] for k in range(0,sn)]
    out = [[] for k in range(0,sn)]

    for m in mod:

        z = np.zeros((len(mod),))
        z[mcount] = 1    
    
        for snr in range(0,sn):

            tb = fam_generate(m,snr)
            tb.start()
            time.sleep(1)
            count = 0
            fin = False
            old = None

            while True: 
                floats = np.array(tb.specest_cyclo_fam_1.get_estimate())

                if old == None:
                    old = floats
                else:
                    if (floats == old).all():
                        continue        
                count = count + 1  
                        
                za = floats
                nx, ny = za.shape[1], za.shape[0]
                x = np.arange(nx)
                y = np.arange(ny)

                """
                 Generate 3D graph
                 hf = plt.figure()
                 ha = hf.add_subplot(111, projection='3d')
                 ha.set_xlabel('Frequency')
                 ha.set_ylabel('Alpha')
                 ha.set_zlabel('SCF')
                 X, Y = numpy.meshgrid(x, y)
                 ha.plot_surface(X, Y, za,rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
                 plt.show()
                """

                xx = []
                dat = []
                i = 0
                for v in za:
                    dat.append(v[np.argmax(v)])
                    xx.append(i)
                    i+=1

                            
                hf = plt.figure()
                ha = hf.add_subplot(111)
                ha.set_xlabel('Alpha')
                ha.set_ylabel('Magnitude')
                ha.set_title(m)
                ha.plot(xx,dat)
                hf.savefig('/tmp/%s.png'%m)   # save the figure to file
                plt.close(hf)

                inp[snr].append(np.array(dat))
                out[snr].append(np.array(z))

                if count > 20:
                    break

                old = floats
                mcount += 1

    return np.array(inp),np.array(out)


if __name__ == '__main__':

    mcount = 0
    SNVAL = 4

    test_i , test_o = getdata(9)
    train_i , train_o = getdata(3)
