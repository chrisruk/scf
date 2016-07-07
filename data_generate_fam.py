#!/usr/bin/env python2
from gnuradio import gr
from gnuradio import audio, analog
from gnuradio import digital
from gnuradio import blocks
from grc_gnuradio import blks2 as grc_blks2
import inspector
import threading
import time
import numpy
import specest
import struct
import numpy as np
import tensorflow as tf    
from tensor import *
from specest import fam_matplotlib
import matplotlib.pylab as plt

snr = ["20","15","10","5","0","-5","-10","-15","-20"] 
snrv = [[1,0.32],[1,0.435],[1,0.56],[1,0.75],[1,1],[0.75,1],[0.56,1],[0.435,1],[0.32,1]]
mod = ["2psk","4psk","8psk","fsk"]

Np = 4
P = 4
L = 2

train = True

input_num = 128

np.set_printoptions(threshold=np.nan)



if train == True:
    n_input = tf.placeholder(tf.float32, shape=[None, input_num], name="inp")
    n_output = tf.placeholder(tf.float32, shape=[None, len(mod)], name="outp")

    hidden_nodes = int( .89 * (input_num) )
    b_hidden = tf.Variable(tf.random_normal([hidden_nodes]), name="hidden_bias")
    W_hidden = tf.Variable(tf.random_normal([input_num, hidden_nodes]), name="hidden_weights")

    # calc hidden layer's activation
    hidden = tf.sigmoid(tf.matmul(n_input, W_hidden) + b_hidden)
    W_output = tf.Variable(tf.random_normal([hidden_nodes, len(mod)]), name="output_weights")  # output layer's weight matrix
    output = tf.sigmoid(tf.matmul(hidden, W_output),name="out")  # calc output layer's activation
    #cross_entropy = -(n_output * tf.log(output) + (1 - n_output) * tf.log(1 - output))

    y_pred = tf.nn.relu(output)
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_pred, output)

    #cross_entropy = -(n_output * tf.log(output) + (1 - n_output) * tf.log(1 - output))

    cross_entropy = -tf.reduce_sum(output*tf.log(tf.clip_by_value(n_output,1e-10,1.0)))
    #cross_entropy = tf.square(n_output - output)
    # cross_entropy = tf.square(n_output - output)  # simpler, but also works
    loss = tf.reduce_mean(cross_entropy)  # mean the cross_entropy
    optimizer = tf.train.AdamOptimizer(0.01)  # take a gradient descent for optimizing with a "stepsize" of 0.1
    train = optimizer.minimize(loss)  # let the optimizer train
    
    init = tf.initialize_all_variables()
    
    sess = tf.Session()  # create the session and therefore the graph
    sess.run(init)  # initialize all variables  































class my_top_block(gr.top_block):
    def __init__(self,modulation,snr):

        self.samp_rate = samp_rate = 32000
        gr.top_block.__init__(self)


        if modulation == "2psk":
            self.digital_mod = digital.psk.psk_mod(
                constellation_points=2,
                mod_code="gray",
                differential=True,
                samples_per_symbol=2,
                excess_bw=0.35,
                verbose=False,
                log=False,
            )
            bits = 1
        elif modulation == "4psk":
            self.digital_mod = digital.psk.psk_mod(
                constellation_points=4,
                mod_code="gray",
                differential=True,
                samples_per_symbol=2,
                excess_bw=0.35,
                verbose=False,
                log=False,
            )
            bits = 2
        elif modulation == "8psk":
            self.digital_mod = digital.psk.psk_mod(
                constellation_points=8,
                mod_code="gray",
                differential=True,
                samples_per_symbol=2,
                excess_bw=0.35,
                verbose=False,
                log=False,
            )
            bits = 3
        elif modulation == "fsk":
            self.digital_mod = digital.gfsk_mod(
        	    samples_per_symbol=2,
        	    sensitivity=1.0,
        	    bt=0.35,
        	    verbose=False,
        	    log=False,
            )
            bits = 1

        self.sink = blocks.vector_sink_f(2*Np) 
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
        #self.connect((self.digital_mod, 0), (self.blocks_throttle_0, 0))  
        self.connect((self.digital_mod, 0), (self.blocks_throttle_0, 0))    
        self.connect((self.blocks_throttle_0, 0),(self.blocks_multiply_const_vxx_3, 0))    

        #self.connect((self.digital_mod, 0), (self.blocks_multiply_const_vxx_3, 0))    
        # self.connect((self.blocks_throttle_0, 0),(self.blocks_multiply_const_vxx_3, 0))    
        self.connect((self.blocks_add_xx_1, 0),(self.specest_cyclo_fam_1, 0))    
        #self.connect((self.specest_cyclo_fam_1,0),(self.blocks_message_sink_0,0))


        #self.connect((self.specest_cyclo_fam_1, 0), (self.blocks_vector_to_stream_0, 0))
        #self.connect((self.blocks_stream_to_vector_0, 0), (self.inspector_TFModel_0, 0))   
        #self.msg_connect((self.inspector_TFModel_0, 'classification'), (self.blocks_message_debug_0, 'print'))         
        self.connect((self.specest_cyclo_fam_1,0),(self.sink,0))





from tensor import *























if __name__ == '__main__':

    try:
        inp = []
        out = [] 

        mcount = 0

        for m in mod:

            z = np.zeros((len(mod),))
            z[mcount] = 1        

            print(z)
            
            for snr in range(0,1):
                print(m,"SNR",snr)
                tb = my_top_block(m,snr)
                tb.start()

                time.sleep(2)

                count = 0
                fin = False
                while True: 
                    #data=tb.msgq_out.delete_head().to_string() # this indeed blocks
                    #data = np.array(tb.specest_cyclo_fam_1.get_estimate())
                    
                    ## Get last bytes
                    floats =  tb.sink.data()#[-2*P*L*(2*Np):] 
                    print(len(floats)/128)
                    #data = np.asarray(estimated_data)
                   
    
                 
                    """ 
                    print("SH",data.shape)
    
                    inp.append(data)
                    out.append(z)
                    count += 1
                    #if count > 5:
    
                    if count > 10:
                        tb.stop()
                        break
                    print("COUNT ",count)
                    """
                    #Np = 32                                                                                                                               
                    #P  = 128                                                                                                                              
                    #L  = Np/8 
               
                    """
                    floats = []
                   
                    for i in range(0,len(data),4):
                        floats.append(struct.unpack_from('f',data[i:i+4])[0])
                    
                    print("ll",len(floats))
                    """

                    for i in range(0, len(floats),input_num):
                        dat = floats[i:i+(input_num)]

                        if not len(dat) == input_num:
                            break
                        inp.append(np.array(dat))
                        out.append(z)
                        
                        if count > 1000:
                            break 
                        count += 1
            
                    break
                    

                    
            #break
                            
            mcount += 1

        print("About to train")
        
        if train:

            print(len(inp),len(out))
            for epoch in xrange(0, 10000):
                cvalues = sess.run([train, loss, W_hidden, b_hidden, W_output],
                       feed_dict={n_input: inp, n_output: out})
                if epoch % 200 == 0:
                    print("")
                    print("step: {:>3}".format(epoch))
                    print("loss: {}".format(cvalues[1]))

            save_graph(sess,"/tmp/","saved_checkpoint","checkpoint_state","input_graph.pb","output_graph.pb")
        else:
            
            sess, inp_, out_ = load_graph("/tmp/output_graph.pb","/tmp")

            print("le",len(inp),type(inp))

            ret = sess.run(out_,feed_dict={inp_: inp})
            print(len(ret),len(inp),len(out))

            ret = [ret[0]]
            out = [out[0]]
            
            print(ret)
            print(out)
            print(100.0 * np.sum(np.argmax(ret, 1) == np.argmax(out, 1))/ len(ret))
            
        
    except [[KeyboardInterrupt]]:
        pass

