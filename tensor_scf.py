#!/usr/bin/python2
from tensor import *
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
#import tflearn
from scipy import signal
import matplotlib.pyplot as plt
import glob
import tflearn
import pickle
import cPickle

import matplotlib.pyplot as plt

snrv = ["20","15","10","5","0","-5","-10","-15","-20"] 

# Load SCF training data from previously pickled file
load_scf_training = True
# Load SCF testing data from previously pickled file
load_scf_testing = True
# Save SCF data to pickled file
save = False

# Modulation schemes
mod = ["2psk","4psk","8psk","fsk"]

# Load ANN from file 
loadann = True

input_num = 760 

training = True


SIGLEN = 1024 * 15


# Generate a graph of SCF data
def graph(za):
    
    nx, ny = za.shape[1], za.shape[0]
    y = np.arange(0,1.0,1.0/ny)
    x = np.arange(-0.5,0.5,1.0/nx)

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    ha.set_xlabel('Frequency')
    ha.set_ylabel('Alpha')
    ha.set_zlabel('SCF')

    X, Y = numpy.meshgrid(x, y) 

    ha.plot_surface(X, Y, za,rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.show()



def scf2(y):

    za = []
    y = y[0:SIGLEN]

    d = collections.deque(maxlen=1)

    FFTsize = 1024*5
    N = int(len(y)/FFTsize)
    T = int(len(y) / N) # Frame length

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

    for a in np.arange(0,1,0.005):
        Fs = T #*2
        al = a * Fs
        alph.append(a)
        #print("A",a)
    
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
        
            Sxf = xfp * np.conj(xfm) 
 
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
        ff = collections.deque(maxlen=10)

        for v in tm:
            print(v)
            ff.append(v)
            ffo = np.mean(ff,axis=0)
            tm2.append(ffo)
        quit()

        d.append(tm2)
        # mean of columns
        smoothed = np.mean(np.array(d),axis=0)
        za.append(smoothed)
    
    out = np.array(za) 
    o = (out/out.max())
    o[o == np.inf] = 0

    return o


# Generate 2D array of SCF data
def scf(y):

    za = []
    d = collections.deque(maxlen=10)
    y = y[0:SIGLEN]

    # Seemed best at 1000, at 88% 

    FFTsize = 200
    N = len(y)/FFTsize



    #N = 100#3000             # Number of frames
    T = int(len(y) / N) # Frame length
    #print("Flen",T)
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

    for a in np.arange(0,1,0.005):
    #for a in np.arange(0,1,0.05):


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
            Sxf = Sxf * (np.e**-(1j*2*np.pi*al*(count*T)))
    
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
 
    out = np.array(za) 
    o = (out/out.max())
    o[o == np.inf] = 0

    return o

from multiprocessing import Process, Queue

def process(path,m,i,z,qu):
    y = np.fromfile("%s/%s-snr%d.dat" % (path,m,i), dtype=np.complex64)
    #y = np.array_split(y,int(len(y)/(SIGLEN)))
    y = np.array_split(y,int(len(y)/(SIGLEN)))


    print("loading %s/%s-snr%d.dat  " % (path,m,i))
    
    o = []
    oo = []
    
    c = 0
    for q in y:
        o.append(scf2(q[0:SIGLEN]))
        oo.append(z)
        c += 1
        if c > 10:
            break
    
    qu.put((o,oo,m,i,z))

# Load dataset of different modulation schemes
def load_data(path,train,rn=range(0,9)):

    out = [[] for k in range(0,9)]
    out_o = [[] for k in range(0,9)]

    count = 0
    plist = []

    q = Queue() #create a queue object

    for m in mod :
        
        z = np.zeros((len(mod),))
        z[count] = 1

        for i in rn:
            
            p = Process(target=process, args=(path,m,i,z,q))
            plist.append(p)

            p.start()
             
            """
            y = np.fromfile("%s/%s-snr%d.dat" % (path,m,i), dtype=np.complex64)
            y = np.array_split(y,int(len(y)/(SIGLEN)))

            print(len(rn),rn,i)
            print("loading %s/%s-snr%d.dat  " % (path,m,i))
            c=0
            for q in y:
                out[i].append(scf(q[0:SIGLEN]))
                out_o[i].append(z)
                c += 1
            """
            

        count += 1

    for p in plist:
        
        job = q.get()

        for i in range(0,len(job[0])):
            out[job[3]].append(job[0][i])
            out_o[job[3]].append(job[1][i])

    for p in plist:
        p.join()
        print("joined")


    #for job in iter(q.get, None): # Replace `None` as you need.
    
    
    #if train:
    #    o = [ x for y in out for x in y]
    #    oo = [ x for y in out_o for x in y]
    #    return (o,oo)        
    #else:
    return (out,out_o)

train_ = []
train_out = []

test = []
test_out = []

# Load pickled SCF training data
if load_scf_training:
    train_ = cPickle.load(open('train2.dat', 'rb'))
    train_out = cPickle.load(open('train_o2.dat', 'rb'))
else:
    train1,train_out = load_data("data/train-rnd1",True)
    #train2,train_out2 = load_data("data/train-rnd2",True,range(0,1))
    
    """
    for i in range(0,9):
        #for noise in range(0,5):    
        for v in train2[i]:
            for val in v:
                train1[i].append(val)

        for v in train_out2[i]:
            for val in v:
                train_out[i].append(val)
    """

    #train_ = train1 + train2
    #train_out = train_out + train_out2
    train_ = train1
    train_out = train_out


#print(len(train_[0]),type(train_[0]))
#quit()





# Load pickled SCF testing data
if load_scf_testing:
    test = cPickle.load(open('test2.dat', 'rb'))
    test_out = cPickle.load(open('test_o2.dat', 'rb'))
else:
    print("loading rnd 3 ")
    test,test_out = load_data("data/train-rnd3",False)



def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]





# Save SCF data to pickled files
if save:
    with open('train2.dat','w') as f:
        pickle.dump(train_,f)

    with open('train_o2.dat','w') as f:
        pickle.dump(train_out,f)

    with open('test2.dat','w') as f:
        pickle.dump(test,f)

    with open('test_o2.dat','w') as f:
        pickle.dump(test_out,f)

"""
t = train_[0][0]
t = np.array_split(t,10)

for i in range(len(t)):
    print(t[i][np.argmax(t[i])])
quit()
"""

print("Tensor flow starting")
inputs = len(train_[0][0])
hidden = int(inputs * (0.89))
print("Inputs ",inputs,"Hidden ", hidden)

input_num = inputs





def getnet(datain,dataout,modulation):
    with tf.Graph().as_default():
        t = []
        to = []
        te = []
        teo = []
        for i in range(0,9):
            c = 0
            #for noise in range(0,5):
            b1 = 0
            b2 = 0
            din, dino = shuffle_in_unison_inplace(numpy.array(datain[i]),numpy.array(dataout[i]))
            for v in din:
                dat = []
                for zz in range(v.shape[0]):
                    dat.append(v[zz][np.argmax(v[zz])])
                t.append(dat)
                mod = dino[c]
                print "Mod",mod,"Modul",modulation
                if (mod == modulation).all():
                    to.append([1])
                    b1 += 1
                else:
                    to.append([0])
                    b2 += 1
                if b1 >= 9 and b2 >= 9:
                    print("BREAK")
                    break
                c += 1
        # Model based on https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py
        # Parameters
        learning_rate = 0.001
        training_epochs = 10000
        batch_size = 100
        display_step = 10
        # Network Parameters
        n_hidden_1 = 15 #int(input_num / 4.0) # 1st layer number of features
        n_hidden_2 = 15 #int(input_num / 4.0) # 1st layer number of features
        #n_hidden_1 = int(input_num / 4) # 1st layer number of features
        #n_hidden_2 = int(input_num / 4) # 2nd layer number of features
        n_input = input_num 
        n_classes = 1 #len(mod) 
        # tf Graph input
        x = tf.placeholder("float", [None, n_input],name="inp")
        y = tf.placeholder("float", [None, n_classes])
        keep_prob = tf.placeholder(tf.float32)
        # Create model
        def multilayer_perceptron(x, weights, biases):
            # Hidden layer with RELU activation
            #layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            #layer_1 = tf.nn.relu(layer_1)
            layer_1 = tf.sigmoid(tf.matmul(x, weights['h1']) + biases['b1'])
            #drop = tf.nn.dropout(layer_1, keep_prob)
            # Hidden layer with RELU activation
            #layer_2 = tf.sigmoid(tf.matmul(layer_1, weights['h2']) + biases['b2'])
            #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            #layer_2 = tf.nn.relu(layer_2)
            # Output layer with linear activation
            out_layer = tf.sigmoid(tf.matmul(layer_1, weights['out'],name="out")) # + biases['out'])
            return out_layer
        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        # Construct model
        pred = multilayer_perceptron(x, weights, biases)
        # Define loss and optimizer
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        cost = tf.reduce_mean(-(y * tf.log(pred) + (1 - y) * tf.log(1 - pred))   )
        cost = tf.reduce_mean(tf.square(y - pred)) 
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        # Initializing the variables
        init = tf.initialize_all_variables()
        # Launch the graph
        #with tf.Session() as sess:
        sess = tf.Session()
        if True:
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = len(t)
                #t, to = shuffle_in_unison_inplace(numpy.array(t),numpy.array(to))
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: t, y: to})
                avg_cost = c
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print "Epoch:", '%04d' % (epoch+1), "{:.9f}".format(avg_cost)
            return sess, pred, x

"""
count = 0
nt = [ [] for k in range(len(mod)) ]
for m in mod:
    z = np.zeros((len(mod),))
    z[count] = 1
    s,pred,x = getnet(train_,train_out,z)
    nt[count] = (s,pred,x)
    count += 1
for i in range(9):
    test_i = test[i]
    good = 0 
    allv = 0
    for v in test_i:
        dat = []
        for zz in range(v.shape[0]):
            dat.append(v[zz][np.argmax(v[zz])])
        p = np.array(np.zeros((len(mod),)))
        c = 0
        for n in nt:
            v = n[0].run(n[1],feed_dict={n[2]: [dat]})[0][0]
            p[c] = v
            c += 1
        if np.argmax(p) == np.argmax(test_out[i][allv]):
            good += 1
        allv += 1
    print "SNR ",i," :  ",float(good)/float(allv)
"""
          
"""          
with tf.Graph().as_default():

t = []
to = []
for i in range(9):
    for i2 in range(len(train_[i])):
        t.append(train_[i][i2])
        to.append(train_out[i][i2])
with tf.Graph().as_default():
    hidden = 7
    tflearn.init_graph(num_cores=4)
    net = None
    net1 = tflearn.input_data(shape=[None,train_[0][0].shape[0],train_[0][0].shape[1]])
    net1 = tflearn.fully_connected(net1, hidden,activation='sigmoid') #, activation='sigmoid')
    net1 = tflearn.fully_connected(net1, len(mod), activation='softmax')
    regressor = tflearn.regression(net1, optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy') #, loss=lossv)
    m = tflearn.DNN(regressor,tensorboard_verbose=3)
    m.fit(t, to, n_epoch=50,shuffle=True, snapshot_epoch=False,show_metric=True)
"""





            
