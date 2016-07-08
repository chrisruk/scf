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
import pickle
import matplotlib.pyplot as plt

snrv = ["20","15","10","5","0","-5","-10","-15","-20"] 

# Load SCF training data from previously pickled file
load_scf_training = True

# Load SCF testing data from previously pickled file
load_scf_testing = True

# Save SCF data to pickled file
save = True

# Modulation schemes
mod = ["2psk","4psk","8psk","fsk"]

# Load ANN from file 
loadann = True

input_num = 760 

training = True

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

# Generate 2D array of SCF data
def scf(y):

    za = []
    d = collections.deque(maxlen=10)
    y = y[0:1024*5]
    N = 100#3000             # Number of frames
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

    for a in np.arange(0,1,0.1):


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

    return o.flatten()


"""
# Graph data

# payload 90
y = np.fromfile("data/train-2/fsk-snr0.dat", dtype=np.complex64)
graph(scf(y))

# payload 189
y = np.fromfile("data/train-3/fsk-snr0.dat", dtype=np.complex64)
graph(scf(y))

"""


# Load dataset of different modulation schemes
def load_data(path,train):

    out = [[] for k in range(9)]
    out_o = [[] for k in range(9)]

    count = 0
    for m in mod :
        
        z = np.zeros((len(mod),))
        z[count] = 1

        for i in range(0,1):
        
            y = np.fromfile("%s/%s-snr%d.dat" % (path,m,i), dtype=np.complex64)
            y = np.array_split(y,int(len(y)/(1024*5)))

            print("loading %s/%s-snr%d.dat  " % (path,m,i))
            c=0
            for q in y:
                out[i].append(scf(q[0:1024*5]))
                out_o[i].append(z)
                c += 1
        count += 1

    
    if train:
        o = [ x for y in out for x in y]
        oo = [ x for y in out_o for x in y]
        return (o,oo)        
    else:
        return (out,out_o)

train_ = []
train_out = []

test = []
test_out = []

# Load pickled SCF training data
if load_scf_training:
    train_ = pickle.load(open('train2.dat', 'rb'))
    train_out = pickle.load(open('train_o2.dat', 'rb'))
else:
    train1,train_out = load_data("data/train-rnd1",True)
    train2,train_out2 = load_data("data/train-rnd2",True)

    train_ = train1 + train2
    train_out = train_out + train_out2

# Load pickled SCF testing data
if load_scf_testing:
    test = pickle.load(open('test2.dat', 'rb'))
    test_out = pickle.load(open('test_o2.dat', 'rb'))
else:
    print("loading rnd 3 ")
    test,test_out = load_data("data/train-rnd3",False)

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



print("Tensor flow starting")
inputs = len(train_[0])
hidden = int(inputs * (0.89))
print("Inputs ",inputs,"Hidden ", hidden)

# Hack - This could be improved likely, by using a built-in function to Tensor Flow, can't
# seem to find one at the moment
def thresh(i):
    if i >= 0.5:
        return 1
    else: 
        return 0


with tf.Graph().as_default():

    if True:
        nop = True

        #print("NEURONS",inp[0].shape[0]*inp[0].shape[1])
        # Parameters
        learning_rate = 0.001
        training_epochs = 2000
        batch_size = 100
        display_step = 1

        # Network Parameters
        n_hidden_1 = input_num / 6 # 1st layer number of features
        n_hidden_2 = input_num / 6 # 2nd layer number of features
        n_input = input_num # MNIST data input (img shape: 28*28)
        n_classes = len(mod) # MNIST total classes (0-9 digits)

        # tf Graph input
        x = tf.placeholder("float", [None, n_input],name="inp")
        y = tf.placeholder("float", [None, n_classes])


        # Create model
        def multilayer_perceptron(x, weights, biases):
            # Hidden layer with RELU activation
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            # Hidden layer with RELU activation
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
            # Output layer with linear activation
            out_layer = tf.matmul(layer_2, weights['out'],name="out") + biases['out']
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
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.

                batch_x = train_
                batch_y = train_out
        
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print "Epoch:", '%04d' % (epoch+1)
            
            print "Optimization Finished!"

            save_graph(sess,"/tmp/","saved_checkpoint","checkpoint_state","input_graph.pb","output_graph.pb")

            #pred.save("mlayer.ann")

            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
            for i in range(9):
                test_i = test[i]
                test_o = test_out[i]
                
                if not len(test_i) == 0:
                    print "Accuracy:", accuracy.eval({x: test_i, y: test_o})

            
