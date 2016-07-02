#!/usr/bin/python2

import tensorflow as tf    
from tensor import *

# Based on the answer at
# http://stackoverflow.com/questions/33997823/tensorflow-mlp-not-training-xor

input_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]  # XOR input
output_data = [[0.], [1.], [1.], [0.]]  # XOR output
n_input = tf.placeholder(tf.float32, shape=[None, 2], name="inp")
n_output = tf.placeholder(tf.float32, shape=[None, 1], name="n_output")

hidden_nodes = 5
b_hidden = tf.Variable(tf.random_normal([hidden_nodes]), name="hidden_bias")
W_hidden = tf.Variable(tf.random_normal([2, hidden_nodes]), name="hidden_weights")

# calc hidden layer's activation
hidden = tf.sigmoid(tf.matmul(n_input, W_hidden) + b_hidden)
W_output = tf.Variable(tf.random_normal([hidden_nodes, 1]), name="output_weights")  # output layer's weight matrix
output = tf.sigmoid(tf.matmul(hidden, W_output),name="out")  # calc output layer's activation
cross_entropy = -(n_output * tf.log(output) + (1 - n_output) * tf.log(1 - output))
# cross_entropy = tf.square(n_output - output)  # simpler, but also works
loss = tf.reduce_mean(cross_entropy)  # mean the cross_entropy
optimizer = tf.train.AdamOptimizer(0.01)  # take a gradient descent for optimizing with a "stepsize" of 0.1
train = optimizer.minimize(loss)  # let the optimizer train

init = tf.initialize_all_variables()

sess = tf.Session()  # create the session and therefore the graph
sess.run(init)  # initialize all variables  

for epoch in xrange(0, 2001):
    # run the training operation
    cvalues = sess.run([train, loss, W_hidden, b_hidden, W_output],
                       feed_dict={n_input: input_data, n_output: output_data})

    if epoch % 200 == 0:
        print("")
        print("step: {:>3}".format(epoch))
        print("loss: {}".format(cvalues[1]))

print("")
print("input: {} | output: {}".format(input_data[0], sess.run(output, feed_dict={n_input: [input_data[0]]})))
print("input: {} | output: {}".format(input_data[1], sess.run(output, feed_dict={n_input: [input_data[1]]})))
print("input: {} | output: {}".format(input_data[2], sess.run(output, feed_dict={n_input: [input_data[2]]})))
print("input: {} | output: {}".format(input_data[3], sess.run(output, feed_dict={n_input: [input_data[3]]})))

#save_graph(sess,output_path,checkpoint,checkpoint_state_name,input_graph_name,output_graph_name):   
save_graph(sess,"/tmp/graph","saved_checkpoint","checkpoint_state","input_graph.pb","output_graph.pb")
