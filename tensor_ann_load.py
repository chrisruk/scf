#!/usr/bin/python2

import tensorflow as tf    
import numpy as np
from tensor import *

input_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]  # XOR input
#output_data = [[0.], [1.], [1.], [0.]]  # XOR output

temp = "/tmp/"
output_graph_name = "output_graph.pb"
output_graph_path = os.path.join(temp, output_graph_name)

sess, inp, out = load_graph(output_graph_path)

print(sess.run(out,feed_dict={inp: [[0., 1. ]]}))
