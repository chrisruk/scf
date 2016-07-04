#!/usr/bin/python2

import tensorflow as tf    
import numpy as np
from tensor import *

input_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]  # XOR input
sess, inp, out = load_graph("/tmp/output_graph.pb")

print(sess.run(out,feed_dict={inp: [[0., 1. ]]}))
