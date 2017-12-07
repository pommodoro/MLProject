
# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline


def network(features, labels, mode):
        self.session = session
        self.n_in = n_in
        self.n_out = n_out
        n_hidden = 2
        self.n_hidden_new = 30
        # data placeholders
        self.x = tf.placeholder(tf.float32, [None, n_in], name='x')
        self.y = tf.placeholder(tf.float32, [None, n_out], name='y')
        self.x_in = tf.reshape(self.x, [-1,self.n_in])
       
        #
        # 3 layer network
        #
        
        # layer 1
        W_fc1 = tf.get_variable('W_fc1', shape=[self.n_in,self.n_hidden])
        h_fc1 = tf.nn.relu(tf.matmul(self.x_in, self.W_fc1), self.b_fc1, name='layer1')

        # layer 2
        self.W_fc_new = tf.get_variable('W_fc_new', shape=[self.n_hidden,self.n_hidden_new])
        self.b_fc_new = tf.get_variable('b_fc_new', shape=[self.n_hidden_new])
        self.h_fc_new = tf.nn.relu(tf.add(tf.matmul(self.h_fc1, self.W_fc_new), self.b_fc_new, name='layer_new'))
        
        # layer 3
        self.W_fc2 = tf.get_variable('W_fc2', shape=[self.n_hidden_new,self.n_out])
        self.b_fc2 = tf.get_variable('b_fc2', shape=[self.n_out])
        self.q = tf.add(tf.matmul(self.h_fc_new, self.W_fc2), self.b_fc2, name='layer2')
        
        # loss, train_step, etc.
        self.loss = tf.reduce_sum(tf.square(self.y - self.q),1)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)