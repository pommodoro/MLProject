

# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import math


###
### Convolutional Neural Network (CNN) for MNIST
###
class CnnMnist:
    
    def __init__( self, session, n_in, n_out, mode ):

    	# instantiate session
        self.session  = session
        self.n_in     = n_in  # 28*28
        self.n_out    = n_out # 10
        self.mode     = mode

        # data placeholders
        self.x    = tf.placeholder(tf.float32, [None, n_in], name='x')
        self.y    = tf.placeholder(tf.float32, [None, n_out], name='y')
        self.x_in = tf.reshape(self.x, [-1,self.n_in])

        ##
        ## Network Architecture
        ##

        # Input Layer
        self.input_layer = tf.reshape(self.x, [-1, 28, 28, 1])

        # Convolutional Layer #1
        self.conv1 = tf.layers.conv2d(
            inputs=self.input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        self.conv2 = tf.layers.conv2d(
            inputs=self.pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling layer #2
        self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = tf.reshape(self.pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training = self.mode )

        # Logits Layer
        self.logits = tf.layers.dense(inputs=dropout, units=10)
        self.q      = tf.argmax(input = self.logits, axis=1)
        
        # Output Layer
        onehot_labels = tf.one_hot( indices = tf.cast(self.y, tf.int32), depth = 10 )
        self.loss     = tf.nn.softmax_cross_entropy_with_logits(
            labels = self.y, logits = self.logits )

        self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

    
    # method to compute y given x
    def compute(self, x):
        return self.session.run(self.q, feed_dict={self.x:np.reshape(x,[-1,self.n_in])})
    
    # method to train network
    def train(self, x_batch, y_batch): 
        # take a training step
       _ = self.session.run(self.train_step, feed_dict={self.x: x_batch, self.y: y_batch})

    # acessor method for output after pooling layers
    def getPools(self):
        return ( self.pool1, self.pool2 )

    # acessor method for output after convolutional layers
    def getConvs(self):
    	return ( self.conv1, self.conv2 )

    # acessor method for loss
    def getLoss(self):
        return self.loss

    # method to instantiate deconvolutional neural net
    def createDeconvNet(self):
        return CnnMnist.DeconvMnist(self, self.session, self.n_in, self.n_out, self.mode )


    ###
    ### Nested Class: Deconvolutional Neural Network (CNN) for MNIST
    ###
    class DeconvMnist:
        
        def __init__( self, outer, session, inDim, outDim, mode ):


        	# def __init__(self, outer_instance):
        	#     self.outer_instance = outer_instance
        	#         self.outer_instance.somemethod()

        	#     def inner_method(self):
        	#         self.outer_instance.anothermethod()

        	# data placeholders
        	#self.inputImage = tf.placeholder(tf.float32, [None, inDim], name='x')
        	#self.inputLabel = tf.placeholder(tf.float32, [None, outDim], name='y')

        	# instantiate outer class in inner class
        	self.cnn = outer#( session, inDim, outDim, mode ) # acho que vai dar ruim - instaciando de novo a classe

        	self.inDim  = inDim
        	self.outDim = outDim

        	#
        	## Deconvoluting 1st layer
        	##

        	# unpool
        	#self.unPool1 = self.unpool( self.cnn.pool1 )

        	# unrelu and deconvolute (filter)
        	self.unConv1 = tf.layers.conv2d_transpose(  # check dimensions
        	    #inputs = unPool1, 
        	    inputs = self.cnn.conv1, 
        	    filters = 32, 
        	    kernel_size=[5, 5],
        	    padding="SAME" ,
        	    activation = tf.nn.relu )

        	# get activations for layer 1
        	#self.activations1 = self.calculateActivations(inputImage, inputLabel, 1)


        	##
        	## Deconvoluting 2nd layer
        	##

        	# 1st unpool
        	#self.unPool21 = self.unpool( self.cnn.pool2 )

        	# 1st unrelu and deconvolution
        	self.unConv21 = tf.layers.conv2d_transpose( 
        	    #inputs = self.unPool21, 
        	    inputs = self.cnn.conv2, 
        	    filters = 64, 
        	    kernel_size=[5,5], 
        	    padding="SAME",
        	    use_bias = False,
        	    activation = tf.nn.relu )

        	# 2nd unpool
        	self.unPool22 = self.unpool( self.unConv21 )

        	# 2nd deconvolution
        	self.unConv22 = tf.layers.conv2d_transpose( 
        	    inputs = self.unPool22, 
        	    filters = 32, 
        	    kernel_size=[5,5], 
        	    padding="SAME",
        	    use_bias = False,
        	    activation = tf.nn.relu )

        	# get activations for layer 2
        	#self.activations2 = self.calculateActivations(inputImage, inputLabel, 2)



        # calculate activations for layer (1 or 2)
        def calculateActivations(self, inputImage, inputLabel, layer):

        	if( layer == 1 ):
        		return self.unConv1.eval(feed_dict={self.cnn.x: np.reshape(inputImage,[-1,self.inDim])})
        	else:
        		return self.unConv22.eval(feed_dict={self.cnn.x: np.reshape(inputImage,[-1,self.inDim])})

        # return activations
        #def getActivations(self):
        #	return self.activations1, self.activations2


        # unpooling method from kvfrans ---- REDO IT
        def unpool(self, value, name='unpool'):
    	    """N-dimensional version of the unpooling operation from
    	    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    	    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    	    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    	    """
    	    # with tf.name_scope(name) as scope:
    	    #     sh = value.get_shape().as_list()
    	    #     dim = len(sh[1:-1])
    	    #     out = (tf.reshape(value, [-1] + sh[-dim:]))
    	    #     for i in range(dim, 0, -1):
    	    #         out = tf.concat(i, [out, out])
    	    #     out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
    	    #     out = tf.reshape(out, out_size, name=scope)
    	    return value

