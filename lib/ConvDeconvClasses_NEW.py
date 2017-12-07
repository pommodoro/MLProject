

###
### maybe try to use tf.nn.conv2d?
###

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
        self.W_c1 = tf.get_variable( 'W_c1', shape = [ 5, 5, 1, 32 ] )
        self.W_c2 = tf.get_variable( 'W_c2', shape = [ 5, 5, 32, 64 ] )

        ##
        ## Network Architecture
        ##

        # Input Layer
        self.input_layer = tf.reshape(self.x, [-1, 28, 28, 1])

        #
        # Convolutional Layer #1
        #

        # filter
        self.conv1 = tf.nn.conv2d(
            input = self.input_layer,
            filter = self.W_c1,
            padding = "SAME",
            strides = [1,1,1,1] )

        # relu
        self.relu1 = tf.nn.relu( self.conv1 )

        #
        # Pooling Layer #1
        #
        self.pool1 = tf.layers.max_pooling2d(inputs=self.relu1, pool_size=[2, 2], strides=2)

        #
        # Convolutional Layer #2
        #

        # filter
        self.conv2 = tf.nn.conv2d(
            input = self.pool1,
            filter = self.W_c2,
            padding = "SAME",
            strides = [1,1,1,1] )

        # relu
        self.relu2 = tf.nn.relu( self.conv2 )

        #
        # Pooling layer #2
        #
        self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=2)

        #
        # Dense Layer
        #
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

    # method to initialize filter weights
    def initWeight(shape):
        weights = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(weights)

    # method to instantiate deconvolutional neural net
    def createDeconvNet(self, inputImage, inputLabel):
        return CnnMnist.DeconvMnist( self, self.session, self.n_in, self.n_out, inputImage, inputLabel )


    ###
    ### Nested Class: Deconvolutional Neural Network (CNN) for MNIST
    ###
    class DeconvMnist:
        
        def __init__( self, outer, session, inDim, outDim, inputImage, inputLabel ):

        	# data placeholders
        	#self.inputImage = tf.placeholder(tf.float32, [None, inDim], name='x')
        	#self.inputLabel = tf.placeholder(tf.float32, [None, outDim], name='y')

        	# instantiate outer class in inner class
            self.cnn    = outer
            self.inDim  = inDim
            self.outDim = outDim
            self.sess   = session
            self.deconv = self.deconvProcess( inputImage, inputLabel )


        def deconvProcess( self, inputImage, inputLabel ):

        	#
        	## Deconvoluting 1st layer
        	##

        	#unPool1PlaceHolder = tf.placeholder("float",[None,14,14,32])
        	#unConv1PlaceHolder = tf.placeholder("float",[None,28,28,32])
            
            # get activations for layer 1
            activations1 = self.calculateActivations( inputImage, inputLabel, 1 )

        	# unpool
        	unPool1 = self.unpool( self.cnn.pool1 )

            # unrelu
            unRelu1 = tf.nn.relu( unPool1 )

        	# deconvolute (filter)
            unConv1 = tf.nn.conv2d_transpose(  # check dimensions
        	    #activations1,
                unRelu1,
                self.cnn.W_c1,
                output_shape = [ inputImage.shape[0], 28, 28, 1],
                strides = [1, 1, 1, 1],
                padding = "SAME"  )

            return unConv1# self.sess.run( unConv1 )


        	# ##
        	# ## Deconvoluting 2nd layer
        	# ##

        	# #unPool2PlaceHolder = tf.placeholder("float",[None,7,7,32])
        	# #self.unConv2PlaceHolder = tf.placeholder("float",[None,14,14,32])

        	# # 1st unpool
        	# #self.unPool21 = self.unpool( self.cnn.pool2 )

        	# # 1st unrelu and deconvolution
        	# self.unConv21 = tf.layers.conv2d_transpose( 
        	#     #inputs = self.unPool21, 
        	#     inputs = self.cnn.conv2, 
        	#     filters = 64, 
        	#     kernel_size=[5,5], 
        	#     padding="SAME",
        	#     activation = tf.nn.relu,
        	#     use_bias = False )

        	# # 2nd unpool
        	# self.unPool22 = self.unpool( self.unConv21 )

        	# # 2nd deconvolution
        	# self.unConv22 = tf.layers.conv2d_transpose( 
        	#     inputs = self.unPool22, 
        	#     filters = 32, 
        	#     kernel_size=[5,5], 
        	#     padding="SAME",
        	#     activation = tf.nn.relu, 
        	#     use_bias = False )

        	# # get activations for layer 2
        	# self.activations2 = self.calculateActivations(inputImage, inputLabel, 2)


        # calculate activations for layer (1 or 2)
        def calculateActivations( self, inputImage, inputLabel, layer ):

            if( layer == 1 ):
                #return self.cnn.conv1.eval(feed_dict={self.cnn.x: np.reshape(inputImage,[-1,self.inDim])})
                return self.cnn.pool1.eval(feed_dict={self.cnn.x: np.reshape(inputImage,[-1,self.inDim])})
            else:
                #return self.cnn.conv2.eval(feed_dict={self.cnn.x: np.reshape(inputImage,[-1,self.inDim])})
                return self.cnn.pool2.eval(feed_dict={self.cnn.x: np.reshape(inputImage,[-1,self.inDim])})


        def getDeconv( self ):
            return self.deconv


        #def displayFeatures( self, layer ):

        	#isolated = self.activations1.copy()
	        #isolated[:,:,:,:1]   = 0
	        #isolated[:,:,:,1+1:] = 0
	        #pixelactive = self.unConv1.eval( feed_dict = {self.unConv1PlaceHolder: isolated} )
	        #return isolated

        	# if layer == 1:

        	# 	isolated = self.activations1.copy()
	        # 	isolated[:,:,:,:1]   = 0
	        # 	isolated[:,:,:,1+1:] = 0
	        # 	return isolated
	        # 	#print("isolated shape")
	        # 	#print (np.shape(isolated))
	        # 	#totals = np.sum( isolated, axis = (1,2,3) )
	        # 	#best   = np.argmin( totals, axis = 0 )
	        # 	#print (best)
	        # 	#pixelactive = self.unPool1.eval(feed_dict={self.unPool1PlaceHolder: isolated})
	        # 	#pixelactive = self.unConv1.eval(feed_dict={self.unConv1PlaceHolder: isolated[5,:,:,1]})

	        # else:

	        # 	# isolated = self.activations2.copy()
	        # 	# isolated[:,:,:,:1]   = 0
	        # 	# isolated[:,:,:,1+1:] = 0
	        # 	# #print (np.shape(isolated))
	        # 	# totals = np.sum( isolated, axis = (1,2,3) )
	        # 	# best   = np.argmin( totals, axis = 0 )
	        # 	# #print (best)
	        # 	# #pixelactive = self.unPool2.eval(feed_dict={self.unPool2PlaceHolder: isolated})
	        # 	# pixelactive = self.unConv2.eval(feed_dict={self.unConv2PlaceHolder: isolated})


	        # # saves pixel-representations of features from Conv layer 1
	        # featuresReLu1 = tf.placeholder("float",[None,32,32,32])
	        # unReLu = tf.nn.relu(featuresReLu1)
	        # unBias = unReLu
	        # unConv = tf.nn.conv2d_transpose(unBias, wConv1, output_shape=[batchsizeFeatures,imagesize,imagesize,colors] , strides=[1,1,1,1], padding="SAME")
	        # activations1 = relu1.eval(feed_dict={img: inputImage, lbl: inputLabel, keepProb: 1.0})
	        # print (np.shape(activations1))
        	# # display features
        	# for i in xrange(32):
        	#     isolated = self.activations1.copy()
        	#     isolated[:,:,:,:i]   = 0
        	#     isolated[:,:,:,i+1:] = 0
        	#     #print (np.shape(isolated))
        	#     totals = np.sum( isolated, axis = (1,2,3) )
        	#     best   = np.argmin( totals, axis = 0 )
        	#     #print (best)
        	#     pixelactive = self.unConv1.eval(feed_dict={self.unPool1PlaceHolder: isolated})
        	#     # totals = np.sum(pixelactive,axis=(1,2,3))
        	#     # best = np.argmax(totals,axis=0)
        	#     # best = 0
        	#     saveImage(pixelactive[best],"activ"+str(i)+".png")
        	#     saveImage(inputImage[best],"activ"+str(i)+"-base.png")

        	# return False


