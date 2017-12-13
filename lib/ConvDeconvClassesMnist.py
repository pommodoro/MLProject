    

###
### maybe try to use tf.nn.conv2d?
###

# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import math
from scipy.misc import imsave

###
### Convolutional Neural Network (CNN) for MNIST
###
class CnnMnist:
    
    def __init__( self, session, n_in = 28*28, n_out = 10, mode = True ):

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

    # acessor method to get filter weights of convolutional layers
    def getWeights(self):
        return ( self.W_c1, self.W_c2 )

    # acessor method for loss
    def getLoss(self):
        return self.loss

    # saver method to save trained cnn in disk 
    def netSaver(self, savePath):
        saver = tf.train.Saver()
        saver.save(self.session, savePath)
        print("Model saved in file: %s" % savePath)

    # loader method to restore weights of a pretrained cnn
    def netLoader(self, loadPath):
        loader = tf.train.Saver({"W_c1":self.W_c1, "W_c2":self.W_c2})
        restoredModel= loader.restore(self.session, loadPath)
        print("Model restored from %s" % loadPath)


    # method to initialize filter weights
    def initWeight(shape):
        weights = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(weights)

    # method to instantiate deconvolutional neural net
    def createDeconvNet(self, inputImage, inputLabel):
        return CnnMnist.DeconvMnist( self, self.session, self.n_in, self.n_out, inputImage, inputLabel )

#''' DON'T COMMENT ME PLEASE!!!
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
            self.deconv1 = self.deconvLayer1( inputImage, inputLabel )
            self.deconv2 = self.deconvLayer2( inputImage, inputLabel )


        def deconvLayer1( self, inputImage, inputLabel ):

            #
            ## Deconvoluting 1st layer
            ##
            
            # get activations for layer 1
            activations1 = self.calculateActivations( inputImage, inputLabel, 1 )

            # convert from array to tensor
            act1_tf = tf.convert_to_tensor( activations1, np.float32 )

            # unpool
            unPool1 = self.unpool( act1_tf )

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

            return unConv1


        def deconvLayer2( self, inputImage, inputLabel ):


            ##
            ## Deconvoluting 2nd layer
            ##

            # get activations for layer 2
            activations2 = self.calculateActivations(inputImage, inputLabel, 2)

            # convert from array to tensor
            act1_tf = tf.convert_to_tensor( activations2, np.float32 )

            # 1st unpool
            unPool1 = self.unpool( act1_tf )

            # 1st unrelu
            unRelu1 = tf.nn.relu( unPool1 )

            # 1st deconvolute (filter)
            unConv1 = tf.nn.conv2d_transpose( 
                #activations1,
                unRelu1,
                self.cnn.W_c2,
                output_shape = [ inputImage.shape[0], 14, 14, 32],
                strides = [1, 1, 1, 1],
                padding = "SAME"  )

            # 2nd unpool
            unPool2 = self.unpool( unConv1 )

            # 2nd relu
            unRelu2 = tf.nn.relu( unPool2 )

            # 2nd deconvolute (filter)
            # 1st deconvolute (filter)
            unConv2 = tf.nn.conv2d_transpose( 
                #activations1,
                unRelu2,
                self.cnn.W_c1,
                output_shape = [ inputImage.shape[0], 28, 28, 1],
                strides = [1, 1, 1, 1],
                padding = "SAME"  )

            return unConv2


        # calculate activations for layer (1 or 2)
        def calculateActivations( self, inputImage, inputLabel, layer ):

            if( layer == 1 ):
                return self.cnn.pool1.eval(feed_dict={self.cnn.x: np.reshape(inputImage,[-1,self.inDim])})
            else:
                return self.cnn.pool2.eval(feed_dict={self.cnn.x: np.reshape(inputImage,[-1,self.inDim])})


        def getDeconv( self ):
            return self.deconv1, self.deconv2

        # method to unpool (taken from kvfrans - put link!)
        def unpool( self, value ):
            """N-dimensional version of the unpooling operation from
            https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

            :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
            :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
            """
            #with tf.name_scope(name) as scope:
            sh = value.get_shape().as_list()
            dim = len(sh[1:-1])
            out = (tf.reshape(value, [-1] + sh[-dim:]))
            for i in range(dim, 0, -1):
                out = tf.concat( [out, out], i)
            out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
            out = tf.reshape(out, out_size)#, name=scope)
            return out

        #Returns de deconvoluted layer1 as numpy array, with isolated nodes,
        #and save the images on the "img" folder
        def displayFeatures1( self, inputImage, inputLabel, n_best = 10, k = 9):

            #
            ## Deconvoluting 1st layer
            ##
            
            # get activations for layer 1
            activations1 = self.calculateActivations( inputImage, inputLabel, 1 )
            
            filters = random.sample(range(activations1.shape[-1]), k)
            aux = activations1.shape[0] - n_best
            
            all_isolations = np.zeros([k, n_best, 28, 28, 1])
            j = 0
            best_index = np.zeros([k, n_best])
            
            for i in filters:
            # Isolate filters
                print("Deconvoluting Layer 1 Filter: {}".format(i))
                isolated = activations1.copy()
                isolated[:,:,:,:i]   = 0
                isolated[:,:,:,i+1:] = 0
    
                Norm1 = np.linalg.norm(isolated, axis = (2, 3))
                Norm2 = np.linalg.norm(Norm1, axis = 1)
                
                best = np.where(aux <= np.argsort(Norm2))[0]
    
                # convert from array to tensor
                act1_tf = tf.convert_to_tensor( isolated, np.float32 )
    
                # unpool
                unPool1 = self.unpool( act1_tf )
    
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
                
                u = unConv1.eval()
                
                u = u[best,]
                best_index[j,:] = best
                
                imsave("img/Deconv1_Node_{}_of_N3.jpg".format(i), u[1,:,:,0])
                
                all_isolations[j,:,:,:,:] = u
                j = j + 1
            
            return all_isolations, best_index, filters


        def displayFeatures2( self, inputImage, inputLabel, n_best = 10, k = 9):

            ##
            ## Deconvoluting 2nd layer
            ##

            # get activations for layer 2
            activations2 = self.calculateActivations(inputImage, inputLabel, 2)
            
            filters = random.sample(range(activations2.shape[-1]), k)
            aux = activations2.shape[0] - n_best
            
            all_isolations = np.zeros([k, n_best, 28, 28, 1])
            j = 0
            best_index = np.zeros([k, n_best])
            
            for i in filters:
            # Isolate filters
                print("Deconvoluting Layer 2 Filter: {}".format(i))
                isolated = activations2.copy()
                isolated[:,:,:,:i]   = 0
                isolated[:,:,:,i+1:] = 0
    
                Norm1 = np.linalg.norm(isolated, axis = (2, 3))
                Norm2 = np.linalg.norm(Norm1, axis = 1)
                
                best = np.where(aux <= np.argsort(Norm2))[0]
    
                # convert from array to tensor
                act1_tf = tf.convert_to_tensor( isolated, np.float32 )
    
                # 1st unpool
                unPool1 = self.unpool( act1_tf )
    
                # 1st unrelu
                unRelu1 = tf.nn.relu( unPool1 )
    
                # 1st deconvolute (filter)
                unConv1 = tf.nn.conv2d_transpose( 
                    #activations1,
                    unRelu1,
                    self.cnn.W_c2,
                    output_shape = [ inputImage.shape[0], 14, 14, 32],
                    strides = [1, 1, 1, 1],
                    padding = "SAME"  )
    
                # 2nd unpool
                unPool2 = self.unpool( unConv1 )
    
                # 2nd relu
                unRelu2 = tf.nn.relu( unPool2 )
    
                # 2nd deconvolute (filter)
                # 1st deconvolute (filter)
                unConv2 = tf.nn.conv2d_transpose( 
                    #activations1,
                    unRelu2,
                    self.cnn.W_c1,
                    output_shape = [ inputImage.shape[0], 28, 28, 1],
                    strides = [1, 1, 1, 1],
                    padding = "SAME"  )
                
                u = unConv2.eval()
                
                u = u[best,]
                best_index[j,:] = best
                
                imsave("img/Deconv2_Node_{}_of_N3.jpg".format(i), u[1,:,:,0])
                
                all_isolations[j,:,:,:,:] = u
                j = j + 1
            
            return all_isolations, best_index, filters

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

#''' DON'T COMMENT ME PLEASE!!!
