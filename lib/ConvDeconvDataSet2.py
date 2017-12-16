
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
class CnnData2: ##### OBS: only works if stride size = filter size of pooling layer
    
    def __init__( self, session, n_in = 128, n_out = 3,
        filterSizeConv1 = 5, nFiltersConv1 = 32, 
        filterSizeConv2 = 5, nFiltersConv2 = 32,
        filterSizeConv3 = 5, nFiltersConv3 = 64, 
        filterSizePool1 = 2, strideFilter1 = 2,
        filterSizePool2 = 2, strideFilter2 = 2,
        filterSizePool3 = 2, strideFilter3 = 2,
        nChannels = 3, fcUnits = 1024, mode = True ):

    	# instantiate session
        self.session   = session
        self.n_in      = n_in  # number of pixels
        self.n_out     = n_out # number of classes
        self.mode      = mode  #True to train
        self.nChannels = nChannels # number of channels (1=grayscale;3=colored)

        # convolution filter sizes
        self.filterSizeConv1 = filterSizeConv1
        self.filterSizeConv2 = filterSizeConv2
        self.filterSizeConv3 = filterSizeConv3
        
        # number of filters of each convolutional layer
        self.nFiltersConv1   = nFiltersConv1
        self.nFiltersConv2   = nFiltersConv2
        self.nFiltersConv3   = nFiltersConv3

        # pooling layer filter sizes
        self.filterSizePool1 = filterSizePool1
        self.filterSizePool2 = filterSizePool2
        self.filterSizePool3 = filterSizePool3

        # pooling layer stride
        self.strideFilter1 = strideFilter1
        self.strideFilter2 = strideFilter2
        self.strideFilter3 = strideFilter3

        # data placeholders
        #self.x    = tf.placeholder(tf.float32, [None, n_in, n_in, nChannels], name='x')
        self.x    = tf.placeholder(tf.float32, [None, int(n_in * n_in * nChannels )], name='x')
        self.y    = tf.placeholder(tf.float32, [None, n_out], name='y')
        #self.x_in = tf.reshape(self.x, [-1, self.n_in * self.n_in])
        self.W_c1 = tf.get_variable( 'W_c1', shape = [ filterSizeConv1, filterSizeConv1, nChannels, nFiltersConv1 ] )
        self.W_c2 = tf.get_variable( 'W_c2', shape = [ filterSizeConv2, filterSizeConv2, nFiltersConv1, nFiltersConv2 ] )
        self.W_c3 = tf.get_variable( 'W_c3', shape = [ filterSizeConv3, filterSizeConv3, nFiltersConv2, nFiltersConv3 ] )

        ##
        ## Network Architecture
        ##

        # Input Layer
        self.input_layer = tf.reshape(self.x, [-1, self.n_in, self.n_in, self.nChannels])

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
        self.pool1 = tf.layers.max_pooling2d(inputs=self.relu1, pool_size=[self.filterSizePool1, self.filterSizePool1], strides=self.strideFilter1)


        #
        # Convolutional Layer #2
        #

        # filter b
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
        self.pool2 = tf.layers.max_pooling2d(inputs=self.relu2, pool_size=[self.filterSizePool2, self.filterSizePool2], strides=self.strideFilter2)


        #
        # Convolutional Layer #3
        #

        # filter
        self.conv3 = tf.nn.conv2d(
            input = self.pool2,
            filter = self.W_c3,
            padding = "SAME",
            strides = [1,1,1,1] )

        # relu
        self.relu3 = tf.nn.relu( self.conv3 )

        #
        # Pooling layer #3
        #
        self.pool3 = tf.layers.max_pooling2d(inputs=self.relu3, pool_size=[self.filterSizePool3, self.filterSizePool3], strides=self.strideFilter3)


        #
        # Dense Layer ---> PARAMETRIZE! change this 7
        #
        nReshape = (self.n_in/filterSizePool1/filterSizePool2/filterSizePool3) * (self.n_in/filterSizePool1/filterSizePool2/filterSizePool3) * nFiltersConv3
        pool3_flat = tf.reshape(self.pool3, [-1, int(nReshape)])
        dense = tf.layers.dense(inputs=pool3_flat, units=fcUnits, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.3, training = self.mode )

        # Logits Layer
        self.logits = tf.layers.dense(inputs=dropout, units = self.n_out)
        self.q      = tf.argmax(input = self.logits, axis = 1) # leave 1?
        
        # Output Layer
        onehot_labels = tf.one_hot( indices = tf.cast(self.y, tf.int32), depth = self.n_out )
        self.loss     = tf.nn.softmax_cross_entropy_with_logits(
            labels = self.y, logits = self.logits )

        self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

    
    # method to compute y given x
    def compute(self, x):
        return self.session.run(self.q, feed_dict={self.x:np.reshape(x,[-1,int(self.n_in*self.n_in*self.nChannels)])})
        #return self.session.run(self.q, feed_dict={self.x:np.reshape(x,[-1, self.n_in, self.n_in, self.nChannels])})
    
    # method to train network
    def train(self, x_batch, y_batch): 
        # take a training step
       #_ = self.session.run(self.train_step, feed_dict={self.x: x_batch, self.y: y_batch})
       _ = self.session.run(self.train_step, feed_dict={self.x:np.reshape(x_batch,[-1,int(self.n_in*self.n_in*self.nChannels)]), self.y: y_batch})

    # acessor method for output after pooling layers
    def getPools(self):
        return ( self.pool1, self.pool2, self.pool3 )

    # acessor method for output after convolutional layers
    def getConvs(self):
    	return ( self.conv1, self.conv2, self.conv3 )

    # acessor method for loss
    def getLoss(self):
        return self.loss

    # acessor method to get filter weights of convolutional layers
    def getWeights(self):
        return (self.W_c1, self.W_c2, self.W_c3 )

    # method to initialize filter weights
    def initWeight(shape):
        weights = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(weights)

    # method to instantiate deconvolutional neural net
    def createDeconvNet(self, inputImage, inputLabel):
        return CnnData2.DeconvData2( self, self.session, inputImage, inputLabel )

    # saver method to save trained cnn in disk 
    def netSaver(self, savePath):
        saver = tf.train.Saver()
        saver.save(self.session, savePath)
        print("Model saved in file: %s" % savePath)

    # loader method to restore weights of a pretrained cnn
    def netLoader(self, loadPath):
        loader = tf.train.Saver({"W_c1":self.W_c1, "W_c2":self.W_c2, "W_c3":self.W_c3})
        restoredModel= loader.restore(self.session, loadPath)
        print("Model restored from %s" % loadPath)


    def activate(self, layer, image, sess):
#    """
#    Within a tensorflow session, calls plotfilter
#    to display the activations of trained filters in a specific layer
#    after passsing an image.
#
#    Parameters
#    ----
#    layer: int
#    image: ndarray of length 784
#    """
        
        
        conv_layer = sess.run(layer, feed_dict={self.x:np.reshape(image, [ 1, 49152], order='F')})
        
        self.plotfilter(conv_layer)
        
        return conv_layer
    
    def plotfilter(self, conv_layer):
#    """
    

#    Parameters
#    ----
#    conv_layer = [?, 28, 28, 32] tensor
#    """
#    
        filters=conv_layer.shape[3]
        plt.figure(1,figsize=(25,25))
        n_columns = 6
        n_rows = math.ceil(filters / n_columns) + 1
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i+1)
            plt.title('Filter ' + str(i))
            plt.imshow(conv_layer[0,:,:,i], interpolation="nearest")




    ###
    ### Nested Class: Deconvolutional Neural Network (CNN) for MNIST
    ###
    class DeconvData2:
        
        def __init__( self, outer, session, inputImage, inputLabel ):

        	# data placeholders
        	#self.inputImage = tf.placeholder(tf.float32, [None, inDim], name='x')
        	#self.inputLabel = tf.placeholder(tf.float32, [None, outDim], name='y')

        	# instantiate outer class in inner class
            self.cnn    = outer
            self.sess   = session

            activations1 = self.calculateActivations( inputImage, inputLabel, 1 )
            self.deconv1 = self.deconvLayer1( inputImage, inputLabel, activations1 )
            activations2 = self.calculateActivations(inputImage, inputLabel, 2)
            self.deconv2 = self.deconvLayer2( inputImage, inputLabel, activations2 )
            activations3 = self.calculateActivations( inputImage, inputLabel, 3 )            
            self.deconv3 = self.deconvLayer3( inputImage, inputLabel, activations3 )


        def deconvLayer1( self, inputImage, inputLabel, activations1 ):

        	#
        	## Deconvoluting 1st layer
        	##
            
            # get activations for layer 1
            #activations1 = self.calculateActivations( inputImage, inputLabel, 1 )

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
                output_shape = [ inputImage.shape[0], self.cnn.n_in, self.cnn.n_in, self.cnn.nChannels ],
                strides = [1, 1, 1, 1],
                padding = "SAME"  )

            return unConv1


        def deconvLayer2( self, inputImage, inputLabel, activations2 ):


            ##
            ## Deconvoluting 2nd layer
            ##

            # get activations for layer 2
            #activations2 = self.calculateActivations(inputImage, inputLabel, 2)

            # convert from array to tensor
            act1_tf = tf.convert_to_tensor( activations2, np.float32 )

            # 1st unpool
            unPool1 = self.unpool( act1_tf )

            # 1st unrelu
            unRelu1 = tf.nn.relu( unPool1 )

            # 1st deconvolute (filter)
            outputShape1 = int(self.cnn.n_in/self.cnn.filterSizePool1)
            unConv1 = tf.nn.conv2d_transpose( 
                #activations1,
                unRelu1,
                self.cnn.W_c2,
                output_shape = [ inputImage.shape[0], outputShape1, outputShape1, self.cnn.nFiltersConv1],
                strides = [1, 1, 1, 1],
                padding = "SAME"  )

            # 2nd unpool
            unPool2 = self.unpool( unConv1 )

            # 2nd relu
            unRelu2 = tf.nn.relu( unPool2 )

            # 2nd deconvolute (filter)
            unConv2 = tf.nn.conv2d_transpose( 
                #activations1,
                unRelu2,
                self.cnn.W_c1,
                output_shape = [ inputImage.shape[0], self.cnn.n_in, self.cnn.n_in, self.cnn.nChannels],
                strides = [1, 1, 1, 1],
                padding = "SAME"  )

            return unConv2


        def deconvLayer3( self, inputImage, inputLabel, activations3 ):


            ##
            ## Deconvoluting 3rd layer
            ##

            # get activations for layer 3
            #activations3 = self.calculateActivations(inputImage, inputLabel, 3)

            # convert from array to tensor
            act1_tf = tf.convert_to_tensor( activations3, np.float32 )

            # 1st unpool
            unPool1 = self.unpool( act1_tf )

            # 1st unrelu
            unRelu1 = tf.nn.relu( unPool1 )

            # 1st deconvolute (filter)
            outputShape1 = int((self.cnn.n_in/self.cnn.filterSizePool2)/self.cnn.filterSizePool1)
            unConv1 = tf.nn.conv2d_transpose( 
                #activations1,
                unRelu1,
                self.cnn.W_c3,
                #output_shape = [ inputImage.shape[0], outputShape1, outputShape1, self.cnn.nFiltersConv1],
                output_shape = [ inputImage.shape[0], outputShape1, outputShape1, self.cnn.nFiltersConv2],
                strides = [1, 1, 1, 1],
                padding = "SAME"  )


            # 2nd unpool
            unPool2 = self.unpool( unConv1 )

            # 2nd relu
            unRelu2 = tf.nn.relu( unPool2 )

            # 2nd deconvolute (filter)
            outputShape2 = int(self.cnn.n_in/self.cnn.filterSizePool1)
            unConv2 = tf.nn.conv2d_transpose( 
                #activations1,
                unRelu2,
                self.cnn.W_c2,
                #output_shape = [ inputImage.shape[0], outputShape2, outputShape2, self.cnn.nChannels],
                output_shape = [ inputImage.shape[0], outputShape2, outputShape2, self.cnn.nFiltersConv1],
                strides = [1, 1, 1, 1],
                padding = "SAME"  )


            # 3rd unpool
            unPool3 = self.unpool( unConv2 )

            # 3rd relu
            unRelu3 = tf.nn.relu( unPool3 )

            # 3rd deconvolute (filter)
            unConv3 = tf.nn.conv2d_transpose( 
                #activations1,
                unRelu3,
                self.cnn.W_c1,
                output_shape = [ inputImage.shape[0], self.cnn.n_in, self.cnn.n_in, self.cnn.nChannels],
                strides = [1, 1, 1, 1],
                padding = "SAME"  )

            return unConv3

        #Returns the random filter activation for layer 1
        def bestActivation1( self, inputImage, inputLabel, n_best=3, k=3):
            activations_layer1=self.calculateActivationsFeature( inputImage, inputLabel, 1)
            random.seed(3)
            filters_layer1=random.sample(range(activations_layer1.shape[-1]),k)
            j=0
            best_index=np.zeros([k,n_best])
            
            all_isolations = np.zeros([k, n_best, 128, 128, 3])
            
            
            
            for i in filters_layer1:
                isolated=activations_layer1.copy()[:,:,:,i]
                
                Norm1 = np.linalg.norm(isolated,axis=(1,2))
                #Norm2 = np.linalg.norm(Norm1, axis=1)
                
                best = np.argsort(Norm1)[-n_best:]
                best_index[j,:]=best
                
                isolated=np.reshape(isolated[best,],(n_best,128,128,3))
                
                all_isolations[j,:,:,:,] = isolated
                j=j+1
                
                
                
            return all_isolations,best_index, filters_layer1
        
        #Returns the random filter activation for layer 2
        def bestActivation2( self, inputImage, inputLabel, n_best=3, k=3):
            activations_layer2=self.calculateActivationsFeature( inputImage, inputLabel, 2)
            random.seed(3)
            filters_layer2=random.sample(range(activations_layer2.shape[-1]),k)
            j=0
            best_index=np.zeros([k,n_best])
            
            all_isolations = np.zeros([k, n_best, 64, 64, 3])
            #print(activations_layer2.shape)
            
            
            for i in filters_layer2:
                isolated=activations_layer2.copy()[:,:,:,i]
                
                Norm1 = np.linalg.norm(isolated,axis=(1,2))
                #Norm2 = np.linalg.norm(Norm1, axis=1)
                
                best = np.argsort(Norm1)[-n_best:]
                best_index[j,:]=best
                
                isolated=np.reshape(isolated[best,],(n_best,64,64,3))
                
                all_isolations[j,:,:,:,] = isolated
                j=j+1
               
               
               
            return all_isolations,best_index, filters_layer2
        
        #Returns the random filter activation for layer 3
        def bestActivation3( self, inputImage, inputLabel, n_best=3, k=3):
            activations_layer3=self.calculateActivationsFeature( inputImage, inputLabel, 3)
            random.seed(5)
            filters_layer3=random.sample(range(activations_layer3.shape[-1]),k)
            j=0
            best_index=np.zeros([k,n_best])
            
            all_isolations = np.zeros([k, n_best, 64, 64, 3])
            #print(activations_layer2.shape)
            
            
            for i in filters_layer3:
                isolated=activations_layer3.copy()[:,:,:,i]
                
                Norm1 = np.linalg.norm(isolated,axis=(1,2))
                #Norm2 = np.linalg.norm(Norm1, axis=1)
                
                best = np.argsort(Norm1)[-n_best:]
                best_index[j,:]=best
                
                isolated=np.reshape(isolated[best,],(n_best,32,32,3))
                
                all_isolations[j,:,:,:,] = isolated
                j=j+1
               
               
               
            return all_isolations,best_index, filters_layer2
        
        #Returns de deconvoluted layer1 as numpy array, with isolated nodes,
        #and save the images on the "img" folder
        def displayFeatures1( self, inputImage, inputLabel, n_best = 3, k = 3):

            #
            ## Deconvoluting 1st layer
            ##
            
            # get activations for layer 1
            activations1 = self.calculateActivations( inputImage, inputLabel, 1 )
            random.seed(3)
            filters = random.sample(range(activations1.shape[-1]), k)
            aux = activations1.shape[0] - n_best
            
            all_isolations = np.zeros([k, n_best, 128, 128, 3])
            j = 0
            best_index = np.zeros([k, n_best])
            
            for i in filters:
            # Isolate filters
                print("Deconvoluting Layer 1 Filter: {}".format(i))
                isolated = activations1.copy()
                isolated[:,:,:,:i]   = 0
                isolated[:,:,:,i+1:] = 0

                Norm1 = np.linalg.norm(isolated[:,:,:,i], axis = (1, 2))
                #Norm2 = np.linalg.norm(Norm1, axis = 1)

                best = np.argsort(Norm1)[-n_best:]

                # devonvolute
                unConv1 = self.deconvLayer1( inputImage, inputLabel, isolated )
                
                u = unConv1.eval()
                
                u = u[best,]
                best_index[j,:] = best
                
                #imsave("img/Deconv1_Node_{}_of_Image1.jpg".format(i), u[0,:,:,:])
                
                all_isolations[j,:,:,:,:] = u
                j = j + 1
            
            return all_isolations, best_index, filters


        def displayFeatures2( self, inputImage, inputLabel, n_best = 3, k = 3):

            ##
            ## Deconvoluting 2nd layer
            ##

            # get activations for layer 2
            activations2 = self.calculateActivations(inputImage, inputLabel, 2)
            random.seed(3)
            filters = random.sample(range(activations2.shape[-1]), k)
            aux = activations2.shape[0] - n_best
            
            all_isolations = np.zeros([k, n_best, 128, 128, 3])
            j = 0
            best_index = np.zeros([k, n_best])
            
            for i in filters:
            # Isolate filters
                print("Deconvoluting Layer 2 Filter: {}".format(i))
                isolated = activations2.copy()
                isolated[:,:,:,:i]   = 0
                isolated[:,:,:,i+1:] = 0
        
                Norm1 = np.linalg.norm(isolated[:,:,:,i], axis = (1, 2))
                #Norm2 = np.linalg.norm(Norm1, axis = 1)
                
                best = np.argsort(Norm1)[-n_best:]
        
                # deconvolute
                unConv2 = self.deconvLayer2( inputImage, inputLabel, isolated )
                
                u = unConv2.eval()
                
                u = u[best,]
                best_index[j,:] = best
                
                #imsave("img/Deconv2_Node_{}_of_Image1.jpg".format(i), u[0,:,:,:])
                
                all_isolations[j,:,:,:,:] = u
                j = j + 1
                
            
            return all_isolations, best_index, filters


        def displayFeatures3( self, inputImage, inputLabel, n_best = 3, k = 3):

            ##
            ## Deconvoluting 2nd layer
            ##

            # get activations for layer 2
            activations3 = self.calculateActivations(inputImage, inputLabel, 3)
            random.seed(3)
            filters = random.sample(range(activations3.shape[-1]), k)
            aux = activations3.shape[0] - n_best
            
            all_isolations = np.zeros([k, n_best, 128, 128, 3])
            j = 0
            best_index = np.zeros([k, n_best])
            
            for i in range(filters):
            # Isolate filters
                if i % 5 == 0:
                    print("Deconvoluting Layer 3 activation number: {}".format(i))
                isolated = activations3.copy()
                isolated[:,:,:,:i]   = 0
                isolated[:,:,:,i+1:] = 0
        
                Norm1 = np.linalg.norm(isolated[:,:,:,i], axis = (1, 2))
                #Norm2 = np.linalg.norm(Norm1, axis = 1)
                
                best =  np.argsort(Norm1)[-n_best:]
        
                # deconvolute
                unConv3 = self.deconvLayer3( inputImage, inputLabel, isolated )
                
                u = unConv3.eval()
                
                u = u[best,]
                best_index[j,:] = best
                
                #imsave("img/Deconv3_Node_{}_of_Image1.jpg".format(i), u[0,:,:,:])
                
                all_isolations[j,:,:,:,:] = u
                j = j + 1
                
            
            return all_isolations, best_index, filters  


        # calculate activations for layer (1 or 2)
        def calculateActivations( self, inputImage, inputLabel, layer ):

            if( layer == 1 ):
                return self.cnn.pool1.eval(feed_dict={self.cnn.x: np.reshape(inputImage,[-1,int(self.cnn.n_in*self.cnn.n_in*self.cnn.nChannels)])})
            elif( layer == 2 ):
                return self.cnn.pool2.eval(feed_dict={self.cnn.x: np.reshape(inputImage,[-1,int(self.cnn.n_in*self.cnn.n_in*self.cnn.nChannels)])})
            else:
                return self.cnn.pool3.eval(feed_dict={self.cnn.x: np.reshape(inputImage,[-1,int(self.cnn.n_in*self.cnn.n_in*self.cnn.nChannels)])})
         
        def calculateActivationsFeature( self, inputImage, inputLabel, layer ):

            if( layer == 1 ):
                return self.cnn.relu1.eval(feed_dict={self.cnn.x: np.reshape(inputImage,[-1,int(self.cnn.n_in*self.cnn.n_in*self.cnn.nChannels)])})
            elif( layer == 2 ):
                return self.cnn.relu2.eval(feed_dict={self.cnn.x: np.reshape(inputImage,[-1,int(self.cnn.n_in*self.cnn.n_in*self.cnn.nChannels)])})
            else:
                return self.cnn.relu3.eval(feed_dict={self.cnn.x: np.reshape(inputImage,[-1,int(self.cnn.n_in*self.cnn.n_in*self.cnn.nChannels)])})


        def getDeconv( self ):
            return self.deconv1, self.deconv2, self.deconv3

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
