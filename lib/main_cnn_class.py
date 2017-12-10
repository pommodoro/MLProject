##
## main 
##


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import script with auxiliar functions
import ConvDeconvClassesMnist as nets


def main(unused_argv):

  ############################################
  ## Load MNIST data
  ############################################

  # Load training and eval data
  mnist        = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data   = mnist.train.images[0:1000] # Returns np.array - reading only 1000 images
  train_labels = np.asarray(mnist.train.labels[0:1000], dtype=np.int32) # - reading only 1000 images
  eval_data    = mnist.test.images[0:200] # Returns np.array - reading only 200 images
  eval_labels  = np.asarray(mnist.test.labels[0:200], dtype=np.int32) # - reading only 200 images


  ############################################
  ## Run session
  ############################################
  with tf.Graph().as_default():
    
    with tf.Session() as sess:


        # instatiate Network
        net = nets.CnnMnist(sess, 28*28, 10, True ) # True stands for training

        #DeconvNet = net.createDeconvNet()

        #act11  = DeconvNet.calculateActivations(train_data, train_labels, 1)
        #plt.imshow(np.array(act11[5,:,:,2]), cmap='gray')
        #plt.show()

        # usual tf initialization
        sess.run(tf.global_variables_initializer())

        # make labels one-hot encoded
        onehot_labels_train = tf.one_hot( indices = tf.cast(train_labels, tf.int32), depth =  10 )

        # print error rate before training
        print('error rate BEFORE training is {}'.format((np.sum(net.compute(train_data)!=train_labels) / train_labels.size)))

        # train network
        net.train( train_data, onehot_labels_train.eval() )

        # now train...
        for i in range(15):
            net.train(train_data,onehot_labels_train.eval())
            print('error rate during training is {}'.format(( np.sum(net.compute(train_data)!=train_labels) / train_labels.size)))
        
        # save the trained network 
        net.netSaver("./tmp/cnnMnist")

        #''' DON'T COMMENT ME PLEASE!!!
        ##
        ## Deconvolution Part - until here it runs OK
        ##
        
        # instantiate deconv net
        DeconvNet = net.createDeconvNet( train_data, train_labels )

        print( "\nDimension of input data")
        print( train_data.shape)

        print( "\nNumber of Images")
        print( train_data.shape[0])

        dec1, dec2  = DeconvNet.getDeconv()

        print( "\nDimension of Deconvoluted images - Layer 1")
        print( dec1.shape )
        print( dec1 )

        print( "\nDimension of Deconvoluted images - Layer 2")
        print( dec2.shape )
        print( dec2 )

        #a1 = dec1.eval()
        #plt.imshow(np.array(a1[1,:,:,0]), cmap='gray')
        #plt.show()

        #conv1, conv2 = net.getConvs()
        #a1 = conv1.eval(feed_dict = {net.x: train_data})
        
        print("\n")
        a1 = DeconvNet.displayFeatures1(train_data, train_labels)
        print(a1.shape)
        
        print("\n")
        a2 = DeconvNet.displayFeatures2(train_data, train_labels)
        print(a2.shape)      
        
        plt.imshow(np.array(a1[0, 1, :, :, 0]), cmap='gray')
        plt.imshow(np.array(a2[0, 1, :, :, 0]), cmap='gray')
        plt.show()

        # plt.imshow(np.array(train_data[1,:,:,0]), cmap='gray')
        # plt.show()

#''' DON'T COMMENT ME PLEASE!!!
if __name__ == "__main__":
  tf.app.run()

