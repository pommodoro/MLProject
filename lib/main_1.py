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
import aux_functions as aux


def main(unused_argv):

  ############################################
  ## Generate letters E,F,L
  ############################################

  train_data, train_labels, test_data, test_labels = aux.generateLetterData( 6000, 1200, True, 234 )

  print("Letter Data Dimensions")
  print("train_data")
  print(train_data.shape)
  print("train_labels")
  print(train_labels.shape)
  print("test")
  print(test_data.shape)
  print("test_labels")
  print(test_labels.shape)

  ############################################
  ## Run session
  ############################################
  with tf.Graph().as_default():
    
    with tf.Session() as sess:


        # instatiate Network
        net = aux.SlidesNetwork(sess, 64, 3, 2)

        # usual tf initialization
        sess.run(tf.global_variables_initializer())

        # make labels one-hot encoded
        onehot_labels_train = tf.one_hot( indices = tf.cast(train_labels, tf.int32), depth = 3 )

        # print MSE before training
        print('error rate BEFORE training is {}'.format((np.sum(net.compute(train_data)!=train_labels) / train_labels.size)))

        # train network
        net.train( train_data, onehot_labels_train.eval() )

        # now train...
        for i in range(3000):
            net.train(train_data,onehot_labels_train.eval())
        
        # print MSE after training
        print('error rate AFTER training is {}'.format(( np.sum(net.compute(train_data)!=train_labels) / train_labels.size)))

        # get weights for each node
        node1, node2 = net.getWeights()

        # plot weights for node 1
        plt.imshow( node1.eval(), cmap = "gray")
        plt.show()

        # plot weights for node 2
        plt.imshow( node2.eval(), cmap = "gray")
        plt.show()



if __name__ == "__main__":
  tf.app.run()

