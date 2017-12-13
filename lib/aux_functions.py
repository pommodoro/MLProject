
# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import math

# set seed for the whole code
random.seed(234)

##
## function to generate 8x8 pixel letters (E,F,L)
##
def generateLetterData( trainSize, testSize, noise = True, seed = 234 ):

    # function to generate a single letter
    def create_letter(a, add_noise = False):

        if a == 'e':
            aux = [10,11,12,13,18,26,27,28,29,34,42,50,51,52,53]
        if a == 'l':
            aux = [10,18,26,34,42,50,51,52,53]
        if a == 'f':
            aux = [10,11,12,13,18,26,27,28,29,34,42,50,]
            
        letter = np.zeros((8*8))
        letter = letter + 1
        letter[aux] = 0
        
        if add_noise == True:
            #random.seed( seed )
            letter = np.add(letter, -np.random.normal(scale = 0.2, size = 64))
            letter[letter < 0] = 0 
        
        return letter

    ##
    ## Generate train set
    ##

    lists_e=[]
    lists_f=[]
    lists_l=[]
    for i in range( trainSize//3 ):
      lists_e.append(create_letter('e',noise))  #For E
      lists_f.append(create_letter('f',noise))  #For F
      lists_l.append(create_letter('l',noise))  #For L
    inp_e=np.array(lists_e)
    inp_f=np.array(lists_f)
    inp_l=np.array(lists_l)

    #Final data array
    train_data=np.concatenate((inp_e,inp_f,inp_l))  

    #Labels (1 for E, 2 for F, 3 for L)
    train_labels=np.concatenate( (np.tile(0,[trainSize//3]), 
      np.tile(1,[trainSize//3]), np.tile(2,[trainSize//3])) )

    ##
    ## Generate test set
    ##
 
    lists_e=[]
    lists_f=[]
    lists_l=[]
    for i in range( testSize//3 ):
      lists_e.append(create_letter('e',noise))  #For E
      lists_f.append(create_letter('f',noise))  #For F
      lists_l.append(create_letter('l',noise))  #For L
    inp_e=np.array(lists_e)
    inp_f=np.array(lists_f)
    inp_l=np.array(lists_l)

    #Final data array
    test_data=np.concatenate((inp_e,inp_f,inp_l))  

    # Labels (1 for E, 2 for F, 3 for L)
    test_labels=np.concatenate( (np.tile(0,[testSize//3]),
      np.tile(1,[testSize//3]), np.tile(2,[testSize//3])) )

    # return np arrays
    return ( train_data, train_labels , test_data, test_labels )


##
## Class for network from slides: 1 hidden layer with 2 hidden units, logit layer to classify
##
class SlidesNetwork:
    
    def __init__(self, session, n_in , n_out, n_hidden):

        self.session  = session
        self.n_in     = n_in
        self.n_out    = n_out
        self.n_hidden = n_hidden

        # data placeholders
        self.x    = tf.placeholder(tf.float32, [None, n_in], name='x')
        self.y    = tf.placeholder(tf.float32, [None, n_out], name='y')
        self.x_in = tf.reshape(self.x, [-1,self.n_in])

        # Hidden Layer
        self.W_fc1 = tf.get_variable('W_fc1', shape=[self.n_in,self.n_hidden])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.x_in, self.W_fc1), name='layer1')

        # Classification layer
        self.W_fc2 = tf.get_variable('W_fc2', shape=[n_hidden, 3])
        self.b2    = tf.get_variable('b2', shape=[3]) 
        logits     = tf.nn.softmax(tf.add(tf.matmul(self.h_fc1 , self.W_fc2), self.b2, name='logits'))
        self.q     = tf.argmax(input = logits, axis=1)
        
        # Output Layer
        # loss, train_step, etc.
        onehot_labels = tf.one_hot( indices = tf.cast(self.y, tf.int32), depth = 3 )
        self.loss     = tf.nn.softmax_cross_entropy_with_logits(
            labels = self.y, logits = logits )

        #self.loss = tf.reduce_sum(tf.square(self.y - self.q),1)
        self.train_step = tf.train.AdamOptimizer(1e-2).minimize(self.loss)

    
    def compute(self, x):
        # evaluate the network and return the action values [q(s,a=0),q(s,a=1)]
        return self.session.run(self.q, feed_dict={self.x:np.reshape(x,[-1,self.n_in])})
    
    def train(self, x_batch, y_batch):
        # take a training step
       _ = self.session.run(self.train_step, feed_dict={self.x: x_batch, self.y: y_batch})

    def getWeights(self):

        node1 = tf.reshape(self.W_fc1[:,0], [8,8])
        node2 = tf.reshape(self.W_fc1[:,1], [8,8])
        return ( node1, node2 )

    def getLoss(self):
        return self.loss


def activate(layer,image):
    """
    Within a tensorflow session, calls plotfilter
    to display the activations of trained filters in a specific layer
    after passsing an image.

    Parameters
    ----
    layer: int
    image: ndarray of length 784
    """

    conv_layer=sess.run(layer, feed_dict={x:np.reshape(image,[ 1,784], order='F')})
    plotfilter(conv_layer)
    
def plotfilter(conv_layer):
    """
    

    Parameters
    ----
    conv_layer = [?, 28, 28, 32] tensor
    """
    
    filters=conv_layer.shape[3]
    plt.figure(1,figsize=(25,25))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(conv_layer[0,:,:,i], interpolation="nearest", cmap="gray")
