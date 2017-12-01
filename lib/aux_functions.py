
# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline


# function with 2 layer fully connected network from slides
def slides_network(features, labels, mode):

    # number of units in hidden layer
    n_hidden = 2
    n_in =  64 #features.shape[1]
    n_out = 3 # 3 classes E,F,L

    ## DOUBT: x dimension should be 6000x64 or 1x64?

    # define placeholder for input/output
    x = tf.placeholder(tf.float32, features["x"].shape, name='x')
    y = tf.placeholder(tf.float32, labels.shape, name='y')

    # Hidden Layer
    W_fc1 = tf.get_variable( 'W_fc1', shape=[ n_in, n_hidden ] )
    h_fc1 = tf.nn.relu( tf.matmul( x , W_fc1 ), name='layer1' )

    # Logits Layer
    logits = tf.layers.dense(inputs=h_fc1, units = n_out)

    # Define predictions
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth = 3)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# function to generate 8x8 pixel letters (E,F,L)

def generateLetterData( trainSize, testSize ):

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
      lists_e.append(create_letter('e',True))  #For E
      lists_f.append(create_letter('f',True))  #For F
      lists_l.append(create_letter('l',True))  #For L
    inp_e=np.array(lists_e)
    inp_f=np.array(lists_f)
    inp_l=np.array(lists_l)

    #Final data array
    train_data=np.concatenate((inp_e,inp_f,inp_l))  

    #Labels (1 for E, 2 for F, 3 for L)
    train_labels=np.concatenate( (np.tile(1,[trainSize//3]), 
      np.tile(2,[trainSize//3]), np.tile(3,[trainSize//3])) )

    ##
    ## Generate test set
    ##
 
    lists_e=[]
    lists_f=[]
    lists_l=[]
    for i in range( testSize//3 ):
      lists_e.append(create_letter('e',True))  #For E
      lists_f.append(create_letter('f',True))  #For F
      lists_l.append(create_letter('l',True))  #For L
    inp_e=np.array(lists_e)
    inp_f=np.array(lists_f)
    inp_l=np.array(lists_l)

    #Final data array
    test_data=np.concatenate((inp_e,inp_f,inp_l))  

    # Labels (1 for E, 2 for F, 3 for L)
    test_labels=np.concatenate( (np.tile(1,[testSize//3]),
      np.tile(2,[testSize//3]), np.tile(3,[testSize//3])) )

    # return np arrays
    return ( train_data, train_labels , test_data, test_labels )



# class Network:
    
#     def __init__(self, session, n_in , n_out):

#         self.session = session
#         self.n_in = n_in
#         self.n_out = n_out
#         self.n_hidden = 2

#         # data placeholders
#         self.x = tf.placeholder(tf.float32, [None, n_in], name='x')
#         self.y = tf.placeholder(tf.float32, [None, n_out], name='y')
#         self.x_in = tf.reshape(self.x, [-1,self.n_in])

#         # 2 layer network
#         self.W_fc1 = tf.get_variable('W_fc1', shape=[self.n_in,self.n_hidden])
#         self.b_fc1 = tf.get_variable('b_fc1', shape=[self.n_hidden])
#         self.h_fc1 = tf.nn.relu(tf.add(tf.matmul(self.x_in, self.W_fc1), self.b_fc1, name='layer1'))
#         self.W_fc2 = tf.get_variable('W_fc2', shape=[self.n_hidden,self.n_out])
#         self.b_fc2 = tf.get_variable('b_fc2', shape=[self.n_out])
#         self.q = tf.add(tf.matmul(self.h_fc1, self.W_fc2), self.b_fc2, name='layer2')
#         # loss, train_step, etc.
#         self.loss = tf.reduce_sum(tf.square(self.y - self.q),1)
#         self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
    
#     def compute(self, x):
#         # evaluate the network and return the action values [q(s,a=0),q(s,a=1)]
#         return self.session.run(self.q, feed_dict={self.x:np.reshape(x,[-1,self.n_in])})
    
#     def train(self, x_batch, y_batch):
#         # take a training step
#         _ = self.session.run(self.train_step, feed_dict={self.x: x_batch, self.y: y_batch})


# class SlidesNetwork:
    
#     def __init__(self, session, n_in , n_out):

#         self.session = session
#         self.n_in = n_in
#         self.n_out = n_out
#         self.n_hidden = 2

#         # data placeholders
#         self.x = tf.placeholder(tf.float32, [None, n_in], name='x')
#         self.y = tf.placeholder(tf.float32, [None, n_out], name='y')
#         self.x_in = tf.reshape(self.x, [-1,self.n_in])

#         # Hidden Layer
#         self.W_fc1 = tf.get_variable('W_fc1', shape=[self.n_in,self.n_hidden])
#         self.h_fc1 = tf.nn.relu(tf.matmul(self.x_in, self.W_fc1), name='layer1')
        
#         # Output Layer


#         self.W_fc2 = tf.get_variable('W_fc2', shape=[self.n_hidden,self.n_out])
#         self.b_fc2 = tf.get_variable('b_fc2', shape=[self.n_out])
#         self.q = tf.add(tf.matmul(self.h_fc1, self.W_fc2), self.b_fc2, name='layer2')
#         # loss, train_step, etc.
#         self.loss = tf.reduce_sum(tf.square(self.y - self.q),1)
#         self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
    
#     def compute(self, x):
#         # evaluate the network and return the action values [q(s,a=0),q(s,a=1)]
#         return self.session.run(self.q, feed_dict={self.x:np.reshape(x,[-1,self.n_in])})
    
#     def train(self, x_batch, y_batch):
#         # take a training step
#        _ = self.session.run(self.train_step, feed_dict={self.x: x_batch, self.y: y_batch})