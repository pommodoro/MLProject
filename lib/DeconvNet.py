
#####################################################################
#####################################################################
## Feature Visualization MNIST via Deconvolutional Neural Networks ##
#####################################################################
#####################################################################


##
## Load data
##

# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)


###
### Convolutional Neural Network (CNN)
###

# Input Layer
input_layer = tf.reshape(train_data, [-1, 28, 28, 1])

# Convolutional Layer #1
conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Convolutional Layer #2 and Pooling Layer #2
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Pooling layer #2
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# Dense Layer
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(
    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

# Logits Layer
logits = tf.layers.dense(inputs=dropout, units=10)

##
## Train CNN
##


# put code to run CNN


###
### Deconvolutional Network for MNIST
###

#
# unpool function from kvfrans : write source
#

def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat(i, [out, out])
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)


##
## Deconvoluting 1st layer
##

# saves pixel-representations of features from Conv layer 1
featuresReLu1 = tf.placeholder("float",[None,32,32,32]) # change dimensions

# un-relu
unReLu1 = tf.nn.relu(featuresReLu1)

# deconvolute
unConv1 = tf.layers.conv2d_transpose( 
    inputs = unRelu1, 
    filters = wConv1, 
    output_shape=[batchsizeFeatures,imagesize,imagesize,colors] , 
    strides=[1,1,1,1], 
    padding="SAME" )

activations1 = relu1.eval(feed_dict={img: inputImage, lbl: inputLabel, keepProb: 1.0})
print (np.shape(activations1))


##
## Deconvoluting 2nd layer
##

# saves pixel-representations of features from Conv layer 2
featuresReLu2 = tf.placeholder("float",[None,16,16,64]) # change dimensions

# 1st un-relu
unReLu2 = tf.nn.relu(featuresReLu2)

# 1st devonvolution
unConv2 = tf.layers.conv2d_transpose( 
    inputs = unRelu2, 
    filters = 64, 
    #output_shape=[batchsizeFeatures,imagesize,imagesize,colors] , 
    strides=[1,1,1,1], 
    padding="SAME" )

# 1st unpool
unPool = unpool(unConv2)

# 2nd un-relu
unReLu = tf.nn.relu(unPool)

# 2nd deconvolution
unConv = tf.nn.conv2d_transpose(unRelu, wConv1, output_shape=[batchsizeFeatures,imagesize,imagesize,colors] , strides=[1,1,1,1], padding="SAME")

unConv = tf.layers.conv2d_transpose( 
    inputs = unRelu1, 
    filters = wConv1, 
    #output_shape=[batchsizeFeatures,imagesize,imagesize,colors] , 
    strides=[1,1,1,1], 
    padding="SAME" )

# get activations
activations1 = relu2.eval(feed_dict={img: inputImage, lbl: inputLabel, keepProb: 1.0})

# print shape of activations
print (np.shape(activations1))




