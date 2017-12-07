import numpy as np
import tensorflow as tf

with tf.Session() as sess:
	tf.train.Saver.restore(sess, "/tmp/cnn_mnist.ckpt")
	print("Model restored")

	print('error rate during training is {}'.format(( np.sum(net.compute(train_data)!=train_labels) / train_labels.size)))