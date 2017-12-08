import numpy as np
import tensorflow as tf

W_c2 = tf.get_variable(name = 'W_c2', shape = [ 5, 5, 1, 32 ] )

with tf.Session() as sess:
	#sess.run(tf.global_variables_initializer())
	#saver3 = tf.train.import_meta_graph("tmp/titty.meta")
	saver_b = tf.train.Saver({"W_c1": W_c2})
	saver_b.restore(sess, "tmp/cnnMnist")
	#saver3.restore(sess, "./tmp/titty")
	print(W_c2.eval())
	#print(W_c1)