
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline

tf.logging.set_verbosity(tf.logging.INFO)


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
        #letter[letter < 0] = 0 
    
    return letter

def model_from_slides(features, labels, mode):

  """Model function for Neural Network from slides."""
  
  # Input Layer
  input_flat = tf.reshape(features["x"], [-1, 8 * 8 * 1])

  # flatten input
  # Layer #1: fully connected
  dense = tf.layers.dense(inputs = input_flat, units = 2, activation = tf.nn.relu)

  # Logits Layer
  logits = tf.layers.dense(inputs=dense, units=3)

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
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
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


def main(unused_argv):

  # Load training and eval data
  
  # generate e's
  for i in range(2000):


  # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  # train_data = mnist.train.images # Returns np.array
  # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  # eval_data = mnist.test.images # Returns np.array
  # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # # Create the Estimator
  # mnist_classifier = tf.estimator.Estimator(
  #   model_fn = model_from_slides, model_dir = "/tmp/model_from_slides")

  # # Set up logging for predictions
  # tensors_to_log = {"probabilities": "softmax_tensor"}
  # logging_hook = tf.train.LoggingTensorHook(
  #     tensors=tensors_to_log, every_n_iter=50)

  # # Train the model
  # train_input_fn = tf.estimator.inputs.numpy_input_fn(
  #   x={"x": train_data},
  #   y=train_labels,
  #   batch_size=100,
  #   num_epochs=None,
  #   shuffle=True)
  # mnist_classifier.train(
  #   input_fn=train_input_fn,
  #   steps=20000,
  #   hooks=[logging_hook])

  # # Evaluate the model and print results
  # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
  #   x={"x": eval_data},
  #   y=eval_labels,
  #   num_epochs=1,
  #   shuffle=False)
  # eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  # print(eval_results)


if __name__ == "__main__":
  tf.app.run()



