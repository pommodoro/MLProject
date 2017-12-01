
## main 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

# Import script with auxiliar functions
import aux_functions as aux


def main(unused_argv):

  ############################################
  ## Generate letters E,F,L
  ############################################

  train_data, train_labels, test_data, test_labels = aux.generateLetterData( 6000, 1200 )

  print("Letter Data Dimensions")
  print("train_data")
  print(train_data.shape)
  print("train_labels")
  print(train_labels.shape)
  print("test")
  print(test_data.shape)
  print("test_labels")
  print(test_labels.shape)


  # Create the Estimator
  net_classifier = tf.estimator.Estimator(
    model_fn=aux.slides_network, model_dir="/tmp/slides_model")

  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
  net_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_data},
    y=test_labels,
    num_epochs=1,
    shuffle=False)
  eval_results = net_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
  

if __name__ == "__main__":
  tf.app.run()

