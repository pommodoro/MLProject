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
import os
import csv
from PIL import Image


# Import script with auxiliar functions
import ConvDeconvDataSet2 as nets


def main(unused_argv):

  ############################################
  ## Load Dog, Muffin, Fried Chicken data
  ############################################

  # Load training and eval data
  # mnist        = tf.contrib.learn.datasets.load_dataset("mnist")
  # train_data   = mnist.train.images[0:1000] # Returns np.array - reading only 1000 images
  # train_labels = np.asarray(mnist.train.labels[0:1000], dtype=np.int32) # - reading only 1000 images
  # eval_data    = mnist.test.images[0:200] # Returns np.array - reading only 200 images
  # eval_labels  = np.asarray(mnist.test.labels[0:200], dtype=np.int32) # - reading only 200 images

  ############################################
  ### RESIZE IMAGES
  ############################################

  # # path where files are "../data/dataset2/train"
  # train_path = "../data/dataset2/train/"
  # resized_train_path = "../data/dataset2/train_resized/"
  # files = [train_path + f for f in os.listdir(train_path) if f.endswith('.jpg')]
  # npixels = 128
  # for fl in files:
  #   print("Reading: ")
  #   print( fl )

  #   # define file name
  #   splitName = fl.split("/")
  #   fileName  = splitName[len(splitName)-1]
  #   # open
  #   img = Image.open(fl)
  #   img = img.resize((npixels,npixels), Image.ANTIALIAS)
  #   img.save( resized_train_path + fileName )


  ############################################
  ### Read resized images and store in np.array
  ############################################
  # #path with resized images
  # resized_train_path = "../data/dataset2/train_resized/"
  # files = [resized_train_path + f for f in os.listdir(resized_train_path) if f.endswith('.jpg')]
  # images = []
  # notRGB = []
  # for fl in files:
  #   img = Image.open( fl )
  #   if img.mode == "RGB":
  #     img_array = np.array( img )
  #     print("image and size")
  #     print( fl )
  #     print( img_array.shape )
  #     images.append( img_array )
  #     img.close()
  #   else:
  #     notRGB.append( fl )


  # train_data = np.array( images )
  # notRGB     = np.array( notRGB )

  # print("train_data shape")
  # print(train_data.shape)
  # print("notRGB")
  # print(notRGB)

  # save training data
  # np.save("../data/dataset2/train_data.npy", train_data)
  # np.savetxt("../data/dataset2/train_data_notRGB.txt", notRGB, fmt='%s')


  ###
  ### Load data
  ###

  
  # Load images
  train_data = np.load("../data/dataset2/train_data.npy")
  print(train_data.shape)

  # Load file names that are not RGB (to be excluded from train data)
  indexNot = ['379', '657', '1033', '1124', '1154', '1164', '1201', '1289', '2154', '2287', '2428']

  # Load labels 
  with open("../data/dataset2/label_train.csv", mode='r') as infile:
    reader = csv.reader(infile, delimiter = ";")
    lbls   = {rows[0]:rows[1] for rows in reader}

  # Remove unwanted images
  for k in indexNot:
      lbls.pop(k, None)

  # Put labels in array format
  train_labels = np.array(list(lbls.values()))


  #print( lbls )
  #print( train_labels.shape )




  # print( images.shape )
  #plt.imshow(np.array(a1[1,:,:,0]), cmap='gray')
  #plt.show()

  ############################################
  ## Run session
  ############################################
  with tf.Graph().as_default():
    
    with tf.Session() as sess:


        train_data   = train_data#[0:200]
        train_labels = train_labels#[0:200]
        train_labels = train_labels.astype( np.int )
        batch_size = 50
        max_iter   = 100


        # instatiate Network
        net = nets.CnnData2( sess ) 

        # usual tf initialization
        sess.run(tf.global_variables_initializer())

        # make labels one-hot encoded
        onehot_labels_train = tf.one_hot( indices = tf.cast(train_labels, tf.int32), depth =  3 ).eval()

        # print error rate before training
        print('error rate BEFORE training is {}'.format((np.sum(net.compute(train_data)!=train_labels) / train_labels.size)))

        # train network
        net.train( train_data, onehot_labels_train )

        # now train...
        for i in range( max_iter ):

          batch = random.sample( range(train_data.shape[0]), batch_size )
          x_batch = train_data[batch,]
          y_batch = onehot_labels_train[batch,]
          net.train( x_batch,  y_batch )
          if i % 10 == 0:
            print('error rate during training is {}'.format(( np.sum(net.compute(train_data)!=train_labels) / train_labels.size)))
          
        
  #       ##
  #       ## Deconvolution Part - until here it runs OK
  #       ##
        
  #       # instantiate deconv net
  #       DeconvNet = net.createDeconvNet( train_data, train_labels )

  #       print( "\nDimension of input data")
  #       print( train_data.shape)

  #       print( "\nNumber of Images")
  #       print( train_data.shape[0])

  #       dec1, dec2  = DeconvNet.getDeconv()

  #       print( "\nDimension of Deconvoluted images - Layer 1")
  #       print( dec1.shape )
  #       print( dec1 )

  #       print( "\nDimension of Deconvoluted images - Layer 2")
  #       print( dec2.shape )
  #       print( dec2 )

  #       a1 = dec2.eval()
  #       plt.imshow(np.array(a1[1,:,:,0]), cmap='gray')
  #       plt.show()


  #       # plt.imshow(np.array(train_data[1,:,:,0]), cmap='gray')
  #       # plt.show()


if __name__ == "__main__":
  tf.app.run()

