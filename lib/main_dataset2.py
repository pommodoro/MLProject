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
import random
import time


# Import script with auxiliar functions
import ConvDeconvDataSet2 as nets


def main(unused_argv):

  ############################################
  ## Load Dog, Muffin, Fried Chicken data
  ############################################

  # Load training and eval data
#   mnist        = tf.contrib.learn.datasets.load_dataset("mnist")
#   train_data   = mnist.train.images[0:1000] # Returns np.array - reading only 1000 images
#   train_labels = np.asarray(mnist.train.labels[0:1000], dtype=np.int32) # - reading only 1000 images
#   eval_data    = mnist.test.images[0:200] # Returns np.array - reading only 200 images
#   eval_labels  = np.asarray(mnist.test.labels[0:200], dtype=np.int32) # - reading only 200 images

  #########################################
  ##RESIZE IMAGES
  #########################################


   # path where files are "../data/dataset2/train"
#   train_path = "data/dataset2/train/"
#   resized_train_path = "data/dataset2/train_resized/"
#   files = [train_path + f for f in os.listdir(train_path) if f.endswith('.jpg')]
#   npixels = 128
#   for fl in files:
#     print("Reading: ")
#     print( fl )
#
#     # define file name
#     splitName = fl.split("/")
#     fileName  = splitName[len(splitName)-1]
#     # open
#     img = Image.open(fl)
#     img = img.resize((npixels,npixels), Image.ANTIALIAS)
#     img.save( resized_train_path + fileName )
#
#
#   ############################################
#   ### Read resized images and store in np.array
#   ############################################
#   #path with resized images
#   resized_train_path = "data/dataset2/train_resized/"
#   files = [resized_train_path + f for f in os.listdir(resized_train_path) if f.endswith('.jpg')]
#   images = []
#   images_flip = []
#   images_rot  = []
#   notRGB = []
#   i = 0
#   for fl in files:
#
#     # for seed
#     i = i + 1
#
#     # open image
#     img = Image.open( fl )
#
#     if img.mode == "RGB":
#
#       # flip image
#       flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
#       img_flp = np.array( flipped )
#       #img_flp = np.multiply( img_flp, 1.0 / 255.0 )
#       images_flip.append( img_flp )
#
#
#       # rotate image
#       random.seed(i)
#       angle = np.random.uniform(-15,15)
#       rotated = img.rotate( angle )
#       img_rot = np.array( rotated )
#       #img_rot = np.multiply( img_rot, 1.0 / 255.0 )
#       images_rot.append( img_rot )
#
#
#       # regular image
#       img_array = np.array( img )
#       #img_array = np.multiply( img_array, 1.0 / 255.0 )
#       images.append( img_array )
#
#       # close image
#       img.close()
#
#     else:
#       notRGB.append( fl )
#
#
#   train_normal  = np.array( images )
#   train_flipped = np.array( images_flip )
#   train_rotated = np.array( images_rot )
#   notRGB        = np.array( notRGB )
#
#   print("train_normal shape")
#   print(train_normal.shape)
#   print("train_flipped shape")
#   print(train_flipped.shape)
#   print("train_rotated shape")
#   print(train_rotated.shape)
#   print("notRGB files")
#   print(notRGB)
#
#   #save training data (3 np arrays)
#   np.save("data/dataset2/train_data_normal.npy", train_normal )
#   np.save("data/dataset2/train_data_flipped.npy", train_flipped )
#   np.save("data/dataset2/train_data_rotated.npy", train_rotated )
#   np.savetxt("data/dataset2/train_data_notRGB.txt", notRGB, fmt='%s')


  ###
  ### Load data
  ###

  
  # Load images
  print("Loading images...")
  train_normal  = np.load("data/dataset2/train_data_normal.npy" )
  train_flipped = np.load("data/dataset2/train_data_flipped.npy" )
  train_rotated = np.load("data/dataset2/train_data_rotated.npy" )

  # stacking in a single np.array
  train_all = np.concatenate( (train_normal, train_flipped, train_rotated) )

  # normalize
  print("Normalizing images...")
  train_all = train_all[ None:, :, :, :,] * (1.0 / 255.0)

  # normalize all values 
  # train_all = np.multiply( train_all[0:0,:,:,:], 1/255.0 )

  # Load file names that are not RGB (to be excluded from train data)
  indexNot = ['379', '657', '1033', '1124', '1154', '1164', '1201', '1289', '2154', '2287', '2428']

  # Load labels 
  with open("data/dataset2/label_train.csv", mode='r') as infile:
    reader = csv.reader(infile, delimiter = ";")
    lbls   = {rows[0]:rows[1] for rows in reader}

  # Remove unwanted images
  for k in indexNot:
      lbls.pop(k, None)


  print("Separating train and test sets")
  random.seed(1)
  subset = random.sample(range(8967), 896 ) # 3 * 2989 = 8967 total input size

  train_data = np.delete(train_all, subset, 0)
  eval_data = train_all[subset,]
  
  print("Train data dimension:", train_data.shape)
  print("Test data dimension:", eval_data.shape)

  # Put labels in array format
  labels_load = np.array(list(lbls.values()))
  labels_all  = np.concatenate( (labels_load, labels_load, labels_load) )

  
  train_labels = np.delete(labels_all, subset, 0)
  eval_labels = labels_all[subset,]
  
  print("Train labels dimension:", train_labels.shape)
  print("Test labels dimension:", eval_labels.shape)


#   # print( images.shape )
#   #plt.imshow(np.array(a1[1,:,:,0]), cmap='gray')
#   #plt.show()

  ############################################
  ## Run session
  ############################################
  with tf.Graph().as_default():
    
    with tf.Session() as sess:

         train_labels = train_labels.astype( np.int )
         eval_labels = eval_labels.astype( np.int )
         batch_size   = 48
         max_iter     = 1000


#         # instatiate Network
#         net = nets.CnnData2( sess, filterSizeConv1 = 3, nFiltersConv1 = 32, 
#        filterSizeConv2 = 3, nFiltersConv2 = 32,
#        filterSizeConv3 = 3, nFiltersConv3 = 64,
#        fcUnits = 256 ) 
#
#         # usual tf initialization
#         sess.run(tf.global_variables_initializer())
#
#         # make labels one-hot encoded
#         onehot_labels_train = tf.one_hot( indices = tf.cast(train_labels, tf.int32), depth =  3 ).eval()
#
#         # print error rate before training
#         #print('error rate BEFORE training is {}'.format((np.sum(net.compute(train_data)!=train_labels) / train_labels.size)))
#
#         # train network
#         #net.train( train_data, onehot_labels_train )
#
#         # now train...
#
#         print("Started training...")
#         start = time.time()
#         
#         for i in range( max_iter ):
#
#           batch = random.sample( range(train_data.shape[0]), batch_size )
#           x_batch = train_data[batch]
#           y_batch = onehot_labels_train[batch]
#           net.train( x_batch,  y_batch )
#           if i % 100 == 0:
#             print('Aproximate error rate at iteration {} is {} %'.format(i, ( np.sum(net.compute(train_data[0:100,])!=train_labels[0:100]) / train_labels[0:100].size *100)))
#
#         elapsed = (time.time() - start)
#         
#         print("Total training time: {} seconds".format(round(elapsed,2)))
#            
#         print('Aproximate Final training error is {} %'.format((np.sum(net.compute(train_data[0:500])!=train_labels[0:500]) / train_labels[0:500].size *100)))
#        
#         print('Test error is {} %'.format(( np.sum(net.compute(eval_data)!=eval_labels) / eval_labels.size *100)))
#
#
#        # # save the trained network 
#         net.netSaver("./tmp/DMF/cnnData2")
          
        
        ##
        ## Deconvolution Part - until here it runs OK
        ##
        
        # load trained model
         net = nets.CnnData2( sess, filterSizeConv1 = 3, nFiltersConv1 = 32, 
                             filterSizeConv2 = 3, nFiltersConv2 = 32, filterSizeConv3 = 3, 
                             nFiltersConv3 = 64, fcUnits = 256  ) 
         
         net.netLoader( "./tmp/DMF/cnnData2" )

         print( "loaded model" )


         # instantiate deconv net
         DeconvNet = net.createDeconvNet( eval_data[:100,], eval_labels[:100] )

         print( "\nDimension of input data")
         print( train_data.shape)

         print( "\nNumber of Images")
         print( train_data.shape[0])

         dec1, dec2, dec3  = DeconvNet.getDeconv()

         print( "\nDimension of Deconvoluted images - Layer 1")
         print( dec1.shape )
         print( dec1 )

         print( "\nDimension of Deconvoluted images - Layer 2")
         print( dec2.shape )
         print( dec2 )

         print( "\nDimension of Deconvoluted images - Layer 3")
         print( dec3.shape )
         print( dec3 )
#
#        #a1 = dec1.eval()
#        #plt.imshow(np.array(a1[0,:,:,0]) )
#        #plt.show()
#        
#        #Activations part----------------------------
#
         np.save("tmp/DMF/eval_data_dmf.npy", eval_data[:100,])
         
         print("\n")
         a1, b1, c1 = DeconvNet.displayFeatures1(eval_data[:100,], eval_labels[:100], n_best = 10, k = 9)
         np.save("tmp/DMF/ActivationsDMF_Layer1.npy", a1)
         np.save("tmp/DMF/BestImagesDMF_Layer1.npy", b1)
         np.save("tmp/DMF/RandomFiltersIndex_Layer1.npy", c1)
         print(a1.shape)
        
         print("\n")
         a2, b2, c2 = DeconvNet.displayFeatures2(eval_data[:100,], eval_labels[:100], n_best = 10, k = 9)
         np.save("tmp/DMF/ActivationsDMF_Layer2.npy", a2)
         np.save("tmp/DMF/BestImagesDMF_Layer2.npy", b2)
         np.save("tmp/DMF/RandomFiltersIndex_Layer2.npy", c2)
         print(a2.shape)  
        
         print("\n")
         a3, b3, c3 = DeconvNet.displayFeatures2(eval_data[:100,], eval_labels[:100], n_best = 10, k = 9)
         np.save("tmp/DMF/ActivationsDMF_Layer3.npy", a3)
         np.save("tmp/DMF/BestImagesDMF_Layer3.npy", b3)
         np.save("tmp/DMF/RandomFiltersIndex_Layer3.npy", c3)
         print(a3.shape)     
        
         #Weights part--------------------------------
        
         w1_t, w2_t, w3_t = net.getWeights()
        
         w1 = w1_t.eval()
         np.save("tmp/DMF/WeightDMF_1.npy", w1)
        
         w2 = w2_t.eval()
         np.save("tmp/DMF/WeightDMF_2.npy", w2)
        
         w3 = w3_t.eval()
         np.save("tmp/DMF/WeightDMF_3.npy", w3)

         # plt.imshow(np.array1(train_data[1,:,:,0]), cmap='gray')
         # plt.show()


if __name__ == "__main__":
  tf.app.run()

