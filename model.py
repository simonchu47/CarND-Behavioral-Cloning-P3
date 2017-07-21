#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 09:53:42 2017

@author: simon
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.python.control_flow_ops = tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('log_path', '', "The directory containing training data")
flags.DEFINE_bool('re_training', False, "New creating or re-training the model")

#Parcing path from the command line flag to find out all subdirectories that
#contain the recorded data
log_path = FLAGS.log_path + '/'
log_files = []
for x in os.listdir(log_path):
    log_files.append(log_path+x+'/driving_log.csv')

#Parcing all the driving_log.csv to add all the images path to the list
#including the current path 
samples = []
for log in log_files:
    print(log)
    with open(log) as f:
        f.readline() # Strip the header\n",
        for line in f:
            samples.append(line + ', ' + log.split('/driving_log')[0])

#Spliting the data into training and validation data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Using generator to read in the training images and steering angle data
from scipy.misc import imread
from random import shuffle
import sklearn
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                
                #Using the center images of the recorded data
                name = batch_sample.split(', ')[-1]+'/IMG/'+batch_sample.split(',')[0].split('/')[-1]                
                center_image = imread(name)
                
                center_angle = float(batch_sample.split(',')[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D, Lambda

def main(__):
    
    
    # TODO: Build the Final Test Neural Network in Keras Here
    model = Sequential()
    # Cropping to 90x320x3 that only with the view of roads
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

    #Normalization of the images
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
                 
    # Layer 1: 43x158x24
    model.add(Convolution2D(24, 5, 5))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    # Layer 2: 20x77x36
    model.add(Convolution2D(36, 5, 5))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    #Layer 3: 8x37x48
    model.add(Convolution2D(48, 5, 5))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    #Layer 4: 6x35x64
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    #Layer 5: 4x33x64
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Flatten())
    #Layer 6: fully connected layer
    model.add(Dense(100))
    #Layer 7: fully connected layer
    model.add(Dense(50))
    #Layer 8: fully connected layer
    model.add(Dense(10))
    #Layer 9: fully connected layer
    model.add(Dense(1))
    
    #Reading from the command flag to decide wether transfer learning or not
    if FLAGS.re_training is True:
        model.load_weights('my_model_weights.h5')
        print('The weights of the model has been loaded from my_model_weights.h5...')
    
    #Using adam optimizer
    model.compile(optimizer='adam', loss='mse')
    
    history = model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=2, verbose=1)

    ### print the keys contained in the history object
    print(history.history.keys())
    ### plot the training and validation loss for each epoch

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    
    #Save the model and its weights
    model.save('model.h5')
    model.save_weights('my_model_weights.h5')
    exit()

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()    
