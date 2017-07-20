#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 09:53:42 2017

@author: simon
"""
import os
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

log_path = '/media/diskC/train_selfdriving/reTrain/'
log_files = []
for x in os.listdir(log_path):
    log_files.append(log_path+x+'/driving_log.csv')

print(log_files)

samples = []
for log in log_files:
    print(log)
    with open(log) as f:
        f.readline() # Strip the header\n",
        for line in f:
            samples.append(line + ', ' + log.split('/driving_log')[0])

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

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
                
                name = batch_sample.split(', ')[-1]+'/IMG/'+batch_sample.split(',')[0].split('/')[-1]
                #center_image = cv2.imread(name)
                center_image = imread(name)
                
                center_angle = float(batch_sample.split(',')[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
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

# TODO: Build the Final Test Neural Network in Keras Here
ch, row, col = 3, 90, 320 #trimmed image
model = Sequential()
# Cropping to 90x320x3
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

#model.add(Lambda(lambda x: x / 255.0 - 0.5,
 #                input_shape=(ch, row, col), output_shape=(ch, row, col)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
                 
# Layer 1: 43x158x24
#model.add(Convolution2D(24, 5, 5, input_shape=(160, 320, 3)))
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
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.load_weights('my_model_weights.h5')

#model.compile('adam', 'categorical_crossentropy', ['accuracy'])
model.compile(optimizer='adam', loss='mse')
#history = model.fit(X_train, y_one_hot, nb_epoch=10, validation_split=0.2, shuffle='true')
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('retrainModel.h5')
exit()    
