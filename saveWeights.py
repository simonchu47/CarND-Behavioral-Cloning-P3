#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:53:42 2017

@author: simon
"""

from keras.models import load_model
#from keras.utils import plot_model
from keras.utils.visualize_util import plot

model = load_model('model.h5')

#plot_model(model, to_file='model.png')
plot(model, to_file='model.png')

model.save_weights('my_model_weights.h5')
