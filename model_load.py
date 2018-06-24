# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:01:16 2018

@author: Shreya
"""

from keras.models import load_model
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def custom_activation(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})
model=load_model('final_baseline0.001.h5')
model.summary()