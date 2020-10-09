# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 09:20:26 2020

@author: Boris
"""

import tensorflow as tf
import numpy as np

def weighted_categorical_crossentropy(y_true, y_pred):
    w = tf.constant([0.05 , 0.95 , 0.95])
    cross_entropy = tf.reduce_mean( -tf.reduce_sum(w*y_true*tf.math.log(tf.clip_by_value(y_pred,1e-10,1.0)),axis = -1))
    return cross_entropy