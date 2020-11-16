# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:30:11 2020

@author: Sergey
"""

import tensorflow as tf
import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import CNNHelper.NNBuilder as nn

def CNN1D(classes):
    input = layers.Input(shape = (250,1))
    x = layers.Conv1D(128,13,padding='same',activation='relu')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128,13,padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    
    x = layers.Conv1D(256,7,padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256,7,padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    
    
    x = layers.Conv1D(512,5,padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(512,5,padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(512,5,padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(512,5,padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool1D(pool_size=2)(x)   
    
    # x = layers.Conv1D(256,1,padding='same',activation='relu')(x)
  
    
    # x = layers.Conv1D(512,3,padding='same',activation='relu')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Conv1D(512,3,padding='same',activation='relu')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.MaxPool1D(pool_size=2)(x)   


    # x = layers.Conv1D(512,3,padding='same',activation='relu')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Conv1D(512,3,padding='same',activation='relu')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.MaxPool1D(pool_size=2)(x)   
    
    
    # x = layers.Conv1D(512,3,padding='same',activation='relu')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Conv1D(512,3,padding='same',activation='relu')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.MaxPool1D(pool_size=2)(x)  
    
    x = layers.Flatten()(x)
    x = layers.Dense(units=128)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=64)(x)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(units=classes, activation = 'softmax')(x)
    
    CNN = tf.keras.Model(input,output,name='CNN1D')
    return CNN