# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:02:31 2020

@author: Sergey
"""
import sys
import numpy as np
import tensorflow as tf
from CNNHelper.ResNET import ResNet50
from CNNHelper.VGG19 import VGG19

from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from Core.DataHandler import DataLoader
import sklearn.model_selection as sk
import matplotlib.pyplot as plt


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    except RuntimeError as e:
        print(e)
# def weighted_accuracy(y_true, y_pred):
#     w = tf.constant([0, 1 , 1])
    
    
#     # cross_entropy = tf.reduce_mean( -tf.reduce_sum(w*y_true*tf.math.log(tf.clip_by_value(y_pred,1e-10,1.0)),axis = -1))
#     return acc


def weighted_categorical_crossentropy(y_true, y_pred):
    w = tf.constant([1/0.92, 1/0.04 , 1/0.04])
    cross_entropy = tf.reduce_mean( -tf.reduce_sum(w*y_true*tf.math.log(tf.clip_by_value(y_pred,1e-10,1.0)),axis = -1))
    return cross_entropy

model  =VGG19(3)
# model  =ResNet50(3)
opt = keras.optimizers.Adam(learning_rate=0.0001)

# opt = keras.optimizers.SGD(learning_rate=1,momentum=0.001)
model.compile(optimizer =opt, loss="categorical_crossentropy", metrics='accuracy')
# model.compile(optimizer =opt, loss=weighted_categorical_crossentropy, metrics='accuracy')

model.summary()


dt = DataLoader()
X_Data ,Y_Data  = dt.LoadTrainingData("D:\Sergey\FluorocodeMain\FluorocodeMain\DataForTraining.npz")
x_v ,y_v        = dt.LoadTrainingData("D:\Sergey\FluorocodeMain\FluorocodeMain\DataForValidation.npz")


history=model.fit(X_Data , Y_Data , batch_size =4, shuffle=True,  epochs=100, validation_data = (x_v,y_v))


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ylim(0.8,1)
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ylim(0,0.3)
plt.show()