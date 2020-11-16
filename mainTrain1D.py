# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:55:37 2020

@author: Boris
"""

import sys
import numpy as np
import tensorflow as tf
from CNNHelper.CNN1D import CNN1D

from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from Core.DataHandler import DataLoader
import sklearn.model_selection as sk
import matplotlib.pyplot as plt
from sklearn import metrics

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    except RuntimeError as e:
        print(e)

class RealDataEval(tf.keras.callbacks.Callback):
    def __init__(self, x_t,y_t):
        self.TestDataX = x_t
        self.TestDataY = y_t
        
    def on_epoch_end(self, epoch, logs={}):       
        y_pred = self.model.predict(self.TestDataX)
        acc = tf.keras.metrics.CategoricalAccuracy()
        acc.update_state(y_pred, self.TestDataY)
        
        
        ls = tf.keras.metrics.CategoricalCrossentropy()
        ls.update_state(y_pred, self.TestDataY)
        
        print(" - test_acc: "+ str(acc.result().numpy()) + " - test_loss: " + str(ls.result().numpy()))


model  =CNN1D(4)
# opt = keras.optimizers.Adam(learning_rate=0.001)

opt = keras.optimizers.SGD(learning_rate=0.003,momentum=0.01)
model.compile(optimizer =opt, loss="categorical_crossentropy", metrics='accuracy')

model.summary()

mcp_save = keras.callbacks.ModelCheckpoint('mdl_wts_step_fullgenomeStep2-Extra.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')

dt = DataLoader()
# X_Data ,Y_Data  = dt.LoadTrainingData("D:\Sergey\FluorocodeMain\FluorocodeMain\DataForTraining1D.npz")
x_v ,y_v        = dt.LoadTrainingData("D:\Sergey\FluorocodeMain\FluorocodeMain\DataForValidation1D.npz")


history=model.fit(x_v , y_v , batch_size =8, epochs=15, validation_data = (x_v,y_v), callbacks=[mcp_save, RealDataEval(x_v,y_v)])


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
plt.ylim(0,1)
plt.show()