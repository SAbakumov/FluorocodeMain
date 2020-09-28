# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:23:07 2020

@author: Sergey
"""
import numpy as np
import cv2
import tensorflow as tf


def ReturnInputAndLabelImage(directoryLabel,directoryInput):
    training_labels= cv2.imread(directoryLabel,-1)
    training_labels= np.sum(training_labels,axis=2)
    training_labels[~(training_labels==765)] = 1
    training_labels[(training_labels==765)] = 0
#    numclass = 2;
#    PerClassTraining_labels = np.empty([numclass,512,512])
#    for i in range(0,numclass-1):
#        PerClassTraining_labels[i,:,:]= training_labels
#    
    
    
    training_data = cv2.imread(directoryInput,-1)/65535
    
#    return training_labels


    return  np.reshape(training_data,[512,512,1]), np.reshape(training_labels,[512,512,1])

