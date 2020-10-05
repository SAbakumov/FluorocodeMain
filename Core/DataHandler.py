# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:23:08 2020

@author: Boris
"""

from Bio import Restriction
import Core.SIMTraces
import numpy as np
import os
import tifffile as tiff
import cv2
import Misc as msc
import tensorflow as tf

class DataLoader():
    def __init__(self):
        self.TrainingImages = []
        self.LabelImages = []
    
    def PrepareTrainingData(self,folder):
        print('Loading Training images from folder ' + folder)
        TrainingImages = []
        Files = os.listdir(folder)
        prog = 0
        for file in Files:
            prog = prog+1
            img = DataLoader.ReturnImage(os.path.join(folder,file))
            TrainingImages.append((msc.GetFilename(file), np.reshape( img,[img.shape[0],img.shape[1],1]))) 
            if prog % 500 == 0:
                print(str(prog) + ' out of ' + str(len(Files))+ ' done')
                
                
        self.TrainingImages = TrainingImages
        return TrainingImages
            
    def ReturnImage(directory):
        img= cv2.imread(directory,-1)

        return img
         
            
    def PrepareLabeledData(self, folder):
        print('Loading corresponding label images from folder ' + folder)
        LabelImages = []
        Files = os.listdir(folder)
        prog = 0

        
        
        for TrainImage in self.TrainingImages:
            Item = [value for value in Files  if  TrainImage[0] in value]
            prog = prog+1
            img =DataLoader.ReturnImage(os.path.join(folder,Item[0]))
            img = np.sum(img,axis=2)
            img[~(img==765)] = 1
            img[(img==765)] = 0   
            LabelImages.append((Item, np.reshape(img,[img.shape[0],img.shape[1],1]))) 
            if prog % 500 == 0:
                print(str(prog) + ' out of ' + str(len(Files))+ ' done')
            
        self.LabelImages = LabelImages
        return LabelImages
        

    

        
class DataConverter():
    
    def ToOneHot(self,img,numclass,numclasses):
        image  = tf.one_hot(img,numclasses)
        return image
        
    # def ToNPZ(imgArray):
        
        
        
        
        
    
    


# Dt = DataLoader()          
# Dt.PrepareTrainingData('D:\Vibrio Harveyi\FOVData\CroppedAndInverted')
# Dt.PrepareLabeledData('D:\Vibrio Harveyi\FOVData\Mask')
  