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
import Core.Misc as msc
import tensorflow as tf

import random

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
        
    def LoadTrainingData(self,path):
        
        Data = np.load(path)
        # TrainImages = (Data['training_data'].astype(np.float32)/pow(2,16)).astype(np.float32)
        TrainImages = Data['training_data'].astype(np.float32)

        LabelImages = Data['training_labels'].astype(np.float32)
    
        
        return TrainImages, LabelImages
    
    def SaveTrainingData(self,training_data,training_labels,path):
        np.savez(path,training_data=training_data,training_labels=training_labels)
        
class DataConverter():
    
    def ToTensor(self,array):
        for item in  array:
            item  = tf.convert_to_tensor(item)
        return array
    
    
    def ToOneHot(self,img,numclass,numclasses):
        image  = tf.one_hot(img,numclasses)
        return image.numpy()
        
    def ToNPZ(self,imgArrays,numclasses):
        DataTensor = np.empty(tuple([len(imgArrays[0])*2+1]) +np.shape(imgArrays[0][0])).astype(np.int16)
        ShuffledArrays = []
        for numclass in range(0,numclasses):
            ClassArray = imgArrays[numclass]
            random.Random(452856).shuffle(ClassArray)
            ShuffledArrays.append(ClassArray)
            
        imgArrays = ShuffledArrays

        if len(np.shape(imgArrays[0][0]))!=3:     
            DataTensor = np.expand_dims(DataTensor,axis = 3)
        nextimg = 0
        for image in range(0, len(imgArrays[0])):
            print(image)
            for numclass in range(0,numclasses):
               ToAddImg = imgArrays[numclass][image]

               if len(np.shape(ToAddImg))!=3:
                   ToAddImg = imgArrays[numclass][image]*pow(2,16)/1000
                   ToAddImg =   np.expand_dims(ToAddImg, axis = 2)
               nextimg+=1
               DataTensor[nextimg,:,:,:] = ToAddImg.astype(np.int16)
               
               # if np.shape(DataTensor)==np.shape(ToAddImg):
               #     DataTensor = np.stack([DataTensor,ToAddImg.astype(np.int16) ],axis=0)
               # else:
               #     DataTensor = np.concatenate([DataTensor,np.expand_dims(ToAddImg.astype(np.int16),axis = 0)])

        return DataTensor
    
    def ToNPZ1D(self,tracearray,numclasses,datatype):
        totalNumElements = np.sum([len(x) for x in tracearray])
        
        if datatype=='data':
            DataTensor = np.empty([totalNumElements,len(tracearray[0][0]),1])
        if datatype=='label':
            DataTensor = np.empty([totalNumElements,len(tracearray[0][0])])



        AllTraces = []
        for Genome in tracearray:
            AllTraces.extend(Genome)
            
        random.Random(452856).shuffle(AllTraces)
        for i in range(0,len(AllTraces)):
            if datatype=='data':
                DataTensor[i,:,:] = np.reshape(AllTraces[i],[len(AllTraces[i]),1])
                
            if datatype=='label':
                DataTensor[i,:] = AllTraces[i]

            
        return DataTensor
        
        
        
        
        
    
                
            
            
            
            
        
        
        
        
        
        
    
    


# Dt = DataLoader()          
# Dt.PrepareTrainingData('D:\Vibrio Harveyi\FOVData\CroppedAndInverted')
# Dt.PrepareLabeledData('D:\Vibrio Harveyi\FOVData\Mask')
  