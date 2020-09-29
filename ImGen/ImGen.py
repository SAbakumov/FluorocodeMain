# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:59:13 2020

@author: Boris
"""

# import tensorflow as tf
# import matplotlib as plt
# from tensorflow import keras
# from tensorflow.keras import layers
import numpy as np
import os
# from scipy.signal import convolve2d
from Core.SIMTraces import TSIMTraces
import Core.Misc as msc

class TrainImageGenerator(TSIMTraces):
    def __init__(self, TrainDir,NumImages,NumTracesPerImage,Resolution,Wavelength,NA,pixelsize):
        if not(os.path.exists(TrainDir)):            
            os.mkdir(TrainDir)   
        self.TrainDir = TrainDir
        self.NumImages = NumImages
        self.NumTracesPerImage = NumTracesPerImage
        self.Resolution = Resolution
        self.Wavelength = Wavelength
        self.NA         = NA
        self.pixelsize  = pixelsize
        
    # @classmethod
    # def from_super_instance(cls, TrainDir,NumImages,NumTracesPerImage,Resolution,Wavelength,NA, super_instance):
    #     return cls(TrainDir,NumImages,NumTracesPerImage,Resolution,Wavelength,NA, **super_instance.__dict__)

        
    def Generate(self, LabeledTraces):
        for image in range(0,self.NumImages):
            img = np.zeros([self.Resolution,self.Resolution])
            for tracenum in range(0,self.NumTracesPerImage):
                trace = LabeledTraces[tracenum]
                trace = trace-trace[0]
                
                GetInitialXPos = np.random.randint(-self.Resolution,self.Resolution-1)
                GetInitialYPos = np.random.randint(0,self.Resolution)
                
                trace = trace+GetInitialXPos
                InitIndex = np.argwhere(trace>=0)-1
                FinIndex  = np.argwhere(trace>self.Resolution)
                if len(InitIndex)==0:
                    InitIndex = 0
                else:
                    InitIndex = InitIndex.item(0)
                if len(FinIndex)==0:
                    FinIndex  = len(trace)-1
                else:    
                    FinIndex = FinIndex.item(0)-1
                    
                    

                
                Positions  = trace[InitIndex:FinIndex]
                for index in Positions:
                    if round(index)<self.Resolution:
                        img[int(round(index))][GetInitialYPos] = img[int(round(index))][GetInitialYPos]+1
                
                
                
        return img
    # def GetImage(self, img):
        
        
# Images =  TrainImageGenerator('D:\Sergey\TrainDirectory', 1, 50, 512,Wavelength =510 , NA = 1.4 )


# class LabelImageGenerator:
#     def __init__(self, TrainDir):
