# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:20:34 2020

@author: Sergey
"""

import numpy as np



class RandomTraceGenerator:
    def __init__(self,avlength,sigmalength,numsamples) :
        self.avlength = avlength
        self.sigmalength = sigmalength
        self.numsamples = numsamples
        
        
    def stratsample(self,arr):
        
        AllTraces  = [];
        for i in range(0, int(round(max(arr)-round(3*self.avlength))),round(3*self.avlength)):
            FirstInd = np.asscalar(np.argwhere(arr>=i)[0])
            LastInd  = np.asscalar(np.argwhere(arr>=i+round(3*self.avlength))[0])
            SubArray = arr[FirstInd:LastInd]
            for j in range(0,round(self.numsamples/(max(arr)/(3*self.avlength))).astype(int)):
                StartIndexOfTrace = np.random.choice(SubArray)
                EndIndexOfTrace = StartIndexOfTrace + np.random.normal(loc = self.avlength,scale = self.sigmalength)
                
                FirstInd =np.asscalar( np.argwhere(arr>=StartIndexOfTrace)[0])
                try:
                    LastInd  =np.asscalar( np.argwhere(arr>=EndIndexOfTrace)[0])
                except:
                    LastInd = len(arr)-1
    
        
                SubTrace = arr[FirstInd:LastInd]
                AllTraces.append(SubTrace)
                
                
     
        return AllTraces
    
    def GetEffLabelingRate(self,Traces,LabelingEfficiency):
        LabelTransform = lambda x : np.sort(np.random.choice(x,int(LabelingEfficiency*len(x))))
        EffLabeledTraces = list(map(LabelTransform, Traces))
        return EffLabeledTraces
        
        