# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:20:34 2020

@author: Sergey
"""

import numpy as np
import random


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
        
        
        LabelTransform = lambda x : np.sort(np.array(random.sample(list(x),int(LabelingEfficiency*len(x)))))
        EffLabeledTraces = list(map(LabelTransform, Traces))
        return EffLabeledTraces
        
    def GetCombinedRandomizedTraces(ListOfRandomTraces,numclasses):
        TotalCombinedList = []
        return TotalCombinedList
    
    
    # def GetRandomFixedLengthTraces(self,arr,length):
    #     AllTraces  = [];
    #     for i in range(0,self.numsamples):
    #         StartIndexOfTrace = np.random.choice(arr)
    #         EndIndexOfTrace = StartIndexOfTrace + length
            
    #         FirstInd =np.asscalar( np.argwhere(arr>=StartIndexOfTrace)[0])
    #         try:
    #             LastInd  =np.asscalar( np.argwhere(arr>=EndIndexOfTrace)[0])
    #         except:
    #             continue
            
    #         SubTrace = arr[FirstInd:LastInd]
    #         AllTraces.append(SubTrace)
                
                
     
    #     return AllTraces

            
    def GetRandomFixedLengthTraces(self,arr,length):
      AllTraces  = [];
      
      # step = np.max(arr)/self.numsamples
      step = np.max(arr)/self.numsamples

      for pos in range(0,int(np.round( np.max(arr))),int(step)):
           StartIndexOfTrace = pos
           
           # StartIndexOfTrace = np.random.choice(arr)
           EndIndexOfTrace = StartIndexOfTrace + length
           
           FirstInd =np.asscalar( np.argwhere(arr>=StartIndexOfTrace)[0])
           try:
               LastInd  =np.asscalar( np.argwhere(arr>=EndIndexOfTrace)[0])
           except:
               continue

          
           SubTrace = arr[FirstInd:LastInd]
           AllTraces.append(SubTrace)
              
              
   
      return AllTraces    

    def GetRandomTraces(self,maxNumDyes,minNumDyes,length,numsamples):
      RandomTraces = []
      for sample in range(0, numsamples):
          numDyes =  random.randint(minNumDyes, maxNumDyes) 
          trace   =  np.zeros([length,1])
          for i  in range(0, numDyes):
              pos = random.randint(0,length-1)
              trace[pos] = trace[pos] + 1 +random.uniform(-0.2,0.2)
          RandomTraces.append(trace)
          
      return RandomTraces
      
         
                            
                
        
            


        