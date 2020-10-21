# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:37:06 2020

@author: Sergey
"""
import numpy as np
from Bio import Entrez
from Bio import SeqIO
import Core.Misc as msc
import Core.RandomTraceGenerator as RTG


class TSIMTraces:
      def __init__(self, Species, Stretch, BPSize, Optics,Enzyme,PixelSZ):
        self.Species = Species
        self.Stretch = Stretch
        self.BPSize = BPSize
        self.Optics = Optics
        self.Enzyme = Enzyme
        self.PixelSize = PixelSZ
        
        
        self.Trace = []
        self.RandomTraces = []
        self.Map = []
        
        
        
        
      def GetTraceRestrictions(self):
        Entrez.email = "abakumov.sergey1997@gmail.com"
        
        search_term = self.Species
        handle = Entrez.esearch(db='nucleotide', term=search_term) 
        record = Entrez.read(handle) 
        ids = record['IdList']
        
        
        handle = Entrez.efetch(db="sequences", id=ids,rettype="gb", retmode="text")
        genome = SeqIO.parse(handle, "gb")
        for record in genome:
            CompleteSequence = record.seq
            
        cuts = msc.rebasecuts(self.Enzyme,CompleteSequence )
        
        
        return cuts
    
    
    
    
    
      def GetDyeLocationsInPixel(self,ReCuts):
        ReCuts = np.array(ReCuts)
        ReCuts = ReCuts-ReCuts[0]        
        ReCutsInPx = msc.kbToPx(ReCuts,self)
          
        return ReCutsInPx
    
      def GetTraceProfile(self,trace,gauss,size,orrarr):
        x = np.zeros(size)
        trace = trace-np.min(trace)
        for i in range(0,len(trace)):
            try:
                x[int(np.round(trace.item(i)))] = x[int(np.round(trace.item(i)))]+1
            except:
                continue
            
        signal = np.convolve(x,gauss, mode = 'same')
        return signal
    
    
        
    
    
    
#      def GetRandomTraces(NumTraces,LabelRate,FPRate,AvLength,):
#        traces = Misc.stratsample(arr,avlength,sigmalength,numsamples)
#        for trace in range(0,NumTraces):
#            
#        
#          
#          
#          
#        return 
        
         
    
    
    
    
    
#    
#SIMTRC = TSIMTraces('CP000948',1.75,0.34,0,'TaqI',40)  
#
#cuts = SIMTRC.GetTraceRestrictions()
#ReCutsInPx = SIMTRC.GetDyeLocationsInPixel(cuts)
#R = RTG.RandomTraceGenerator(msc.kbToPx(40,SIMTRC),msc.kbToPx(5,SIMTRC),10000)


