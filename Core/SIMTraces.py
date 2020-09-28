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
        self.map = []
        
        
        
        
      def GetTraceRestrictions(self):
        Entrez.email = ""
        
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
#        ReCutsInPx = (ReCuts*self.Stretch*self.BPSize)/self.PixelSize
        ReCutsInPx = msc.kbToPx(ReCuts,self)
          
        return ReCutsInPx
      
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


