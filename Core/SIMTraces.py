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
import random
import os


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
        
        
        ROOT_DIR = os.path.abspath(os.curdir)
        DataBasePath = os.path.join(ROOT_DIR, 'DataBases')
        search_term = self.Species
        
        FileName =  '%s.fasta' % search_term
        
        if not os.path.exists(os.path.join(DataBasePath,FileName )):
            handle = Entrez.efetch(db="nucleotide", id=search_term, rettype="fasta", retmode= 'text')
            
            f = open(os.path.join(DataBasePath,FileName ), 'w')
            f.write(handle.read())
            f.close()
        
        genome = SeqIO.parse(os.path.join(DataBasePath,FileName ), "fasta")


        
        # handle = Entrez.esearch(db='nucleotide', term=search_term, retmode="xml") 
        # record = Entrez.read(handle) 
        # ids = record['IdList']
        
        
        # handle = Entrez.efetch(db="nucleotide", id=ids,rettype="gb", retmode="text")
        # genome = SeqIO.parse(handle, "gb")
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
                x[int(np.round(trace.item(i)))] = x[int(np.round(trace.item(i)))]+1+random.uniform(-0.2,0.2)
            except:
                continue
            
        signal = np.convolve(x,gauss, mode = 'same')
        signal = msc.ZScoreTransform(signal)
        return signal
    
      def GetFluorocodeProfile(self,trace,gauss,size):
        signal = np.convolve(trace,gauss, mode = 'same')
        signal = msc.ZScoreTransform(signal)
        return signal
    
        
      def GetFullProfile(self,genome):
        genome = genome[0]
        Trace = np.zeros([int(np.round(np.max(genome)).item())])
        for i in range(0,len(genome)-1):
            try:
                pos = int(np.round(genome[i].item()+np.random.uniform(-1.5,1.5)))
                # pos = int(np.round(genome[i].item()))

                Trace[pos] =  Trace[pos]+1+np.random.uniform(-0.1,0.1)
            except:
                pos = int(np.round(genome[i].item()+np.random.uniform(-2,2)))

          
        return Trace
    
    
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


