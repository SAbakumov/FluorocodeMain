# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:58:18 2020

@author: Boris
"""

"""
Created on Sun Sep 27 19:59:10 2020

@author: Sergey
"""
import Core.Misc as Misc
import Core.SIMTraces as SIMTraces
import Core.RandomTraceGenerator as RTG
import numpy as np
# from ImGen.ImGen import TrainImageGenerator
from Core.DataHandler import DataConverter
from Core.DataHandler import DataLoader
from Core.Noise import perlin
import matplotlib.pyplot as plt

Wavelength = 576
NA = 1.4
size =250
pixelsz = 39.68
ResEnhancement=2
# genomes = ['Other','CP014051','CP039296','CP000948']
# genomes = ['NC_000913.3','Other','CP014051','CP014787']
genomes = ['NC_000913.3','CP014051','CP014787','NZ_CP009467.1']

# genomes = ['CP014787']
# genomes = ['CP000948']
# genomes = ['CP014051','CP039296', 'CP009467.1','CP014787','CP000948']

AllLabels = []
AllProfiles = []
numsamples =5000
    
Dt = DataConverter()
Ds = DataLoader()



for genome in genomes:
    EffLabeledTraces = []
    SIMTRC     = SIMTraces.TSIMTraces(genome,1.66,0.34,0,'TaqI',pixelsz )  
    Gauss  = Misc.GetGauss1d(size,  Misc.FWHMtoSigma(Misc.GetFWHM(Wavelength,NA,ResEnhancement)),pixelsz)

    if genome!='Other':

        Profiles = []
        ReCuts     = SIMTRC.GetTraceRestrictions()
        ReCutsInPx = SIMTRC.GetDyeLocationsInPixel(ReCuts)
        R      = RTG.RandomTraceGenerator(Misc.kbToPx(100000,SIMTRC),Misc.kbToPx(20000,SIMTRC),numsamples)
    else:
        R      = RTG.RandomTraceGenerator([],[],numsamples)
        
        Profiles = []
        
        
        
        
        
        
        
    if genome!='Other':
        AllTrc = [];
        for transform in range(0,50):
            print(transform)
            FullTrace = SIMTRC.GetFullProfile(R.GetEffLabelingRate([ReCutsInPx ],np.random.uniform(0.75,1)))
            FullTraceProfile = SIMTRC.GetFluorocodeProfile(FullTrace,Gauss,FullTrace.shape[0])
            start = 0
            for sample in range(0,numsamples):
                step =int(np.ceil( len(FullTraceProfile)/numsamples))
                try:
                    if start==0:
                        trc = FullTraceProfile[30419-350:30419-100]
                        AllTrc.append(trc)
                    EffLabeledTraces.append(FullTraceProfile[start:start+size]+np.random.uniform(0.1,0.35)* Misc.GetPerlinNoise(size))
                    
                    start = start+step
                except:
                    start = start+step


    else:
        Traces = R.GetRandomTraces(85,30,380,80*numsamples)
        for trace in Traces:

            EffLabeledTraces.append(trace)

        # EffLabeledTraces.append(np.array(trace))
    
    

    
    
    progress = 0
    for x in EffLabeledTraces:  
        if genome!='Other':
            Profiles.append(x) 
            Profiles.append(np.flipud(x))
        else:
            prf = SIMTRC.GetFluorocodeProfile(np.squeeze(x),Gauss,size)
            Profiles.append(prf) 


        progress+=1
        print('\r'+ str(progress) + " from " + str(len(EffLabeledTraces)) + " done", end = ""  )
        
        
    
    Label   = np.zeros(len(genomes))
    Label[genomes.index(genome)] = 1
    Labels  =  [ Label for x in Profiles]
    
    AllLabels.append(Labels)
    AllProfiles.append(Profiles)
    
    
Profiles= Dt.ToNPZ1D(AllProfiles,2,datatype='data')
Labels= Dt.ToNPZ1D(AllLabels,2,datatype='label')

print("Saving...")
Ds.SaveTrainingData(Profiles,Labels ,path="D:\Sergey\FluorocodeMain\FluorocodeMain\DataForValidation1D.npz")    
    

    
    
    
    

    
    
    
    
    
    
    
    
    
    
